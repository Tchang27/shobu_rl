import cProfile
import copy
from collections import deque
from datetime import datetime
from torch.multiprocessing import set_sharing_strategy, set_start_method, Queue
import torch.multiprocessing as mp
from rl_utils import ReplayMemory_MCTS, Transition_MCTS
import numpy as np
import torch
import time
import torch.nn.functional as F
import torch.nn as nn
from rl_utils import *
from models import Shobu_MCTS, HISTORY_SIZE
import gc
import psutil
from shobu import ShobuMove, Shobu, Player
import random

MAX_GAME_LEN = 256


class MCNode:
    def __init__(self, prior: float, player: Player):
        self.num_visits = 0
        self.total_reward = 0 # We want the average reward over time, total_reward / num_visits, so we save both separately
        self.children = {}
        self.board_state = None
        self.prior = prior
        self.is_expanded = False
        self.state: Shobu = None
        self.player = player

    def ucb(self, child: "MCNode") -> float:
        """
        Computes UCB score for one of your children
        i.e. Q(s,a) + u(s,a) where
        Q = average reward, u ~= prior / (1 + visit count)
        """

        # exploitation bonus. We negate total_reward because that reward is from
        # opponent POV (high reward for them is bad for us)
        q = 0 if child.num_visits == 0 else -child.total_reward / child.num_visits
        # exploration bonus
        u = child.prior * np.sqrt(self.num_visits) / (1 + child.num_visits)
        return q + u

    def selection(self) -> tuple[ShobuMove, "MCNode"]:
        """
        Computes argmax_a (Q(s,a) + u(s,a)). See wikipedia link for "selection"
        and "expansion" terminology definitions
        """
        moves = list(self.children.keys())
        children = list(self.children.values())

        # vist and reward vector
        visits = np.array([c.num_visits for c in children], dtype=np.float32)
        rewards = np.array([-c.total_reward for c in children], dtype=np.float32)

        q_values = np.divide(rewards, visits, out=np.zeros_like(rewards), where=visits!=0)

        # exploration bonus
        priors = np.array([c.prior for c in children], dtype=np.float32)
        sqrt_parent_visits = np.sqrt(self.num_visits)
        exploration = priors * sqrt_parent_visits / (1 + visits)

        # UCB scores
        ucb_scores = q_values + exploration

        # Find max
        max_idx = np.argmax(ucb_scores)
        return moves[max_idx], children[max_idx]

    def expansion(self, candidate_moves: dict[ShobuMove, torch.tensor]):
        for move, probability in candidate_moves.items():
            self.children[move] = MCNode(probability.item(), Player(not self.player))
        self.is_expanded = True

    def sample_move(self, tau: float) -> ShobuMove:
        # Useful arrays:
        potential_moves = list(self.children.keys())
        children_visit_counts = np.array([child.num_visits for child in self.children.values()])

        # If tau is 0, pick the most visited move
        if tau == 0:
            chosen_move = potential_moves[np.argmax(children_visit_counts)]
        
        # If tau is infinity, just choose randomly
        elif tau == float('inf'):
            chosen_move = np.random.choice(potential_moves)

        # Otherwise, sample with temperature-scaled probabilities
        else:
            visit_counts_scaled = children_visit_counts ** (1 / tau)
            visit_counts_probs = visit_counts_scaled / sum(visit_counts_scaled)
            chosen_move = np.random.choice(potential_moves, p=visit_counts_probs)

        return chosen_move


class MCTree:
    # init
    ## model (policy + value)
    def __init__(self, model: Shobu_MCTS, starting_state: Shobu, device: torch.device):
        self.root = MCNode(0, starting_state.next_mover)
        self.root.state = starting_state
        self.model = model
        self.device = device
        
    def __del__(self):
        self.root = None
        # Explicitly delete large objects
        if hasattr(self, 'model'):
            del self.model

    def _value_and_policy(self, path: list[MCNode], noise=True) -> tuple[float, dict[ShobuMove, torch.tensor]]:
        recent_history = path[-HISTORY_SIZE:] # not necessarily 8 elts long!!
#         past_boards = []
#         for node in recent_history:
#             # if node.state.next_mover == Player.WHITE:
#             #     flipped = node.state.copy()
#             #     flipped.flip()
#             #     past_boards.append(flipped.as_matrix())
#             # else:
#             #     past_boards.append(node.state.as_matrix())
#             assert node.state.next_mover == Player.BLACK
#             past_boards.append(node.state.as_matrix())
#         past_boards = [np.zeros((8,4,4)) for _ in range(HISTORY_SIZE-len(past_boards))] + past_boards
        past_boards = [recent_history[-1].state.as_matrix()]
        state_tensor = torch.tensor(np.concatenate(past_boards), device=self.device, dtype=torch.float32).unsqueeze(0)
        ## value evaluates leaves
        
        with torch.no_grad():
            output = self.model(state_tensor)
            evaluation = output['q_value'].item()
            move_to_probability = get_joint_logits(path[-1].state, output, noise=noise)
        return evaluation, move_to_probability

    def simulation(self, noise=True):
        cur_node = self.root
        path_to_leaf = [cur_node]
        while cur_node.is_expanded:
            move, cur_node = cur_node.selection()
            path_to_leaf.append(cur_node)

        # Now we are at a MCNode which is not yet expanded. Compute the board
        # state at this node and get static eval
        parent_node = path_to_leaf[-2]
        cur_state = parent_node.state.apply(move)
        cur_state.flip()
        cur_node.state = cur_state

        if (winner := cur_state.check_winner()) is not None:
            if cur_node.player == winner:
                evaluation = 1
            else:
                evaluation = -1
        elif len(path_to_leaf) >= MAX_GAME_LEN:
            evaluation = 0
        else:
            # get past 8 board states.
            evaluation, move_to_probability = self._value_and_policy(path_to_leaf, noise)
            cur_node.expansion(move_to_probability)

        self.backprop(evaluation, path_to_leaf, cur_node.player) # use `evaluation` in backprop

    def search(self, num_simulations: int, noise=True) -> MCNode:
        _, move_to_probability = self._value_and_policy([self.root], noise)
        # TODO: add dirichlet noise to move_to_probability
        self.root.expansion(move_to_probability)
        for _ in range(num_simulations):
            self.simulation(noise)
        return self.root

    def backprop(self, evaluation: float, path_to_leaf: list[MCNode], cur_player: Player):
        # For each node on the path back up to the root:
        for node in reversed(path_to_leaf):
            # increment value by the found reward  # increment visits by 1
            node.num_visits += 1
            if node.player == cur_player:
                node.total_reward += evaluation
            else:
                node.total_reward -= evaluation


#### WORKER FUNCTIONS ####
def temperature_scheduler(epoch_no, move_no):
    """
    this is just a straight up guess, can try something diff too
    but generally temperature should decrease over time
    """
    if move_no < 30:
        return 1
    elif move_no > 60:
        return 0
    else:
        return (60 - move_no) / 30

    
def play_game(model, device, memory: ReplayMemory_MCTS, epoch: int):    
    board = Shobu.starting_position()
    generated_training_data = []
    num_moves = 0
    game_end_reward = None
    while True:

        ######
        # PERFORMANCE PROFILING
        ###
        # pr = cProfile.Profile()
        # pr.enable()
        mcts = MCTree(model, board, device)
        # playout randomization
        full_search = False
        if np.random.random() < 0.75:
            rollout = mcts.search(100, noise=False)
        else:
            rollout = mcts.search(600, noise=True)
            full_search = True
        del mcts
        _sum_pi = sum([child.num_visits for child in rollout.children.values()])

        # This is what policy should learn
        pi = {}
        for k in rollout.children.keys():
            pi[k] = rollout.children[k].num_visits / _sum_pi
        # Only append full search
        generated_training_data.append((board, pi, full_search))

        # choose the next move based on tree search results
        tau = temperature_scheduler(epoch, num_moves)
        selected_move = rollout.sample_move(tau)
        board = board.apply(selected_move)

        # Check winner
        if (winner := board.check_winner()) is not None:
            assert winner == Player.BLACK
            game_end_reward = 1
            break
        elif num_moves >= MAX_GAME_LEN:
            game_end_reward = 0
            break

        # next player also plays black
        board.flip()
        num_moves += 1
        
        gc.collect()

        ###
        # END PROFILING
        ######
        # pr.disable()
        # pr.dump_stats(datetime.today().strftime('%Y-%m-%d-%H:%M:%S') + f"-epoch:{epoch}-move:{num_moves}.dmp")

    # last element of generated_training_data, generated_training_data[-1],
    # is the game state right before the winning move. so that "player"
    # has game reward 1. the reward for preceding board states thus alternates
    # between -1 and 1. This "label" of -1 or 1 will be used to train the
    # value net
    generated_training_data_with_rewards = []
    for board, pi, full_search in reversed(generated_training_data):
        if full_search:
            generated_training_data_with_rewards.append((board, pi, game_end_reward))
        game_end_reward *= -1

    # Push entire game history into ReplayMemory
    history = deque()
    for board, pi, reward in generated_training_data_with_rewards:
        history.append(board.as_matrix())
        padded = [np.zeros((8,4,4)) for _ in range(HISTORY_SIZE-len(history))] + list(history)
        state_tensor = torch.tensor(np.concatenate(padded), device=device, dtype=torch.float32).unsqueeze(0)
        memory.push(board, state_tensor, reward, pi)

        if len(history) >= HISTORY_SIZE:
            history.popleft()
            
            
#### MULTIPROC TRAINING+SIMUL ####            
MINIBATCH_SIZE = 256
POOL_SIZE = 30
TRAINER_SIZE = 1
WINDOW_SIZE = 50000 # TODO tune this
WARMUP = 5000      
        
# worker process for simulating game        
def pickled_play_game(shared_model, buffer, lock, device, seed):
    # to avoid file descriptor issues
    set_sharing_strategy('file_system')
    torch.set_num_threads(1)
        
    torch.manual_seed(seed)
    np.random.seed(seed)
    local_model = Shobu_MCTS(device)
    local_model.load_state_dict(shared_model.state_dict())
    local_model.eval()
    episode = 0
    while True:
        try:
            with lock:
                snapshot = shared_model.state_dict()
                local_model.load_state_dict(snapshot)
            memory = ReplayMemory_MCTS()
            play_game(local_model, device, memory, episode)
            with lock:
                for m in memory.memory:
                    buffer.put(m) 
            del memory
            episode += 1 
        except Exception as e:
            print(f"[Worker error]: {e}")
        gc.collect()
            

# class for rl training
class Shobu_MCTS_RL:
    # init
    def __init__(self, device: torch.device):
        super().__init__()
        # parameters
        self.device = device                
    
    
    def train_loop(self, model, buffer, lock):
        set_sharing_strategy('file_system')
        torch.set_num_threads(TRAINER_SIZE)
        model.train()
        
        # local buffer
        local_buffer = deque(maxlen=WINDOW_SIZE)
        
        # metrics
        loss_list = []
        policy_loss_list = []
        value_loss_list = []
        reward_list = []
        
        # optimizer
        critic_params = list(model.critic.parameters())
        backbone_params = list(model.backbone.parameters())
        actor_params = [
            p for p in model.parameters() 
            if (not any(p is cp for cp in critic_params)) and (not any(p is bp for bp in backbone_params))
        ]  # All other params (policy heads)
        opt = torch.optim.Adam([
            {'params': actor_params, 'lr': 2e-5},
            {'params': backbone_params, 'lr': 2e-5},
            {'params': critic_params, 'lr': 2e-5}
        ], amsgrad=True, weight_decay=3e-5)
        
        tot1 = time.time()
        # warm up period
        batch_size = MINIBATCH_SIZE
        while buffer.qsize() < WARMUP:
            print(f"Warming up... ({buffer.qsize()})")
            time.sleep(10)
        
        # train step
        epoch = 0
        while True:
            # empty simulations
            with lock:
                while buffer.qsize() > 0:
                    local_buffer.append(buffer.get())
                
            print(f"Rolling window size: {len(local_buffer)}")
            t0 = time.time()
            # Randomly sample a batch of items
            minibatch = random.sample(local_buffer, batch_size)
            if not minibatch:
                print("Minibatch was empty, skipping train step.")
                time.sleep(1)
                continue
            batch = Transition_MCTS(*zip(*minibatch))
            boards = batch.board
            states = torch.concatenate(batch.state)
            rewards = torch.tensor(np.stack(batch.reward), device=self.device, dtype=torch.float32)
            mcts_dist = np.stack(batch.mcts_dist)

            ### model output ###
            output = model(states)
            
            ### value loss ###
            values = output['q_value'].squeeze()
            value_loss = F.mse_loss(values, rewards)

            ### policy loss ###
            # get distributions
            policy_losses = []
            policy_outputs = model.get_policy(states)
            p_pos = output["passive"]["position"]
            p_dir = output["passive"]["direction"]
            p_dist = output["passive"]["distance"]
            a_pos = output["aggressive"]["position"]
            i = 0
            for board, pi_dict in zip(boards, mcts_dist):
                po = {
                    "passive": {
                        "position": p_pos[i],
                        "direction": p_dir[i],
                        "distance": p_dist[i],
                    },
                    "aggressive": {
                        "position": a_pos[i]
                    }
                }
                move_to_logit = get_joint_logits(board, po, logits=True)
                policy = torch.stack([move_to_logit[k] for k in move_to_logit.keys()])
                pi_dist = torch.tensor([pi_dict[k] for k in pi_dict.keys()], device=self.device, dtype=torch.float32)
                policy_losses.append(F.kl_div(F.log_softmax(policy, dim=-1), pi_dist, reduction="sum"))
                i += 1
            policy_loss = torch.mean(torch.stack(policy_losses))

            ### optimize ###
            loss = value_loss + policy_loss
            opt.zero_grad()
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            # prevent workers from loading weird state dicts
            with lock:
                opt.step()

            ### metrics ###
            policy_loss_list.append(policy_loss.item())
            value_loss_list.append(value_loss.item())
            loss_list.append(loss.item())
            reward_list.append(np.mean(rewards.clone().cpu().numpy()))


            ### update plot ###
            t1 = time.time()
            plot_progress_MCTS(reward_list, loss_list, policy_loss_list, value_loss_list, epoch)
            print(f"Train step time: {t1-t0}")
            print(f"Total time elapsed: {t1-tot1}")
            print(f"Trained on batch, loss = {loss:.4f}")
            print(f"Total grad norm: {total_norm}")
            grad_norms = []
            for param in critic_params:
                if param.grad is not None:  # Ensure that gradient exists
                    grad_norm = param.grad.norm()  # Compute the L2 norm of the gradient
                    grad_norms.append(grad_norm.item())  # Convert to Python float for easier logging
            total_grad_norm = sum(grad_norms)
            print(f"Total grad norm for value head (critic): {total_grad_norm}")
            grad_norms = []
            for param in actor_params:
                if param.grad is not None:  # Ensure that gradient exists
                    grad_norm = param.grad.norm()  # Compute the L2 norm of the gradient
                    grad_norms.append(grad_norm.item())  # Convert to Python float for easier logging
            total_grad_norm = sum(grad_norms)
            print(f"Total grad norm for policy head (actor): {total_grad_norm}")
            grad_norms = []
            for param in backbone_params:
                if param.grad is not None:  # Ensure that gradient exists
                    grad_norm = param.grad.norm()  # Compute the L2 norm of the gradient
                    grad_norms.append(grad_norm.item())  # Convert to Python float for easier logging
            total_grad_norm = sum(grad_norms)
            print(f"Total grad norm for backbone: {total_grad_norm}")
            print(values)
            print(rewards)
            

            # save checkpoints
            if ((epoch+1)%100) == 0:
                torch.save(model.state_dict(), f'mcts_checkpoints_696/mcts_checkpoint_{epoch+3801}.pth')

            # garbage collect
            gc.collect()

            # update epoch
            epoch += 1

        # save final model
        torch.save(model.state_dict(), 'mcts_checkpoints_696/mcts_final.pth')
            
                
    # simultaneous simulation and training
    def train(self):
        # multiproc setting for torch
        set_sharing_strategy('file_system')
        set_start_method('spawn', force=True)
        mp_ctx = mp.get_context('spawn')
        manager = mp_ctx.Manager()
        lock = mp_ctx.Lock()
        
        # Get total CPU count
        available_cpus = psutil.Process().cpu_affinity()
        
        # Assign CPU cores - first half to workers, second half to training
        simulation_cores = available_cpus[:POOL_SIZE]
        training_cores = available_cpus[POOL_SIZE:POOL_SIZE+TRAINER_SIZE]
        
        print(f"Num available cores: {len(available_cpus)}")
        print(f"Simulation cores: {simulation_cores}")
        print(f"Training cores: {training_cores}")
        
        # Set affinity for main process (training)
        main_process = psutil.Process(os.getpid())
        main_process.cpu_affinity(training_cores)
        
        # model
        model = Shobu_MCTS(self.device)
        model.to(self.device)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # load from previous checkpoint
        model.load_state_dict(torch.load(f'mcts_checkpoints_696/mcts_checkpoint_{3800}.pth', map_location=self.device))
        # share model memory
        model.share_memory()
                
        # buffer
        sim_to_train_queue = Queue(maxsize=WINDOW_SIZE)

        workers = []
        for i in range(POOL_SIZE):
            p = mp_ctx.Process(target=pickled_play_game, args=(model, sim_to_train_queue, lock, self.device, 42 + i))
            p.daemon = True
            p.start()
            try:
                worker_process = psutil.Process(p.pid)
                worker_process.cpu_affinity([simulation_cores[i]])
                print(f"Worker {i} pinned to core {simulation_cores[i]}")
            except Exception as e:
                print(f"Could not set affinity for worker {i}: {e}")
            workers.append(p)
        
         # Set main process (training) affinity
        try:
            main_process = psutil.Process(os.getpid())
            main_process.cpu_affinity(training_cores)
            print(f"Training process pinned to cores {training_cores}")
        except Exception as e:
            print(f"Could not set affinity for training process: {e}")
        
        try:
            # Run the training loop
            self.train_loop(model, sim_to_train_queue, lock)
        finally:
            # Clean shutdown
            for w in workers:
                w.terminate()
                w.join()
    

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    model = Shobu_MCTS(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    shobu_mcts = Shobu_MCTS_RL(model, device)
    shobu_mcts.train(optimizer, scheduler)