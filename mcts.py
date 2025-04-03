# https://github.com/JoshVarty/AlphaZeroSimple/blob/master/monte_carlo_tree_search.py
# https://joshvarty.github.io/AlphaZero/
# https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
# https://jonathan-hui.medium.com/monte-carlo-tree-search-mcts-in-alphago-zero-8a403588276a
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from rl_utils import *
from models import Shobu_MCTS, HISTORY_SIZE

from shobu import ShobuMove, Shobu, Player

MAX_GAME_LEN = 128

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

    def selection(self):
        """
        Computes argmax_a (Q(s,a) + u(s,a)). See wikipedia link for "selection"
        and "expansion" terminology definitions
        """
        max_a = -np.inf
        a_move, a_t = None, None
        for move, child in self.children.items():
            if (score := self.ucb(child)) > max_a:
                max_a = score
                a_move, a_t = move, child
        return a_move, a_t

    def expansion(self, candidate_moves: dict[ShobuMove, float]):
        for move, probability in candidate_moves.items():
            self.children[move] = MCNode(probability, Player(not self.player))
        self.is_expanded = True

class MCTree:
    # init
    ## model (policy + value)
    def __init__(self, model: Shobu_MCTS, starting_state: Shobu, device: torch.device):
        self.root = MCNode(0)
        self.root.state = starting_state
        self.model = model
        self.device = device

    def simulation(self):
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
            recent_history = path_to_leaf[-HISTORY_SIZE:] # not necessarily 8 elts long!!
            past_boards = []
            for node in recent_history:
                if node.state.next_mover == Player.WHITE:
                    past_boards.append(node.state.copy().flip().as_matrix())
                else:
                    past_boards.append(node.state.as_matrix())
            past_boards = [np.zeros((8,4,4)) for _ in range(HISTORY_SIZE-len(past_boards))] + past_boards
            state_tensor = torch.from_numpy(np.concatenate(past_boards)).to(self.device)
            ## value evaluates leaves
            with torch.no_grad():
                evaluation = self.model.get_value(state_tensor)
                policy_output = self.model.get_policy(state_tensor)
                move_to_probability = get_joint_logits(cur_state, policy_output)
            cur_node.expansion(move_to_probability)

        self.backprop(...)

    def search(num_simulations: int) -> MCNode:
        pass

    def backprop():
        ...


class Shobu_MCTS:
    # init
    def __init__(self, model: Shobu_MCTS, device: torch.device):
        super().__init__()
        # init models
        self.model = model
        # init game
        self.init_game()
        # parameters
        self.epochs = 200000
        self.minibatch_size = 512
        self.device = device


    def init_game(self):
        '''
        Initializes MCTS
        '''
        board = Shobu.starting_position()
        self.MCTS = MCTree(self.model, board, self.device)    

    # train
    def train(self, opt, scheduler):
        self.model.train()
        loss_list = []
        policy_loss_list = []
        value_loss_list = []
        reward_list = []

        for epoch in self.epochs:
            # get memory
            rollout = self.MCTS.search()
            rollout.shuffle()
            batch_policy_loss_list = []
            batch_value_loss_list = []
            batch_loss = []
            batch_reward = []
            while rollout:
                transitions = rollout.sample(self.minibatch_size)
                batch = Transition_MCTS(*zip(*transitions))
                boards = torch.stack(batch.board)
                states = torch.stack(batch.state)
                rewards = torch.stack(batch.reward)
                ucb = torch.stack(batch.ucb)

                ### value loss ###
                values = self.model.value_function(states)
                value_loss = F.mse_loss(values, rewards)

                ### policy loss ###
                policy_outputs = self.model.get_policy(states)
                # get distributions
                policy_distributions = []
                for po, board in zip(policy_outputs,boards):
                    move_to_probability = get_joint_logits(board, po)
                    distribution = torch.stack(list(move_to_probability.values()))
                    policy_distributions.append(distribution)
                policy_distributions = torch.stack(policy_distributions)
                policy_loss = torch.sum(-ucb * torch.log(policy_distributions), dim=1).mean()

                ### optimize ###
                loss = value_loss + policy_loss
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                opt.step()
                scheduler.step()

                ### metrics ###
                batch_policy_loss_list.append(policy_loss.item())
                batch_value_loss_list.append(value_loss.item())
                batch_loss.append(loss.item())
                batch_reward.append(np.mean(rewards.clone().cpu().numpy()))
        
        ### update metrics ###
        avg_epoch_policyloss = np.mean(batch_policy_loss_list)
        avg_epoch_value_loss = np.mean(batch_value_loss_list)
        avg_epoch_loss = np.mean(batch_loss)
        avg_epoch_reward = np.mean(batch_reward)
        policy_loss_list.append(avg_epoch_policyloss)
        value_loss_list.append(avg_epoch_value_loss)
        loss_list.append(avg_epoch_loss)
        reward_list.append(avg_epoch_reward)

        ### update plot ###
        plot_progress_MCTS(reward_list, loss_list, policy_loss_list, value_loss_list)

 
if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    model = Shobu_MCTS()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    shobu_mcts = Shobu_MCTS(model, device)
    shobu_mcts.train(optimizer, scheduler)


