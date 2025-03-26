from shobu import *
from rl_utils import *
from models import *
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import random
from collections import deque
import time
from itertools import combinations

# hyperparameters
# training parameters
MAX_TURNS = 60
MAX_GAMES = 50000
BATCH_SIZE = 256
EPOCHS = 4
ON_POLICY = 20


# Value function loss weight
C1 = 0.5
# Entropy loss weight
C2 = 0.05
# Clipped PPO parameter
EPSILON = 0.2
# Discount factor parameters
GAMMA = 0.95
LAMBDA = 0.9
# size of replay buffer
MEMORY_SIZE = 500000
# win reward
WIN_REWARD = 20
# add new opponent every UPDATE_OPPS epochs
UPDATE_OPPS = 1000
# max number of opponents to track at a given time
OPP_QUEUE_LENGTH = 6
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)


class Shobu_RL(Shobu):
    def __init__(self, ppo_model: nn.Module):
        super().__init__()
        # init models
        self.set_model(ppo_model)
        # steps throughout training
        self.train_steps = 0
        # player color - always plays as black
        self.player_color = Player.BLACK
        # init game
        self.init_game()

        
    def init_game(self):
        # steps in a game
        self.steps_done = 0
        # Shobu board
        self.board = Shobu.starting_position()


    def set_model(self, ppo_model) -> None:
        '''
        Set models for training

        Input:
        - Value model
        - Target model

        Return: None
        '''
        self.ppo_model = ppo_model
        # opponents - periodically freeze value model and add to list
        self.opponent_pool = deque([], maxlen=OPP_QUEUE_LENGTH)
        opponent = copy.deepcopy(ppo_model) 
        for param in opponent.parameters():
            param.requires_grad = False
        self.opponent_pool.append(opponent)

        
    def remove_opponent(self):
        """ 
        Removes an opponent from the pool:
        - 75% chance: Remove oldest (popleft)
        - 25% chance: Remove mid-tier opponent
        """
        if len(self.opponent_pool) < OPP_QUEUE_LENGTH:
            # Don't remove if queue not full
            return

        # Random removal strategy
        if random.random() < 0.75:
            # 75% chance remove oldest
            removed = self.opponent_pool.popleft()
        else:
            # 25% chance remove a mid-tier opponent
            mid_index = random.randint(2, min(len(self.opponent_pool) - 1, 5))
            removed = self.opponent_pool[mid_index]
            del self.opponent_pool[mid_index]
    
    
    def passive_logits(self, policy_output):
        '''
        From the policy model, extract passive move logits
        '''
        # Combine passive probabilities into joint distribution
        p_pos = policy_output["passive"]["position"]
        p_dir = policy_output["passive"]["direction"]
        p_dist = policy_output["passive"]["distance"]

        # Create passive joint distribution tensor
        passive_logits = p_pos.unsqueeze(-1).unsqueeze(-1) * p_dir.unsqueeze(1).unsqueeze(-1) * p_dist.unsqueeze(1).unsqueeze(1)
        # Flatten the passive move probabilities
        passive_logits = passive_logits.flatten(start_dim=1)
        return passive_logits
              
        
    def aggressive_logits(self, policy_output):
        '''
        From the policy model, extract aggressive move logits
        '''
        # Extract aggressive probabilities
        a_pos = policy_output["aggressive"]["position"]
        
        aggressive_logits = a_pos
        return aggressive_logits
        
    
    def sample_passive(self, policy_output, valid_moves):
        '''
        From the policy model, sample passive action
        '''
        # Flatten the passive move probabilities
        passive_logits = self.passive_logits(policy_output).squeeze()
            
        # create mask
        mask = torch.zeros(passive_logits.shape, device=device, dtype=torch.float32)
        for move in valid_moves:
            px, py, pd, ps = move[0]
            pfrom = (px*8) + py 
            pidx = (pfrom * (8 * 2)) + (pd * 2) + (ps-1)
            mask[pidx] = 1
            
        masked_probs = torch.softmax(passive_logits, dim=-1)*mask
        masked_log_probs = torch.log_softmax(passive_logits, dim=-1)*mask
        # Sample passive move
        passive_index = torch.multinomial(masked_probs, 1).item()

        # Decode the sampled passive move
        p = (passive_index // (8 * 2)) % 64
        d = (passive_index // 2) % 8
        s = passive_index % 2

        # Decode board and position coordinates
        passive_start_x = p // 8
        passive_start_y = p % 8
        direction = d
        dist = s+1

        # check if valid:
        return (passive_start_x, passive_start_y, direction, dist), passive_index, masked_log_probs


    def sample_aggressive(self, policy_output, passive_move, valid_moves):
        '''
        From the policy model, sample aggressive action
        '''
        aggressive_logits = self.aggressive_logits(policy_output).squeeze()
        # create mask, subset conditioned on passive move
        mask = torch.zeros(aggressive_logits.shape, device=device, dtype=torch.float32)
        for move in valid_moves:
            if move[0] == passive_move:
                (ax, ay)= move[1]
                adix = (ax*8) + ay 
                mask[adix] = 1

        # probabilities
        masked_probs = torch.softmax(aggressive_logits, dim=-1)*mask
        masked_log_probs = torch.log_softmax(aggressive_logits, dim=-1)*mask
        
        # Sample aggressive move
        aggressive_index = torch.multinomial(masked_probs, 1).item()

        # Decode aggressive move
        a_start_x = aggressive_index // 8
        a_start_y = aggressive_index % 8
        
        return (a_start_x, a_start_y), aggressive_index, masked_log_probs
        

                
    def model_action(self, policy_output):
        '''
        Hierarchical combinatorial sampling 
        First select valid passive move
        Then subset to legal aggressive moves and sample
        Return ShobuMove, model passive action index, model aggressive action index, and model unmasked log probabilities
        '''
        # Generate legal moves
        valid_bit_moves = self.board.move_gen()
        # Convert moves into model-readable format
        valid_moves = [convert_to_PPOMove(move) for move in valid_bit_moves]
        # Passive sampling
        passive_move, passive_index, passive_probs = self.sample_passive(policy_output, valid_moves)
        # Aggressive sampling
        aggressive_move, aggressive_index, aggressive_probs = self.sample_aggressive(policy_output, passive_move, valid_moves)
        # ShobuMove conversion
        move = convert_to_ShobuMove(passive_move, aggressive_move)
        return move, passive_index, aggressive_index, passive_probs, aggressive_probs 


    def select_action(self, state, train=True):
        '''
        Select policy from model for a given board state
        '''
        # train
        if train:
            # PPO moves
            if self.board.next_mover == self.player_color:  
                self.train_steps += 1
                self.steps_done += 1 
                with torch.no_grad():
                    policy_output = self.ppo_model.get_policy(state)
                    move, passive_index, aggressive_index, passive_probs, aggressive_probs = self.model_action(policy_output)
                return move, passive_index, aggressive_index, passive_probs, aggressive_probs
            # opp network moves
            else:
                with torch.no_grad():
                    policy_output = self.opp.get_policy(state)
                    move, passive_index, aggressive_index, passive_probs, aggressive_probs = self.model_action(policy_output)
                return move, passive_index, aggressive_index, passive_probs, aggressive_probs
        
        # inference
        else:
            with torch.no_grad():
                policy_output = self.ppo_model.get_policy(state)
                move, passive_probs, aggressive_probs = self.model_action(policy_output)
            return move, passive_index, aggressive_index, passive_probs, aggressive_probs

    
    def model_play_game(self, memory, train=True):
        '''
        Play a single training game of shobu
        '''
        # select opponents
        if train:
            self.opp = random.choice(self.opponent_pool)
            
        while (self.steps_done < MAX_TURNS):    
            # check for wincon
            if (winner := self.board.check_winner()) is not None:
                print(f"The winner is {winner}.")
                return memory
            
            ### TODO: get board_state as matrix rep
            board_state = self.board.as_matrix()
            
            # get input move by sampling from policy model
            start_state = torch.tensor(board_state.copy(), device=device, dtype=torch.float32).unsqueeze(0)
            move, passive_index, aggressive_index, passive_probs, aggressive_probs = self.select_action(start_state)
            
            # push to memory if player is the trainable model
            if self.board.next_mover == Player.BLACK:
                memory.push(start_state, passive_index, aggressive_index, passive_probs, aggressive_probs, 0, 0)
                        
            # apply move
            self.board = self.board.apply(move)
        return memory   

    def train(self, opt, scheduler, sparse=True):
        '''
        Train a PPO model for Shobu
        '''
        loss_list = []
        reward_list = []
        win_list = []
        intermediate_list = []
        # memory for caching states, action, next state, and reward
        train_memory = ReplayMemory(capacity=MEMORY_SIZE)
        
        for episode in range(MAX_GAMES):
            # init a new game
            self.episode = episode
            self.init_game()
            
            # play game
            t0 = time.time()
            self.ppo_model.eval() # to make dropout and batchnorm behave correctly
            episode_memory = ReplayMemory(capacity=MEMORY_SIZE)
            episode_memory = self.model_play_game(episode_memory)
            self.ppo_model.train()
            t1 = time.time()
            print(f"\n Game simulated in: {t1-t0}")
            
            # Display ending board for qualitative check
            print(self.board)
            print(f"Total moves:{self.steps_done}")

            # compute advantages for PPO loss
            transitions = episode_memory.memory
            batch = Transition(*zip(*transitions))
            state_batch = torch.cat(batch.state)
            with torch.no_grad():
                # get q values from value function
                q_values = self.ppo_model.value_function(state_batch)
                # intermediate rewards
                if sparse:
                    rewards = [0 for s in state_batch]
                else:
                    rewards = [intermediate_reward(s, self.player_color) for s in state_batch]
                # win/loss reward
                rewards[-1] += WIN_REWARD if (self.board.check_winner()==self.player_color) else -WIN_REWARD
                # compute returns + advantages
                returns, advantages = compute_returns(q_values, rewards)  
                advantages = torch.cat(advantages)
                returns = torch.stack(returns)
            # add advantage and returns to queue of transitions
            transitions_with_advantage = []
            for i in range(len(episode_memory.memory)):
                t = episode_memory.memory[i]
                transitions_with_advantage.append(Transition(t.state, t.passive, t.aggressive, t.passive_probs, t.aggressive_probs, advantages[i], returns[i]))
            for transition in transitions_with_advantage:
                train_memory.push(*transition) 
         
            # record winner every episode
            win_list.append(1 if (self.board.check_winner()==self.player_color) else 0)
            
            # training step every ON_POLICY episodes
            if ((episode+1) % ON_POLICY) == 0:
                for e in range(EPOCHS):
                    # minibatch
                    train_memory_copy = copy.deepcopy(train_memory)
                    train_memory_copy.shuffle()
                    batch_loss = []
                    batch_reward = []
                    while len(train_memory_copy) > 0:
                        # sample states
                        transitions = train_memory_copy.sample(BATCH_SIZE)
                        batch = Transition(*zip(*transitions))
                        # states
                        state_batch = torch.cat(batch.state)
                        # get old joint probability
                        passive_actions = torch.tensor(np.stack(batch.passive), device=device, dtype=torch.int64)
                        aggressive_actions = torch.tensor(np.stack(batch.aggressive), device=device, dtype=torch.int64)
                        old_passive_probs = torch.stack(batch.passive_probs).detach().gather(1, passive_actions.unsqueeze(1))
                        old_aggressive_probs = torch.stack(batch.aggressive_probs).detach().gather(1, aggressive_actions.unsqueeze(1))
                        old_probs_joint = old_passive_probs + old_aggressive_probs
                        # advantages and returns
                        advantages = torch.stack(batch.advantages)
                        returns = torch.stack(batch.returns)

                        # ratio loss on passive and aggressive move probs
                        policy_outputs = self.ppo_model.get_policy(state_batch)
                        passive_logits = self.passive_logits(policy_outputs)
                        aggressive_logits = self.aggressive_logits(policy_outputs)
                        new_passive_probs = torch.log_softmax(passive_logits, dim=1).gather(1, passive_actions.unsqueeze(1))
                        new_aggressive_probs = torch.log_softmax(aggressive_logits, dim=1).gather(1, aggressive_actions.unsqueeze(1))   
                        new_probs_joint = new_passive_probs + new_aggressive_probs
                        ratio = torch.exp(new_probs_joint-old_probs_joint)
                        
                        # Clipped loss using PPO loss
                        clip_loss = torch.min(ratio * advantages, torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * advantages)
                        policy_loss = -clip_loss.mean()
                        
                        # value function loss
                        value_loss = F.mse_loss(self.ppo_model.value_function(state_batch), returns)

                        # entropy loss
                        entropy_passive = torch.distributions.Categorical(logits=passive_logits).entropy()
                        entropy_aggressive = torch.distributions.Categorical(logits=aggressive_logits).entropy()
                        entropy_loss = -torch.mean(entropy_passive + entropy_aggressive) 

                        # Final PPO loss
                        total_loss = policy_loss + (C1*value_loss) + (C2*entropy_loss)

                        # backprop and optimize
                        opt.zero_grad()
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.ppo_model.parameters(), max_norm=1.0)
                        opt.step()

                        # metrics
                        batch_loss.append(total_loss.item())
                        batch_reward.append(np.mean(returns.clone().cpu().numpy()))
                    
                    # update metrics
                    avg_epoch_loss = np.mean(batch_loss)
                    avg_epoch_reward = np.mean(batch_reward)
                    loss_list.append(avg_epoch_loss)
                    reward_list.append(avg_epoch_reward)

                # lr scheduler if needed
                scheduler.step()
                
                # new memory queue - PPO requires fresh data
                train_memory = ReplayMemory(capacity=MEMORY_SIZE)

            # periodically add agents to opp pool
            if ((episode+1) % UPDATE_OPPS) == 0:
                opponent = copy.deepcopy(self.ppo_model) 
                for param in opponent.parameters():
                    param.requires_grad = False
                self.remove_opponent()
                self.opponent_pool.append(opponent)
                
            # plot training progress
            plot_every = ON_POLICY
            if ((episode+1) % plot_every) == 0:
                # plot progress
                plot_progress(reward_list, loss_list, win_list, plot_every, episode)

            # save checkpoints
            if ((episode+1)%1000) == 0:
                torch.save(self.ppo_model.state_dict(), f'checkpoints/ppo_checkpoint_{episode+1}.pth')

        # save final model
        torch.save(self.ppo_model.state_dict(), 'checkpoints/ppo_final.pth')


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    print(device)
    ppo_model = Shobu_PPO()
    ppo_model.to(device)
    optimizer = torch.optim.Adam(ppo_model.parameters(), lr=1e-4, amsgrad=True, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)
    
    shobu_rl = Shobu_RL(ppo_model)
    shobu_rl.train(optimizer, scheduler)