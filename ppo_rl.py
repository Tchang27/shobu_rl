from shobu import *
from rl_utils import *
from models import *
from agent import RandomAgent
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import random
from collections import deque
import time
from itertools import combinations
from torch.distributions import Categorical

# hyperparameters
# training parameters
MAX_TURNS = 100
MAX_GAMES = 100000
BATCH_SIZE = 256
EPOCHS = 6
ON_POLICY = 50


# Value function loss weight
C1 = 1.0
# Entropy loss weight
C2 = 0.03
C3 = 0.015
# Clipped PPO parameter
EPSILON = 0.2
# Discount factor parameters
GAMMA = 0.995
LAMBDA = 0.95
# size of replay buffer
MEMORY_SIZE = 500000
# win reward
WIN_REWARD = 1
# add new opponent every UPDATE_OPPS episodes
UPDATE_OPPS = 2000
# max number of opponents to track at a given time
OPP_QUEUE_LENGTH = 8
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

class Shobu_RL():
    def __init__(self, ppo_model: nn.Module):
        super().__init__()
        # init models
        self.set_model(ppo_model)
        # init game
        self.init_game()
        

        
    def init_game(self):
        '''
        Initializes new board, randomly selects side for the model, and initializes board representations
        '''
        # steps in a game
        self.steps_done = 0
        # Shobu board
        self.board = Shobu.starting_position()
        # current player
        self.model_player = random.choice([-1, 1])
        # always start with black
        self.cur_turn = -1
        # board representations - separate boards for each subboard and piece type, with history of 8 past states
        self.board_reps = [torch.zeros(8,4,4, device=device, dtype=torch.float32) for _ in range(8)]
        self.opp_board_reps = [torch.zeros(8,4,4, device=device, dtype=torch.float32) for _ in range(8)]


    def set_model(self, ppo_model) -> None:
        '''
        Set models for training

        Input:
        - PPO model (policy and value)

        Return: None
        '''
        self.ppo_model = ppo_model
        # opponents - periodically freeze value model and add to list
        self.opponent_pool = deque([], maxlen=OPP_QUEUE_LENGTH)

        
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


    def select_action(self, state: torch.tensor, device: torch.device):
        '''
        Select policy from model for a given board state
        
        Input:
        - state: torch tensor of the board representation
        - device: torch device
        
        Outputs:
        - move: sampled ShobuMove
        - passive_index: index of sampled passive move
        - aggressive_index: index of sampled aggressive move
        - passive_probs: log probability of sampled passive move
        - aggressive_pros: log probability of sampled aggressive move
        - passive_mask: mask of valid passive moves
        - aggressive mask: mask of valid aggressive moves
        ''' 
        with torch.no_grad():
            policy_output = self.ppo_model.get_policy(state)
            move, passive_index, aggressive_index, passive_probs, aggressive_probs, passive_mask, aggressive_mask = model_action(policy_output, self.board, device)
        return move, passive_index, aggressive_index, passive_probs, aggressive_probs, passive_mask, aggressive_mask
        
        
    def select_opponent_action(self, board: Shobu, device: torch.device):
        '''
        Select policy from opponent for a given board state
        
        Input:
        - board: Shobu board
        - device: torch device
        
        Outputs:
        - move: sampled ShobuMove
        ''' 
        if type(self.opp) is RandomAgent:
            move = self.opp.move(board)
        else:
            start_state = torch.concatenate(self.opp_board_reps).unsqueeze(0)
            with torch.no_grad():
                policy_output = self.opp.get_policy(start_state)
                move, _, _, _, _, _, _ = model_action(policy_output, self.board, device)
        return move
    
    
    def model_play_game(self, memory: ReplayMemory, train=True):
        '''
        Play a single training game of shobu
        
        Input:
        - memory: ReplayMemory object to store game states
        
        Output:
        - memory: ReplayMemory object holding game states from simulation
        '''
        # select opponents
        if train:
            if random.uniform(0, 1) < (1/(len(self.opponent_pool)+1)):
                self.opp = RandomAgent()
            else:
                self.opp = random.choice(self.opponent_pool)
          
        while (self.steps_done < MAX_TURNS):
            if not train:
                print("Current board")
                print(self.board)
                
            # update board representation for model
            if self.cur_turn == self.model_player:
                self.board_reps = get_board_representation(self.board, self.board_reps, device)
            else:
                self.board.flip()
                self.board_reps = get_board_representation(self.board, self.board_reps, device)
                self.board.flip()
                
            # update board representation for opp
            if self.cur_turn != self.model_player:
                self.opp_board_reps = get_board_representation(self.board, self.opp_board_reps, device)
            else:
                self.board.flip()
                self.opp_board_reps = get_board_representation(self.board, self.opp_board_reps, device)
                self.board.flip()
                
            if self.cur_turn == self.model_player:
                start_state = torch.concatenate(self.board_reps).unsqueeze(0)
                move, passive_index, aggressive_index, passive_probs, aggressive_probs, passive_mask, aggressive_mask = self.select_action(start_state, device)
            else:
                move = self.select_opponent_action(self.board, device)
                   
            # apply move
            self.board = self.board.apply(move)
            if self.cur_turn == self.model_player:
                #push to memory if player is the trainable model
                memory.push(start_state, passive_index, aggressive_index, passive_probs, aggressive_probs, passive_mask, aggressive_mask, 0, 0)
            
            if not train:
                print("After move")
                print(self.board)
            
            # check for wincon
            if (winner := self.board.check_winner()) is not None:
                if self.cur_turn == self.model_player:
                    print(f"The winner is the model.")
                else:
                    print(f"The winner is the opponent.")
                return memory
            
            # flip board
            self.board.flip()
            # next turn
            self.cur_turn *= -1
            self.steps_done += 1 
        
        # return memory of transitions
        return memory   

    
    def train(self, opt, scheduler, sparse=True):
        '''
        Train a PPO model for Shobu
        
        Input:
        - opt: torch optimizer 
        - scheduler: torch learning rate scheduler
        
        Output:
        - none 
        '''
        loss_list = []
        ppo_loss_list = []
        value_loss_list = []
        p_entropy_loss_list = []
        a_entropy_loss_list = []
        reward_list = []
        win_list = []
        draw_list = []
        opp_win_list = []
        # memory for caching states, action, next state, and reward
        train_memory = ReplayMemory(capacity=MEMORY_SIZE)
        self.ppo_model.train()
        
        for episode in range(MAX_GAMES):
            # init a new game
            self.episode = episode
            self.init_game()
            
            # play game
            t0 = time.time()
            episode_memory = ReplayMemory(capacity=MEMORY_SIZE)
            episode_memory = self.model_play_game(episode_memory)
            t1 = time.time()
            print(f"\n Game simulated in: {t1-t0}")
            
            # Display ending board for qualitative check
#             print(f"Board from model view: {self.cur_turn==self.model_player}")
#             print(self.board)
#             print(f"Total moves:{self.steps_done}")

            # compute returns and advantages for PPO loss
            transitions = episode_memory.memory
            batch = Transition(*zip(*transitions))
            state_batch = torch.cat(batch.state)
            
            with torch.no_grad():
                # get q values from value function
                q_values = self.ppo_model.value_function(state_batch)
                # intermediate rewards using resulting states (after action)
                if sparse:
                    rewards = [0 for s in state_batch]
                else:
                    rewards = [intermediate_reward(state=s, step=i, max_step=MAX_TURNS) for i,s in enumerate(state_batch)]
                # shift rewards to reflect reward after action
                rewards = rewards[1:]
                # win/loss reward - discourage stalling
                if self.board.check_winner():
                    if (self.board.check_winner() and (self.cur_turn==self.model_player)): 
                        rewards.append(WIN_REWARD)
                    else:
                        rewards.append(-WIN_REWARD)
                else:
                    rewards.append(-WIN_REWARD)
                # compute returns + advantages
                returns, advantages = compute_returns(rewards=rewards, values=q_values, device=device, gamma=GAMMA, lam=LAMBDA)
                
            # add advantage and returns to queue of transitions
            transitions_with_advantage = []
            for i in range(len(episode_memory.memory)):
                t = episode_memory.memory[i]
                transitions_with_advantage.append(Transition(t.state, t.passive, t.aggressive, t.passive_probs, t.aggressive_probs, t.passive_mask, t.aggressive_mask, advantages[i], returns[i]))
            for transition in transitions_with_advantage:
                train_memory.push(*transition) 
         
            # record winner every episode
            win_list.append(1 if (self.board.check_winner() and (self.cur_turn==self.model_player)) else 0)
            draw_list.append(1 if (not self.board.check_winner()) else 0)
            opp_win_list.append(1 if (self.board.check_winner() and (self.cur_turn!=self.model_player)) else 0)
            
            # training step every ON_POLICY episodes
            if ((episode+1) % ON_POLICY) == 0:
                # normalize advantages
                advantages = torch.stack([t.advantages for t in train_memory.memory])
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                for i, t in enumerate(train_memory.memory):
                    train_memory.memory[i] = Transition(t.state, t.passive, t.aggressive, t.passive_probs, t.aggressive_probs, t.passive_mask, t.aggressive_mask, advantages[i], t.returns)
                    
                # train for EPOCHS epochs
                for e in range(EPOCHS):
                    # minibatch
                    train_memory_copy = copy.deepcopy(train_memory)
                    train_memory_copy.shuffle()
                    batch_ppo_loss_list = []
                    batch_value_loss_list = []
                    batch_p_entropy_loss_list = []
                    batch_a_entropy_loss_list = []
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
                        old_passive_logprobs = torch.stack(batch.passive_probs).detach()
                        old_aggressive_logprobs = torch.stack(batch.aggressive_probs)
                        old_logprobs_joint = old_passive_logprobs + old_aggressive_logprobs
                        # masks for logits
                        passive_masks = torch.stack(batch.passive_mask)
                        aggressive_masks = torch.stack(batch.aggressive_mask)
                        # advantages and returns
                        advantages = torch.stack(batch.advantages)
                        returns = torch.stack(batch.returns)

                        # ratio loss on passive and aggressive move probs
                        policy_outputs = self.ppo_model.get_policy(state_batch)
                        passive_logits = get_passive_logits(policy_outputs)
                        aggressive_logits = get_aggressive_logits(policy_outputs)
                        passive_logits, _ = normalized_mask_logits(passive_logits, passive_masks)
                        aggressive_logits, _ = normalized_mask_logits(aggressive_logits, aggressive_masks)
                        passive_dist = Categorical(logits=passive_logits)
                        aggressive_dist = Categorical(logits=aggressive_logits)
                        new_passive_logprobs = passive_dist.log_prob(passive_actions)
                        new_aggressive_logprobs = aggressive_dist.log_prob(aggressive_actions) 
                        new_logprobs_joint = new_passive_logprobs + new_aggressive_logprobs
                        ratio = torch.exp(new_logprobs_joint-old_logprobs_joint)

                        # Clipped PPO loss
                        clipped_ratio = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON)
                        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

                        # value function loss
                        values = self.ppo_model.value_function(state_batch)
                        value_loss = F.mse_loss(values, returns)

                        # entropy loss
                        entropy_passive = passive_dist.entropy()
                        entropy_aggressive = aggressive_dist.entropy()
                        passive_entropy_loss = - (entropy_passive.mean())
                        aggressive_entropy_loss = - (entropy_aggressive.mean())

                        # Final PPO loss
                        total_loss = policy_loss + (C1*value_loss) + (C2*passive_entropy_loss) + (C3*aggressive_entropy_loss)

                        # backprop and optimize
                        opt.zero_grad()
                        total_loss.backward()
                        total_norm = torch.nn.utils.clip_grad_norm_(self.ppo_model.parameters(), max_norm=5.0)
                        print(total_norm.item())
                        opt.step()

                        # metrics
                        batch_ppo_loss_list.append(policy_loss.item())
                        batch_value_loss_list.append(value_loss.item())
                        batch_p_entropy_loss_list.append(passive_entropy_loss.item())
                        batch_a_entropy_loss_list.append(aggressive_entropy_loss.item())
                        batch_loss.append(total_loss.item())
                        batch_reward.append(np.mean(returns.clone().cpu().numpy()))
                        
                    
                    # update metrics
                    avg_epoch_ppo_loss = np.mean(batch_ppo_loss_list)
                    avg_epoch_value_loss = np.mean(batch_value_loss_list)
                    avg_epoch_p_entropy_loss = np.mean(batch_p_entropy_loss_list)
                    avg_epoch_a_entropy_loss = np.mean(batch_a_entropy_loss_list)
                    avg_epoch_loss = np.mean(batch_loss)
                    avg_epoch_reward = np.mean(batch_reward)
                    ppo_loss_list.append(avg_epoch_ppo_loss)
                    value_loss_list.append(avg_epoch_value_loss)
                    p_entropy_loss_list.append(avg_epoch_p_entropy_loss)
                    a_entropy_loss_list.append(avg_epoch_a_entropy_loss)
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
                plot_progress(reward_list, loss_list, ppo_loss_list, value_loss_list, p_entropy_loss_list, a_entropy_loss_list, win_list, opp_win_list, draw_list, plot_every, episode)
                print("Top passive action prob:", torch.softmax(passive_logits, dim=-1).max().item())
                print("Top aggressive action prob:", torch.softmax(aggressive_logits, dim=-1).max().item())
                print("Avg valid passive moves:", (passive_masks == 1).float().mean().item() * passive_masks.shape[-1])
                print("Avg valid aggressive moves:", (aggressive_masks == 1).float().mean().item() * aggressive_masks.shape[-1])
                print("Entropies (p and a):" , avg_epoch_p_entropy_loss, avg_epoch_a_entropy_loss)
                print("Passive action prob:", torch.softmax(passive_logits, dim=-1)[0])
                print("Aggressive action prob:", torch.softmax(aggressive_logits, dim=-1)[0])
                print(f"Values: mean={values.mean().item():.3f}, std={values.std().item():.3f}")

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