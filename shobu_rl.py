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

# hyperparameters
# training parameters
MAX_TURNS = 100
MAX_GAMES = 500000
BATCH_SIZE = 256
EPOCHS = 4
ON_POLICY = 50


# Value function loss weight
C1 = 0.5
# Entropy loss weight
C2 = 0.05
C3 = 0.01
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
UPDATE_OPPS = 5000
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
        # steps in a game
        self.steps_done = 0
        # Shobu board
        self.board = Shobu.starting_position()
        # current player
        self.model_player = random.choice([-1, 1])
        # always start with black
        self.cur_turn = -1


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


    def select_action(self, state, device, train=True):
        '''
        Select policy from model for a given board state
        '''
        # if train, count steps
        if train:
            self.steps_done += 1 
        with torch.no_grad():
            policy_output = self.ppo_model.get_policy(state)
            move, passive_index, aggressive_index, passive_probs, aggressive_probs, passive_mask, aggressive_mask = model_action(policy_output, self.board, device)
        return move, passive_index, aggressive_index, passive_probs, aggressive_probs, passive_mask, aggressive_mask
        
        
    def select_opponent_action(self, board, device):
        if type(self.opp) is RandomAgent:
            move = self.opp.move(board)
        else:
            board_state = board.as_matrix()
            start_state = torch.tensor(board_state.copy(), device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                policy_output = self.opp.get_policy(start_state)
                move, _, _, _, _, _, _ = model_action(policy_output, self.board, device)
        return move
    
    
    def model_play_game(self, memory, train=True):
        '''
        Play a single training game of shobu
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
                
            if self.cur_turn == self.model_player:
                board_state = self.board.as_matrix()
                start_state = torch.tensor(board_state.copy(), device=device, dtype=torch.float32).unsqueeze(0)
                move, passive_index, aggressive_index, passive_probs, aggressive_probs, passive_mask, aggressive_mask = self.select_action(start_state, device)
            else:
                move = self.select_opponent_action(self.board, device)
                   
            # apply move
            self.board = self.board.apply(move)
            if self.cur_turn == self.model_player:
                #push to memory if player is the trainable model
                board_state = self.board.as_matrix()
                after_state = torch.tensor(board_state.copy(), device=device, dtype=torch.float32).unsqueeze(0)
                memory.push(start_state, after_state, passive_index, aggressive_index, passive_probs, aggressive_probs, passive_mask, aggressive_mask, 0, 0)
            
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
        
        # return memory of transitions
        return memory   

    
    def train(self, opt, scheduler, sparse=True):
        '''
        Train a PPO model for Shobu
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
        
        for episode in range(MAX_GAMES):
            # init a new game
            self.episode = episode
            self.init_game()
            
            # play game
            t0 = time.time()
            self.ppo_model.eval()
            episode_memory = ReplayMemory(capacity=MEMORY_SIZE)
            episode_memory = self.model_play_game(episode_memory)
            self.ppo_model.train()
            t1 = time.time()
            print(f"\n Game simulated in: {t1-t0}")
            
            # Display ending board for qualitative check
            print(self.board)
            print(f"Total moves:{self.steps_done}")

            # compute returns and advantages for PPO loss
            transitions = episode_memory.memory
            batch = Transition(*zip(*transitions))
            state_batch = torch.cat(batch.state)
            after_state_batch = torch.cat(batch.after_state)
            with torch.no_grad():
                # get q values from value function
                q_values = self.ppo_model.value_function(state_batch)
                # intermediate rewards using resulting states (after action)
                if sparse:
                    rewards = [0 for s in after_state_batch]
                else:
                    rewards = [intermediate_reward(state=s, step=i, max_step=MAX_TURNS) for i,s in enumerate(after_state_batch)]
                # win/loss reward - discourage stalling
                if self.board.check_winner():
                    rewards[-1] += WIN_REWARD if (self.board.check_winner() and (self.cur_turn==self.model_player)) else -WIN_REWARD
                else:
                    rewards[-1] += -WIN_REWARD*0.5
                # compute returns + advantages
                returns, advantages = compute_returns(rewards=rewards, values=q_values, device=device, gamma=GAMMA, lam=LAMBDA)
            # add advantage and returns to queue of transitions
            transitions_with_advantage = []
            for i in range(len(episode_memory.memory)):
                t = episode_memory.memory[i]
                transitions_with_advantage.append(Transition(t.state, t.after_state, t.passive, t.aggressive, t.passive_probs, t.aggressive_probs, t.passive_mask, t.aggressive_mask, advantages[i], returns[i]))
            for transition in transitions_with_advantage:
                train_memory.push(*transition) 
         
            # record winner every episode
            win_list.append(1 if (self.board.check_winner() and (self.cur_turn==self.model_player)) else 0)
            draw_list.append(1 if (not self.board.check_winner()) else 0)
            opp_win_list.append(1 if (self.board.check_winner() and (self.cur_turn!=self.model_player)) else 0)
            
            # training step every ON_POLICY episodes
            if ((episode+1) % ON_POLICY) == 0:
                # normalize returns and advantages
                #returns = torch.stack([t.returns for t in train_memory.memory])
                #returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                advantages = torch.stack([t.advantages for t in train_memory.memory])
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                for i, t in enumerate(train_memory.memory):
                    train_memory.memory[i] = Transition(t.state, t.after_state, t.passive, t.aggressive, t.passive_probs, t.aggressive_probs, t.passive_mask, t.aggressive_mask, advantages[i], t.returns)
                    
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
                        old_passive_logprobs = torch.stack(batch.passive_probs).detach().gather(1, passive_actions.unsqueeze(1))
                        old_aggressive_logprobs = torch.stack(batch.aggressive_probs).detach().gather(1, aggressive_actions.unsqueeze(1))
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
                        new_passive_logprobs = torch.log_softmax(passive_logits, dim=1).gather(1, passive_actions.unsqueeze(1))
                        new_aggressive_logprobs = torch.log_softmax(aggressive_logits, dim=1).gather(1, aggressive_actions.unsqueeze(1))  
                        new_logprobs_joint = new_passive_logprobs + new_aggressive_logprobs
                        ratio = torch.exp(new_logprobs_joint-old_logprobs_joint)

                        # Clipped PPO loss
                        clipped_ratio = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON)
                        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

                        # value function loss
                        values = self.ppo_model.value_function(state_batch).squeeze()
                        value_loss = F.mse_loss(values, returns)

                        # entropy loss
                        passive_logits[passive_masks == 0] = -1e10
                        aggressive_logits[aggressive_masks == 0] = -1e10
                        entropy_passive = torch.distributions.Categorical(logits=passive_logits).entropy()
                        entropy_aggressive = torch.distributions.Categorical(logits=aggressive_logits).entropy()
                        passive_entropy_loss = - (entropy_passive.mean())
                        aggressive_entropy_loss = - (entropy_aggressive.mean())

                        # Final PPO loss
                        total_loss = policy_loss + (C1*value_loss) + (C2 * passive_entropy_loss) + (C3 * aggressive_entropy_loss)

                        # backprop and optimize
                        opt.zero_grad()
                        total_loss.backward()
                        total_norm = torch.nn.utils.clip_grad_norm_(self.ppo_model.parameters(), max_norm=0.5)
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