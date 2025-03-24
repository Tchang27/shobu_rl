from shobu import *
from utils import *
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import random
from collections import deque

# hyperparameters
# epsilon parameters for exploration phase
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 0.001

# training parameters
MAX_TURNS = 250
MAX_GAMES = 50000
BATCH_SIZE = 128
MINIBATCH_NUM = 3

# target network temporal difference weight
GAMMA = 0.95
# weight of win/loss reward
LAMBDA = 0.6
# target network soft update hyperparameter
TAU = 0.001
# size of replay buffer
MEMORY_SIZE = 1000000
# win reward
WIN_REWARD = 15
# add new opponent every UPDATE_OPPS epochs
UPDATE_OPPS = 500
# max number of opponents to track at a given time
OPP_QUEUE_LENGTH = 10
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)


class Shobu_RL(Shobu):
    def __init__(self, value_model: nn.Module, target_model: nn.Module):
        super().__init__()
        # init models
        self.set_model(value_model, target_model)
        # steps throughout training
        self.train_steps = 0
        # init game
        self.init_game()
        
    
    def init_game(self):
        # steps in a game
        self.steps_done = 0


    def set_model(self, value_model, target_model) -> None:
        '''
        Set models for training

        Input:
        - Value model
        - Target model

        Return: None
        '''
        self.value_model = value_model
        self.target_model = target_model
        # opponents - periodically freeze value model and add to list
        self.opponent_pool = deque([], maxlen=OPP_QUEUE_LENGTH)
        opponent = copy.deepcopy(value_model) 
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

    def valid_move_mask(self, valid_moves):
        '''
        Get move mask for RL model

        Input: valid moves from self.generate_all_valid_moves
        valid_move[0]: passive moves
            - 0: idx of board
            - 1: move from coordinate
            - 2: move to coordinate
        valid_move[1]: aggressive moves
            - 0: idx of board
            - 1: move from coordinate
            - 2: move to coordinate
        valid_move[2]: pushing moves
            - 1: cur player piece coordinate
            - 2: enemy player pushed piece to coordinate

        Output: mask of moves
        '''
        '''
        The RL agent has the following available moves:
        - Piece can move to any square on each board (16 squares on each board)
        - There's passive turn and an aggressive turn
        Total moves = 32*32 = 1,024 possible move combinations
        Expression for move idx:
        - Input: pi (px, py), ai (ax, ay)
        - Output:[(pi*16)+((px*4)+py+1)]*[(ai*16)+(ax*4)+ay+1] -1
        '''
        mask = np.zeros(shape=(1,1024))
        # TODO - fill out masking based on valid moves
        return mask

    def select_action(self, state, train=True) -> torch.tensor:
        '''
        Select action based on epsilon-greedy policy
        Reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        '''
        sample = random.random()
        
        valid_moves = self.generate_all_valid_moves(self.board_state, self.turn)
        # epsilon greedy search
        if train:
            # update epsilon and select for value model
            eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-EPS_DECAY * self.episode)
            if self.turn == 0:  
                self.train_steps += 1
                self.steps_done += 1 
                if sample > eps_threshold:
                    with torch.no_grad():
                        action = self.value_model(state)
                        mask = (torch.tensor(mask, device=device, dtype=torch.float32) == 0)
                        action = action.masked_fill(mask.bool(), -1e9)
                        masked_action = torch.argmax(action)
                    return masked_action, mask
                else:
                    with torch.no_grad():
                        indices = np.nonzero(mask)[0]
                        mask = (torch.tensor(mask, device=device, dtype=torch.float32) == 0)
                        masked_action = torch.tensor(random.choice(indices), device=device)
                    return masked_action, mask
            # target network moves
            else:
                with torch.no_grad():
                    action = self.current_opps[self.turn-1](state)
                    mask = (torch.tensor(mask, device=device, dtype=torch.float32) == 0)
                    action = action.masked_fill(mask.bool(), -1e9)
                    masked_action = torch.argmax(action)
                return masked_action, mask
        # inference
        else:
            with torch.no_grad():
                action = self.value_model(state)
                mask = (torch.tensor(mask, device=device, dtype=torch.float32) == 0)
                action = action.masked_fill(mask.bool(), -1e9)
                masked_action = torch.argmax(action)
            return masked_action, mask
        
    def intermediate_reward(self):
        pass

    def model_play_game(self):
        pass

    def train(self):
        pass