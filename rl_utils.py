import numpy as np
import torch
import math
from collections import namedtuple, deque
import random
from shobu import *
from IPython.display import clear_output
import matplotlib.pyplot as plt
from itertools import combinations
import os

#### CONSTANTS ####
DIRECTION_MAPPING = {
    Direction.NW: 0,
    Direction.N: 1,
    Direction.NE: 2,
    Direction.E: 3,
    Direction.SE: 4,
    Direction.S: 5,
    Direction.SW: 6,
    Direction.W: 7
}
DIRECTION_IDX_MAPPING = {
    0: Direction.NW,
    1:  Direction.N,
    2: Direction.NE,
    3: Direction.E,
    4: Direction.SE,
    5: Direction.S,
    6: Direction.SW,
    7: Direction.W
}

#### SEED ####
def seed(seed=1000):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    

#### REPLAY BUFFER ####

Transition = namedtuple('Transition',
                        ('state', 'after_state', 'passive', 'aggressive', 'passive_probs','aggressive_probs', 'passive_mask', 'aggressive_mask', 'advantages', 'returns'))


class ReplayMemory(object):
    '''
    Class for storing transitions
    Reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    '''
    def __init__(self, capacity = 100000):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        #return random.sample(self.memory, batch_size)
        batch = [self.memory.popleft() for _ in range(min(batch_size, len(self.memory)))]  
        return batch
    
    def shuffle(self):
        self.memory = deque(random.sample(self.memory, len(self.memory)))

    def __len__(self):
        return len(self.memory)
    
    
#### REPRESENTATION CONVERSIONS ####

def convert_to_PPOMove(move: ShobuMove):
    '''
    Convert ShobuMove object for RL model
    Output format:
    ((passive_start_x: int, passive_start_y: int, direction: (int,int), distance: int)) , ((aggressive_start_x: int, aggressive_start_y: int))
    
    '''
    # map coordinates
    passive_start_x, passive_start_y = move.passive_from.x, move.passive_from.y
    aggressive_start_x, aggressive_start_y = move.aggressive_from.x, move.aggressive_from.y
    # map direction from Direction to vector
    direction = DIRECTION_MAPPING[move.direction]
    # map distance
    distance = move.steps
        
    return ((passive_start_x, passive_start_y, direction, distance), 
            (aggressive_start_x, aggressive_start_y))


def convert_to_ShobuMove(passive_move, aggressive_move):
    '''
    Convert ShobuMove object for RL model
    Output format:
    ((passive_start_x: int, passive_start_y: int, direction: (int,int), distance: int)) , ((aggressive_start_x: int, aggressive_start_y: int))
    
    '''
    # map coordinates
    passive_from = ShobuSquare(passive_move[0], passive_move[1])
    aggressive_from = ShobuSquare(aggressive_move[0], aggressive_move[1])
    # map direction from Direction to vector
    direction = DIRECTION_IDX_MAPPING[passive_move[2]]
    # map distance
    distance = passive_move[3]
        
    return ShobuMove(passive_from, aggressive_from, direction, distance)


#### SAMPLING FUNCTION UTILS ####

def get_passive_logits(policy_output):
    '''
    From the policy model, extract passive move logits
    '''
    # Extract passive probabilities
    p_pos = policy_output["passive"]["position"]
    p_dir = policy_output["passive"]["direction"]
    p_dist = policy_output["passive"]["distance"]

    # Create passive joint distribution tensor
    passive_logits = p_pos.unsqueeze(-1).unsqueeze(-1) + p_dir.unsqueeze(1).unsqueeze(-1) + p_dist.unsqueeze(1).unsqueeze(1)
    # Flatten the passive move probabilities
    passive_logits = passive_logits.flatten(start_dim=1)
    return passive_logits


def get_aggressive_logits(policy_output):
    '''
    From the policy model, extract aggressive move logits
    '''
    # Extract aggressive probabilities
    a_pos = policy_output["aggressive"]["position"]

    aggressive_logits = a_pos
    return aggressive_logits


def mask_passive_logits(logits, valid_moves, device):
    # create mask
    mask = torch.zeros(logits.shape, device=device, dtype=torch.float32)
    for move in valid_moves:
        px, py, pd, ps = move[0]
        pfrom = (px*8) + py 
        pidx = (pfrom * (8 * 2)) + (pd * 2) + (ps-1)
        mask[pidx] = 1
    logits[mask == 0] = -1e10
    return logits, mask


def mask_aggressive_logits(logits, passive_move, valid_moves, device):
    # create mask
    mask = torch.zeros(logits.shape, device=device, dtype=torch.float32)
    for move in valid_moves:
        if move[0] == passive_move:
            (ax, ay)= move[1]
            adix = (ax*8) + ay 
            mask[adix] = 1
    logits[mask == 0] = -1e10
    return logits, mask


def sample_passive(policy_output, valid_moves, device):
    '''
    From the policy model, sample passive action
    '''
    # Flatten the passive move probabilities
    passive_logits = get_passive_logits(policy_output).squeeze()

    valid_passive_moves = set()
    for m in valid_moves:
        valid_passive_moves.add(m[0])

    # create mask
    masked_logits, mask = mask_passive_logits(passive_logits, valid_moves, device)

    # get masked probs
    masked_probs = torch.softmax(masked_logits, dim=-1)
    masked_log_probs = torch.log_softmax(masked_logits, dim=-1)

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
    if (passive_start_x, passive_start_y, direction, dist) not in valid_passive_moves:
        print(valid_passive_moves)
        print(masked_logits[passive_index])
    assert (passive_start_x, passive_start_y, direction, dist) in valid_passive_moves
    return (passive_start_x, passive_start_y, direction, dist), passive_index, masked_log_probs, mask


def sample_aggressive(policy_output, passive_move, valid_moves, device):
    '''
    From the policy model, sample aggressive action
    '''
    aggressive_logits = get_aggressive_logits(policy_output).squeeze()
    masked_logits, mask = mask_aggressive_logits(aggressive_logits, passive_move, valid_moves, device)

    # get masked probs
    masked_probs = torch.softmax(masked_logits, dim=-1)
    masked_log_probs = torch.log_softmax(masked_logits, dim=-1)

    # Sample aggressive move
    aggressive_index = torch.multinomial(masked_probs, 1).item()

    # Decode aggressive move
    a_start_x = aggressive_index // 8
    a_start_y = aggressive_index % 8

    # check validity
    assert ((passive_move),(a_start_x, a_start_y)) in valid_moves
    return (a_start_x, a_start_y), aggressive_index, masked_log_probs, mask  


def model_action(policy_output, board, device):
        '''
        Hierarchical combinatorial sampling 
        First select valid passive move
        Then subset to legal aggressive moves and sample
        Return ShobuMove, model passive action index, model aggressive action index, and model unmasked log probabilities
        '''
        # Generate legal moves
        valid_shobu_moves = board.move_gen()
        # Convert moves into model-readable format
        valid_moves = [convert_to_PPOMove(move) for move in valid_shobu_moves]
        # Passive sampling
        passive_move, passive_index, passive_probs, passive_mask = sample_passive(policy_output, valid_moves, device)
        # Aggressive sampling
        aggressive_move, aggressive_index, aggressive_probs, aggressive_mask = sample_aggressive(policy_output, passive_move, valid_moves, device)
        # ShobuMove conversion
        move = convert_to_ShobuMove(passive_move, aggressive_move)
        return move, passive_index, aggressive_index, passive_probs, aggressive_probs, passive_mask, aggressive_mask


#### LOSS FUNCTION UTILS ####
    
def compute_returns(rewards, values, device, gamma=0.99, lam=0.95) -> list[list, list]:
    """
    Compute returns and advantages using Generalized Advantage Estimation (GAE).
    """
    returns = []
    advantages = []
    gae = 0
    
    # Iterate backwards to calculate GAE and returns
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0  # No next value at terminal state
        else:
            next_value = values[t + 1]

        # Temporal difference (TD) error
        delta = rewards[t] + gamma * next_value - values[t]

        # GAE calculation
        gae = delta + gamma * lam * gae
        # Return is the advantage + value estimate, used for vaue function loss
        return_t = gae + values[t]
        returns.insert(0, return_t)  # Insert at the beginning for reverse order
        advantages.insert(0, gae)
        
    advantages = torch.tensor(advantages, device=device, dtype=torch.float32).squeeze()
    returns = torch.tensor(returns, device=device, dtype=torch.float32).squeeze()

    return returns, advantages
      
        
def intermediate_reward(state, step, max_step) -> torch.Tensor:
    '''
    Intermediate reward function - we can shape this later
    
    '''   
    # reward for piece discrepancy
    piece_discrep = torch.sum(state)

    return (-0.01)*(step) +(piece_discrep/16)

    
#### PLOTTING ####    
    
def rolling_win_rate(win_list, window_size=25):
    """Calculate moving_average rolling win rate over the last `window_size` episodes."""
    if len(win_list) < window_size:
        return np.cumsum(win_list) / np.arange(1, len(win_list) + 1)
    return moving_average(win_list, window_size)

def rolling_exp_win_rate(win_list, window_size=25):
    """Calculate exponential_moving_average rolling win rate over the last `window_size` episodes."""
    return exponential_moving_average(win_list, window_size)


def moving_average(values, window_size=10):
    """Compute moving average to smooth the reward curve."""
    return np.convolve(values, np.ones(window_size)/window_size, mode='valid')


def exponential_moving_average(values, alpha=0.05):
    """Compute moving average to smooth the reward curve."""
    ema = np.zeros_like(values, dtype=np.float64)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
    return ema


def plot_progress(reward_list, loss_list, ppo_loss_list, value_loss_list, p_entropy_loss_list, a_entropy_loss_list, win_list, opp_win_list, draw_list, plot_every: int, episode: int):
    '''
    Plot training progress
    
    Inputs:
    - reward_list: list of average win/loss reward/return sampled after each episode
    - loss_list: list of average loss after each episode
    - win_list: list of 0s and 1s representing wins by the model
    - draw_list: list of 0s and 1s representing draws (episode terminated at max moves)
    - plot_every: episode intervals when you plot
    - episode: current episode/game
    '''
    clear_output(wait=True)
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot rewards (left y-axis)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Return", color="tab:blue")
    ax1.plot(reward_list, label="Raw Batch Return", alpha=0.4, color="tab:blue")
    
    smoothed_rewards = exponential_moving_average(reward_list)
    ax1.plot(range(len(smoothed_rewards)), smoothed_rewards, label="Smoothed Batch Return", color="blue", linewidth=5)

    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Plot losses (right y-axis)
    ax2 = ax1.twinx()  
    ax2.set_ylabel("PPO Loss", color="tab:orange")  
    ax2.plot(ppo_loss_list, label=" Raw Batch PPO Loss", alpha=0.2, color="tab:orange")
    
    smoothed_loss = exponential_moving_average(ppo_loss_list)
    ax2.plot(range(len(smoothed_loss)), smoothed_loss, label="Smoothed Batch PPO Loss", color="orange", linewidth=5)

    ax2.tick_params(axis="y", labelcolor="tab:orange")
    
    # Plot losses (right y-axis)
    ax6 = ax1.twinx() 
    ax6.spines["right"].set_position(("outward", 40)) 
    ax6.set_ylabel("Value Loss", color="tab:purple")  
    ax6.plot(value_loss_list, label=" Raw Batch Value Loss", alpha=0.2, color="tab:purple")
    
    smoothed_loss = exponential_moving_average(value_loss_list)
    ax6.plot(range(len(smoothed_loss)), smoothed_loss, label="Smoothed Batch Value Loss", color="purple", linewidth=5)

    ax6.tick_params(axis="y", labelcolor="tab:purple")
    
    # Plot losses (right y-axis)
    ax7 = ax1.twinx()  
    ax7.spines["right"].set_position(("outward", 80)) 
    ax7.set_ylabel("Passive Entropy Loss", color="tab:green")  
    ax7.plot(p_entropy_loss_list, label=" Raw Batch Passive Entropy Loss", alpha=0.2, color="tab:green")
    
    smoothed_loss = exponential_moving_average(p_entropy_loss_list)
    ax7.plot(range(len(smoothed_loss)), smoothed_loss, label="Smoothed Batch Passive Entropy Loss", color="green", linewidth=5)

    ax7.tick_params(axis="y", labelcolor="tab:green")
    
    # Plot losses (right y-axis)
    ax8 = ax1.twinx()  
    ax8.spines["right"].set_position(("outward", 120)) 
    ax8.set_ylabel("Aggressive Entropy Loss", color="tab:pink")  
    ax8.plot(a_entropy_loss_list, label=" Raw Batch Aggressive Entropy Loss", alpha=0.2, color="tab:pink")
    
    smoothed_loss = exponential_moving_average(a_entropy_loss_list)
    ax8.plot(range(len(smoothed_loss)), smoothed_loss, label="Smoothed Batch Aggressive Entropy Loss", color="pink", linewidth=5)

    ax8.tick_params(axis="y", labelcolor="tab:pink")
    
    # Plot losses (right y-axis)
    ax9 = ax1.twinx()  
    ax9.spines["right"].set_position(("outward", 160)) 
    ax9.set_ylabel("Loss", color="tab:red")  
    ax9.plot(loss_list, label=" Raw Batch Loss", alpha=0.4, color="tab:red")
    
    smoothed_loss = exponential_moving_average(loss_list)
    ax9.plot(range(len(smoothed_loss)), smoothed_loss, label="Smoothed Batch Loss", color="red", linewidth=5)

    ax9.tick_params(axis="y", labelcolor="tab:red")

    # Plot win rate for model
    fig2, ax3 = plt.subplots(figsize=(12, 8))
    ax3.set_xlabel("Episode")
    ax3.spines["right"].set_position(("outward", 60))  # Move win rate axis outward
    ax3.set_ylabel(f"Rolling Win Rate)", color="tab:green")

    # Calculate and plot rolling win rate
    rolling_rate = exponential_moving_average(win_list)
    ax3.plot(range(len(rolling_rate)), rolling_rate, alpha=0.2,label="Rolling Win Rate", color="green", linewidth=3)

    ax3.tick_params(axis="y", labelcolor="tab:green")
    ax3.set_ylim(0, 1)
    
    
    # Plot draw rate
    ax4 = ax3.twinx()
    ax4.spines["right"]  # Move win rate axis outward
    ax4.set_ylabel(f"Rolling Draw Rate)", color="tab:purple")

    # Calculate and plot rolling draw rate
    rolling_drawrate = exponential_moving_average(draw_list)
    ax4.plot(range(len(rolling_drawrate)), rolling_drawrate, alpha=0.2,label="Rolling Draw Rate", color="purple", linewidth=3)

    ax4.tick_params(axis="y", labelcolor="tab:purple")
    ax4.set_ylim(0, 1)
    
    # Plot win rate for opp
    ax5 = ax3.twinx()
    ax5.spines["right"].set_position(("outward", 60))  # Move win rate axis outward
    ax5.set_ylabel(f"Rolling Opponent Win Rate)", color="tab:cyan")

    # Calculate and plot rolling draw rate
    rolling_opprate = exponential_moving_average(opp_win_list)
    ax5.plot(range(len(rolling_opprate)), rolling_opprate, alpha=0.2,label="Rolling Opponent Win Rate", color="cyan", linewidth=3)

    ax5.tick_params(axis="y", labelcolor="tab:cyan")
    ax5.set_ylim(0, 1)


    plt.title(f"Training Progress Episode {episode+1}")
    fig.tight_layout()  # Adjust layout for better spacing
    plt.grid()
    plt.show()
