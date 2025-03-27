import numpy as np
import torch
import math
from collections import namedtuple, deque
import random
from shobu import *
from IPython.display import clear_output
import matplotlib.pyplot as plt
from itertools import combinations

# mapping from 
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


Transition = namedtuple('Transition',
                        ('state', 'passive', 'aggressive', 'passive_probs','aggressive_probs', 'passive_mask', 'aggressive_mask', 'advantages', 'returns'))


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

    return (piece_discrep/16) #+ (-0.01)*(step/max_step)

    
def rolling_win_rate(win_list, window_size=25):
    """Calculate rolling win rate over the last `window_size` episodes."""
    if len(win_list) < window_size:
        return np.cumsum(win_list) / np.arange(1, len(win_list) + 1)
    return moving_average(win_list, window_size)


def moving_average(values, window_size=10):
    """Compute moving average to smooth the reward curve."""
    return np.convolve(values, np.ones(window_size)/window_size, mode='valid')


def plot_progress(reward_list, loss_list, ppo_loss_list, value_loss_list, entropy_loss_list, win_list, opp_win_list, draw_list, plot_every: int, episode: int):
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
    
    if len(reward_list) > plot_every:
        smoothed_rewards = moving_average(reward_list, window_size=(plot_every))
        ax1.plot(range(len(smoothed_rewards)), smoothed_rewards, label="Smoothed Batch Return", color="blue", linewidth=5)

    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Plot losses (right y-axis)
    ax2 = ax1.twinx()  
    ax2.set_ylabel("PPO Loss", color="tab:orange")  
    ax2.plot(ppo_loss_list, label=" Raw Batch PPO Loss", alpha=0.2, color="tab:orange")
    
    if len(ppo_loss_list) > plot_every:
        smoothed_loss = moving_average(ppo_loss_list, window_size=(plot_every))
        ax2.plot(range(len(smoothed_loss)), smoothed_loss, label="Smoothed Batch PPO Loss", color="orange", linewidth=5)

    ax2.tick_params(axis="y", labelcolor="tab:orange")
    
    # Plot losses (right y-axis)
    ax6 = ax1.twinx() 
    ax6.spines["right"].set_position(("outward", 40)) 
    ax6.set_ylabel("Value Loss", color="tab:purple")  
    ax6.plot(value_loss_list, label=" Raw Batch Value Loss", alpha=0.2, color="tab:purple")
    
    if len(value_loss_list) > plot_every:
        smoothed_loss = moving_average(value_loss_list, window_size=(plot_every))
        ax6.plot(range(len(smoothed_loss)), smoothed_loss, label="Smoothed Batch Value Loss", color="purple", linewidth=5)

    ax6.tick_params(axis="y", labelcolor="tab:purple")
    
    # Plot losses (right y-axis)
    ax7 = ax1.twinx()  
    ax7.spines["right"].set_position(("outward", 80)) 
    ax7.set_ylabel("Entropy Loss", color="tab:green")  
    ax7.plot(entropy_loss_list, label=" Raw Batch Entropy Loss", alpha=0.2, color="tab:green")
    
    if len(entropy_loss_list) > plot_every:
        smoothed_loss = moving_average(entropy_loss_list, window_size=(plot_every))
        ax7.plot(range(len(smoothed_loss)), smoothed_loss, label="Smoothed Batch Entropy Loss", color="green", linewidth=5)

    ax7.tick_params(axis="y", labelcolor="tab:green")
    
    # Plot losses (right y-axis)
    ax8 = ax1.twinx()  
    ax8.spines["right"].set_position(("outward", 120)) 
    ax8.set_ylabel("Loss", color="tab:red")  
    ax8.plot(loss_list, label=" Raw Batch Loss", alpha=0.4, color="tab:red")
    
    if len(entropy_loss_list) > plot_every:
        smoothed_loss = moving_average(loss_list, window_size=(plot_every))
        ax8.plot(range(len(smoothed_loss)), smoothed_loss, label="Smoothed Batch Loss", color="red", linewidth=5)

    ax8.tick_params(axis="y", labelcolor="tab:red")

    # Plot win rate for model
    fig2, ax3 = plt.subplots(figsize=(12, 8))
    ax3.set_xlabel("Episode")
    ax3.spines["right"].set_position(("outward", 60))  # Move win rate axis outward
    ax3.set_ylabel(f"Rolling Win Rate (Last {plot_every})", color="tab:green")

    # Calculate and plot rolling win rate
    if len(win_list) > plot_every:
        rolling_rate = rolling_win_rate(win_list, window_size=(plot_every))
        ax3.plot(range(len(rolling_rate)), rolling_rate, alpha=0.2,label="Rolling Win Rate", color="green", linewidth=3)

    ax3.tick_params(axis="y", labelcolor="tab:green")
    
    
    # Plot draw rate
    ax4 = ax3.twinx()
    ax4.spines["right"]  # Move win rate axis outward
    ax4.set_ylabel(f"Rolling Draw Rate (Last {plot_every})", color="tab:purple")

    # Calculate and plot rolling draw rate
    if len(draw_list) > plot_every:
        rolling_drawrate = rolling_win_rate(draw_list, window_size=(plot_every))
        ax4.plot(range(len(rolling_drawrate)), rolling_drawrate, alpha=0.2,label="Rolling Draw Rate", color="purple", linewidth=3)

    ax4.tick_params(axis="y", labelcolor="tab:purple")
    
    # Plot win rate for opp
    ax5 = ax3.twinx()
    ax5.spines["right"].set_position(("outward", 60))  # Move win rate axis outward
    ax5.set_ylabel(f"Rolling Opponent Win Rate (Last {plot_every})", color="tab:cyan")

    # Calculate and plot rolling draw rate
    if len(draw_list) > plot_every:
        rolling_opprate = rolling_win_rate(opp_win_list, window_size=(plot_every))
        ax5.plot(range(len(rolling_opprate)), rolling_opprate, alpha=0.2,label="Rolling Opponent Win Rate", color="cyan", linewidth=3)

    ax5.tick_params(axis="y", labelcolor="tab:cyan")


    plt.title(f"Training Progress (Episode {episode+1})")
    fig.tight_layout()  # Adjust layout for better spacing
    plt.grid()
    plt.show()
