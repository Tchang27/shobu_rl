import numpy as np
import math
from collections import namedtuple, deque
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'mask', 'win'))

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
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def create_move_offsets():
    # Only need 1-step directions (aggressive will handle 2-step separately)
    return np.array([
        [-1, -1], [-1, 0], [-1, 1], # Up-left, Up, Up-right
        [0, -1],          [0, 1], # Left,      Right
        [1, -1],  [1, 0],  [1, 1] # Down-left, Down, Down-right
    ])

def get_aggressive_boards(passive_board_id):
    """Get the two aggressive boards based on the passive board choice."""
    if passive_board_id == 0:
        return [1, 3]
    elif passive_board_id == 1:
        return [0, 2]
    elif passive_board_id == 2:
        return [1, 3]
    elif passive_board_id == 3:
        return [0, 2]
    return []

def print_board_row(board1, board2):
    """Print two 4x4 boards side by side with aligned stones."""
    for row1, row2 in zip(board1, board2):
        # Convert numbers to symbols for better readability
        row1_str = " ".join(["○" if cell == 1 else "●" if cell == -1 else "." for cell in row1])
        row2_str = " ".join(["○" if cell == 1 else "●" if cell == -1 else "." for cell in row2])
        print(f"{row1_str}    {row2_str}")

def apply_move(board, move_from, move_to):
    """Apply a move to the board."""
    x1, y1 = move_from
    x2, y2 = move_to
    board[x2, y2] = board[x1, y1]
    board[x1, y1] = 0

    return board

def apply_push(board, aggressive_move, push_target, current_player):
    """
    Update the board after the aggressive move and handle pushing mechanics.

    Parameters:
    - boards: List of 4 boards.
    - aggressive_move: (board_id, (ax, ay)) → new position of the aggressive piece.
    - push_target: (px, py) → enemy's piece position after the push.
    """
    ax, ay = aggressive_move
    px, py = push_target

    # check if piece was pushed onto valid square
    if (
        0 <= ax < board.shape[0] and
        0 <= ay < board.shape[1] and
        board[ax, ay] == current_player
    ):
        # Check if the pushed piece lands inside the board
        if (
            0 <= px < board.shape[0] and
            0 <= py < board.shape[1]
        ):
            # Move the enemy stone to the push target
            board[px, py] = current_player*-1
    return board

def game_won(boards, player):
    """Check if a player has won by clearing a board."""
    for board in boards:
        if np.all(board != -player):   # Opponent stones are gone
            return True
    return False