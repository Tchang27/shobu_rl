import numpy as np
import torch
import math
from utils import *

WHITE, BLACK, EMPTY = 1, -1, 0
BOARD_SIZE = 4
MOVE_OFFSETS = create_move_offsets()

class Shobu():
    def __init__(self):
        # game is finished
        self.game_finished = False
        # current player turn
        self.turn = WHITE
        # current player turn is finished
        self.turn_finished = False
        # baord representation
        self.board_state = self.init_board()

    def init_board(self) -> np.array:
        '''
        Initializes game board in the following manner:
        Each board is a 4x4 numpy array
        
        Input: None
        
        Return: np.array representing intial board state
        '''
        board = np.array([
            [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, -1, -1, -1]],  # 0, Light board 1, for player white
            [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, -1, -1, -1]],  # 1, Light board 2, for player black
            [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, -1, -1, -1]],  # 2, Dark board 1, for player white
            [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, -1, -1, -1]]   # 3, Dark board 2, for player black
        ])
        return board
    
    def generate_all_valid_moves(self, boards, player):
        all_moves = []
        # Passive boards: WHITE uses light (0,2), BLACK uses dark (1,3)
        passive_boards = [0, 2] if player == WHITE else [1, 3]

        for passive_board_id in passive_boards:
            passive_board = boards[passive_board_id]
            stone_positions = np.argwhere(passive_board == player)

            # Generate all 1-step passive moves
            for (x, y) in stone_positions:
                for dx, dy in MOVE_OFFSETS[:8]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and passive_board[nx, ny] == EMPTY:
                        # Now find matching aggressive moves on opposite-color boards
                        aggressive_boards = [2, 3] if passive_board_id in [0, 1] else [0, 1]
                        
                        for aggressive_board_id in aggressive_boards:
                            aggressive_board = boards[aggressive_board_id]
                            aggressive_stones = np.argwhere(aggressive_board == player)

                            for (ax, ay) in aggressive_stones:
                                # Aggressive move must mirror passive direction (1 or 2 steps)
                                for step in [1, 2]:
                                    anx, any = ax + dx*step, ay + dy*step
                                    if 0 <= anx < BOARD_SIZE and 0 <= any < BOARD_SIZE:
                                        target = aggressive_board[anx, any]
                                        if target == EMPTY:
                                            all_moves.append((
                                                (passive_board_id, (x, y), (nx, ny)),
                                                (aggressive_board_id, (ax, ay), (anx, any)),
                                                ((-1, -1), (-1, -1))
                                            ))
                                        elif target == -player:
                                            # Check push rules
                                            push_x, push_y = anx + dx, any + dy
                                            if (0 <= push_x < BOARD_SIZE and 
                                                0 <= push_y < BOARD_SIZE and 
                                                aggressive_board[push_x, push_y] == EMPTY):
                                                all_moves.append((
                                                    (passive_board_id, (x, y), (nx, ny)),
                                                    (aggressive_board_id, (ax, ay), (anx, any)),
                                                    ((anx, any), (push_x, push_y))
                                                ))
        return all_moves
        

    def play_game(self):
        '''
        REPL for playing Ludo game
        We will make a similar function or modify this for training the RL model
        '''
        while not self.game_finished:
            # Display boards
            print("\n White's Boards (Top)\n")
            print_board_row(*[self.board_state[0], self.board_state[2]])
            print("\n Black's Boards (Bottom)\n")
            print_board_row(*[self.board_state[1], self.board_state[3]])

            valid_moves = self.generate_all_valid_moves(self.board_state, self.turn)
            if valid_moves:
                # print valid moves
                # for idx, pair in enumerate(valid_moves):
                #     print(f"Move {idx}")
                #     print(f"Passive: Board {pair[0][0]}, move from {pair[0][1]}, move to {pair[0][2]}")
                #     print(f"Aggressive: Board {pair[1][0]}, move from {pair[1][1]}, move to {pair[1][2]}, push to {pair[2][0]}, push enemy {pair[2][1]}")

                # get input move
                move_idx = int(input("\nSelect move pair by index: "))

                # apply moves
                passive, aggressive, push = valid_moves[move_idx]
                self.board_state[passive[0]] = apply_move(
                    self.board_state[passive[0]],
                    passive[1],
                    passive[2]
                )
                self.board_state[aggressive[0]] = apply_move(
                    self.board_state[aggressive[0]],
                    aggressive[1],
                    aggressive[2]
                )

                # handle push case
                self.board_state[aggressive[0]] = apply_push(self.board_state[aggressive[0]], push[0], push[1], self.turn)
            
            # check win con
            if game_won(self.board_state, self.turn):
                print(f"{'White' if self.turn == 1 else ' Black'} wins!")
                break

            # switch players
            self.turn *= -1
        return
    

if __name__ == "__main__":
    shobu_game = Shobu()
    shobu_game.play_game()