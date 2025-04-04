from itertools import product
import random
from typing import Union
from multiprocessing import Pool
from tqdm import tqdm

import numpy as np
from agent import Agent, RLAgent
from shobu import Player, Shobu
import torch
from rl_utils import get_board_representation

"""
This module defines helpful functions for evaluating the relative strengths
of Agents. This includes facilities for playing one-off games as well as
round robin tournaments.
"""

POOL_SIZE = 4

def play_game(
		black_player: Agent,
		white_player: Agent,
		initial_board_state=Shobu.starting_position(),
		max_moves: Union[int, None] = None,
		print_info=False) -> Union[Player, None]:
	"""
	Play a single game between two Agents.
	"""
	board = initial_board_state
	half_ply = 0
	board_reps = [torch.zeros(8,4,4, device = torch.device('cpu'), dtype=torch.float32) for _ in range(8)]
	while max_moves == None or half_ply / 2 < max_moves:
		if print_info: print(board)
		if (winner := board.check_winner()) is not None:
			if print_info: print(f"The winner is {winner}.")
			return winner
		
		# update board rep
		if type(black_player) is RLAgent:
			board_reps = get_board_representation(board, board_reps, torch.device('cpu'))
		elif type(white_player) is RLAgent:
			board.flip()
			board_reps = get_board_representation(board, board_reps, torch.device('cpu'))
			board.flip()
			
		if board.next_mover == Player.BLACK:
			if type(black_player) is RLAgent: 
				move = black_player.move(board, board_reps)
			else:
				move = black_player.move(board)
		else:
			if type(white_player) is RLAgent: 
				move = white_player.move(board, board_reps)
			else:
				move = white_player.move(board)
		board = board.apply(move)
		half_ply += 1

		if print_info: print(f"Next move is {move}")
		if print_info: print("\n---------------------------------")
	return None


class RatedAgent:
	"""
	Not important to read, just Elo rating system impl details. If we want to add
	provisional ratings, https://gameknot.com/help-answer.pl?question=29 is an
	easy system to use, though not perfect.
	"""
	def __init__(self, agent: Agent):
		self.agent = agent
		self.rating = 1200

	def update(self, played_as: Player, result: Union[Player, None], opponent_elo: int, k: int) -> int:
		"""
		returns the old rating
		"""
		old_rating = self.rating
		e = 1 / (1 + 10**((opponent_elo - self.rating)/ 400))
		if result == played_as:
			s = 1
		elif result is None:
			s = 0.5
		else:
			s = 0
		self.rating = round(self.rating + k * (s - e))
		return old_rating


# Picklable function
def _run_game_and_update(args):
	rated_agents, max_moves, id_a, id_b = args
	black = rated_agents[id_a]
	white = rated_agents[id_b]
	return (id_a, id_b, play_game(black.agent, white.agent, max_moves=max_moves))


def round_robin(
		agents: list[Agent],
		max_moves_per_game: int = 100,
		num_rounds: int = 10,
		k: int = 24
) -> tuple[np.ndarray, np.array]:
	"""
	Plays a (potentially multi-round) round robin tournament between all Agents.
	Each "round", Agents will play against all other Agents twice: once as white,
	and again as black. Elo scores are updated per-game. At the end of the match,
	we return
	  (1) a 2-dimensional square matrix m, where m[a][b] is `a`'s number of wins
	      against `b` when `a` played BLACK. Remember, `a` played against `b` as
	      BLACK for a total of `num_rounds` number of games. To get an overall
	      match score of `a` vs. `b`, be sure to also take into account m[b][a].
	  (2) a 1d array with the same number of elements as the input list of agents,
	      where each array element is the new Elo rating of that agent.

	For the most accurate results, consider playing multiple rounds of
	round-robin.

	:param agents: A list of Agents participating in this tournament.
	:param max_moves_per_game: The maximum number of (whole, 1-ply) moves to allow
	  a game to proceed before terminating the game in a draw.
	:param num_rounds: the number of rounds to run the tournament for. Let `t` be
	  the number of rounds, and `n` be the number of agents. The total number of
	  games to be played is then `nt(n-1)`. The total number of games played by
	  each player is t(n-1).
	:param print_info: whether to print diagnostic each round.
	:param k: The volatility of the Elo rating system; the maximum a player's
	  rating can change in a single Elo update.
	"""
	num_players = len(agents)
	rated_agents = list(map(lambda a: RatedAgent(a), agents))
	# pass rated_agents and max_moves to pickled fn
	rounds = [[(rated_agents, max_moves_per_game, p[0], p[1]) for p in product(range(num_players), repeat=2) if p[0] != p[1]]] * num_rounds
	win_matrix = np.zeros((num_players, num_players))

	with Pool(POOL_SIZE) as p:
		for round in tqdm(rounds):
			random.shuffle(round)  # this may be good or bad idea
			num_decisive = 0
			for (black_id, white_id, result) in p.map(_run_game_and_update, round):
				black = rated_agents[black_id]
				white = rated_agents[white_id]
				# update outside of parallel to avoid race
				old_black_rating = black.update(Player.BLACK, result, white.rating, k)
				white.update(Player.WHITE, result, old_black_rating, k)
				win_matrix[black_id][white_id] += 1
				if result is not None: num_decisive +=1
			print(f"{num_decisive} decisive games")

	elos = list(map(lambda ra: ra.rating, rated_agents))
	return win_matrix, elos

def get_WDL_from_round_robin(win_matrix: np.ndarray, A: int, B: int, num_rounds=10) -> tuple[int, int, int]:
	"""
	Given the results of a round-robin tournament, get W-D-L stats for
	player A vs. player B (i.e. player A won W times and lost L times)

	:param win_matrix: the win_matrix (first return value) from round_robin
	:param A: the index of player A in the list of Agents which was passed into
		round_robin
	:param B: the index of player B in the list of Agents which was passed into
		round_robin
	:param num_rounds: the number of rounds that the tournament went on for.
	:return: (w, d, l)
	"""
	return (win_matrix[A][B], num_rounds * 2 - win_matrix[A][B] - win_matrix[B][A], win_matrix[B][A])

def n_game_match(a: Agent, b: Agent, n: int = 10, max_moves: int = 100, k: int = 24) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int]]:
	"""
	Play n (default 10) games between agents A and B, and returns (
		(W,D,L) overall,
		(W,D,L) as BLACK,
		(W,D,L) as WHITE,
		(Elo A, Elo B)
	).
	Note the first return value is the sum of the second and third return values.
	Examine the second and third values to see if there are significant differences
	between the two.
	"""

	num_games_as_white = n // 2
	num_games_as_black = n - num_games_as_white
	rated_agents: list[RatedAgent] = [RatedAgent(a), RatedAgent(b)]
	games = [(rated_agents, max_moves, 0, 1)] * num_games_as_black + [(rated_agents, max_moves, 1, 0)] * num_games_as_white
	random.shuffle(games) # definitely not needed unless some seeding is going on
	with Pool(POOL_SIZE) as p:
		results = p.map(_run_game_and_update, games)

	overall = [0, 0, 0]
	as_black = [0, 0, 0]
	as_white = [0, 0, 0]
	for (b_i, w_i, result) in results:
		old_b_elo = rated_agents[b_i].update(Player.BLACK, result, rated_agents[w_i].rating, k)
		rated_agents[w_i].update(Player.WHITE, result, old_b_elo, k)
		if result is None:
			overall[1] += 1
			if b_i == 0: as_black[1] += 1 # player A was playing black and drew
			else: as_white[1] += 1 # player A was playing white and drew
		elif result == Player.BLACK:
			if b_i == 0: # player A was playing black and won
				as_black[0] += 1
				overall[0] += 1
			else: # player A was playing white and lost
				as_white[2] += 1
				overall[2] += 1
		else:
			if b_i == 0: # player A was playing black and lost
				as_black[2] += 1
				overall[2] += 1
			else: # player A was playing white and won
				as_white[0] += 1
				overall[0] += 1

	return tuple(overall), tuple(as_black), tuple(as_white), (rated_agents[0].rating, rated_agents[1].rating)
