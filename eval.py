import os
from itertools import product, combinations, chain
import random
from typing import Union
from multiprocessing import Pool
from tqdm import tqdm

from agent import Agent, RLAgent
from shobu import Player, Shobu
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from rl_utils import get_board_representation

"""
This module defines helpful functions for evaluating the relative strengths
of Agents. This includes facilities for playing one-off games as well as
round robin tournaments.
"""

POOL_SIZE = 12

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
		if len(board.move_gen()) == 0:
			return Player(not board.next_mover.value)
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
				move = black_player.move(board, half_ply)
		else:
			if type(white_player) is RLAgent: 
				move = white_player.move(board, board_reps)
			else:
				move = white_player.move(board, half_ply)
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
	rated_agents, max_moves, id_a, id_b, seed = args
	black = rated_agents[id_a]
	white = rated_agents[id_b]
	torch.manual_seed(seed)
	np.random.seed(seed)
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
	games_and_seeds = [(*game, random.randint(1, 1000000)) for game in games]
	with Pool(POOL_SIZE) as p:
		results = p.map(_run_game_and_update, games_and_seeds)

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


def _arena_play_game(args):
	agents, max_moves, id_a, id_b, seed = args
	black = agents[id_a]
	white = agents[id_b]
	torch.manual_seed(seed)
	np.random.seed(seed)
	return id_a, id_b, play_game(black, white, max_moves=max_moves)


def tiered_arena(
				newcomer_agents: list[Agent],
				established_agents: list[Agent],
				output_csv_dir: str,
				n_games_between_each: int = 10,
				max_moves: int = 32) -> None:
	"""
	Plays a "tiered" round robin. Unlike `round_robin`, there is no rounds
	structure. We simply generate a large list of games and pmap over them,
	meaning we make maximum use of the multiprocessing Pool whenever we can.

	There is a two-tier structure to the tournament: all newcomers face each
	other, and then each newcomer will also face off against each of the
	established agents. The difference between this format and a round-robin
	is that established agents do not play against each other.

	Each pairing plays `n_games_between_each` games. It is recommended to choose
	an even number for this parameter so that both sides get to play black an
	equal number of times.

	Results (WDL) are saved to a CSV file in the specified folder.

	:param newcomer_agents: agents that should play against each other as well
		as against the established pool.
	:param established_agents: agents that do not need to play against each other,
		e.g. we've already evaluated their relative strength before and don't want
		to duplicate our efforts.
	:param output_csv_dir: the path to the output *directory* in which to put the
		CSV file containing WDL results from all of the pairings in this arena.
	:param n_games_between_each: the number of games to play in each pairing.
	:param max_moves: the maximum moves per game; if exceeded, game ends in draw.
	"""
	num_new = len(newcomer_agents)
	num_old = len(established_agents)
	print(f"Playing arena with {num_new} newcomers and {num_old} established agents.")
	all_agents = newcomer_agents + established_agents

	# quick sanity check on names
	names = set()
	for a in all_agents:
		names.add(a.name())
	if len(names) != len(all_agents):
		print("WARNING: some agents have the same name. Results will be ambiguous")

	new_idxs = range(len(newcomer_agents))
	old_idxs = range(len(newcomer_agents), len(newcomer_agents) + len(established_agents))
	newcomer_pairings = combinations(new_idxs, 2)
	new_vs_old_pairings = product(new_idxs, old_idxs)
	all_pairings = chain(newcomer_pairings, new_vs_old_pairings)

	per_game_players = []
	for (a, b) in all_pairings:
		num_black = n_games_between_each // 2
		num_white = n_games_between_each - num_black
		per_game_players.extend([(a, b)] * num_black + [(b, a)] * num_white)

	games = []
	for (id_a, id_b) in per_game_players:
		games.append((all_agents, max_moves, id_a, id_b, random.randint(1, 1000000)))

	random.shuffle(games)
	with Pool(POOL_SIZE) as p:
		results = list(tqdm(p.imap(_arena_play_game, games), total=len(games)))

	print(f"Done running, collecting results...")
	result_map = {}
	for id_a, id_b, result in results:
		if id_a in result_map:
			if id_b not in result_map[id_a]:
				result_map[id_a][id_b] = [0, 0, 0]
			if result is None:
				result_map[id_a][id_b][1] += 1  # draw
			elif result == Player.BLACK:
				result_map[id_a][id_b][0] += 1  # id_a played black and won
			else:
				result_map[id_a][id_b][2] += 1  # id_a played black and lost
		elif id_b in result_map:
			if id_a not in result_map[id_b]:
				result_map[id_b][id_a] = [0, 0, 0]
			if result is None:
				result_map[id_b][id_a][1] += 1  # draw
			elif result == Player.BLACK:
				result_map[id_b][id_a][2] += 1  # id_b played white and lost
			else:
				result_map[id_b][id_a][0] += 1   # id_b played white and won
		else:
			result_map[id_a] = {}
			result_map[id_a][id_b] = [0, 0, 0]
			if result is None:
				result_map[id_a][id_b][1] += 1  # draw
			elif result == Player.BLACK:
				result_map[id_a][id_b][0] += 1  # id_a played black and won
			else:
				result_map[id_a][id_b][2] += 1  # id_a played black and lost

	results_list = []
	for id_a in result_map.keys():
		for id_b in result_map[id_a].keys():
			results_list.append({
				"player_A": all_agents[id_a].name(),
				"player_B": all_agents[id_b].name(),
				"A_wins": result_map[id_a][id_b][0],
				"draws": result_map[id_a][id_b][1],
				"B_wins": result_map[id_a][id_b][2]
			})

	print("Done collecting results, writing to file...")
	df = pd.DataFrame(results_list)
	filename = f"arena_results_{num_new}_vs_{num_old}_{datetime.now().strftime('%Y:%m:%d-%H:%M:%S')}.csv"
	path = os.path.join(output_csv_dir, filename)
	df.to_csv(path, index=False)
	print(f"Done, file written to {path}")


def elo_from_arena_results(
				ingest_dir: str,
				output_csv_path: str) -> None:
	"""
	`tiered_arena` outputs a CSV file into some directory. You can pass that
	directory in here, and all `tiered_arena` results stored in that directory
	will be collated into a unified dataset based on which Elos will be computed.
	This outputs another CSV file, this time containing Elos for each agent which
	has participated in any of the arenas whose results are stored in
	`ingest_dir`.

	:param ingest_dir: The directory containing CSV files generated by
		`tiered_arena`
	:param output_csv_path: The path to the output *file* to write CSV Elo data
		for each agent which participated in any of the `tiered_arena`s.
	"""
	...
