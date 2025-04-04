import abc
import random
import time
from shobu import Shobu, ShobuMove, Player
from models import Shobu_PPO, Shobu_MCTS
from rl_utils import model_action
from mcts_sequential import MCTree
import torch


class Agent(abc.ABC):
	"""
	An `Agent` is someone / something who is capable of "playing the game", that
	is, given a board position, decide on a move to make. This can be a user
	manually inputting moves, or a bot which arbitrarily chooses a move out of
	the set of all legal moves, or a more advanced RL bot which plays by
	evaluating the position, or a Flask server which makes moves as it receives
	requests to its API endpoints.
	"""

	@abc.abstractmethod
	def move(self, board: Shobu) -> ShobuMove:
		"""
		Given the current board state, decide the next move that you wish to play.
		:param board: current board state
		:return: your move
		"""


class RandomAgent(Agent):
	"""
	A `RandomAgent` is a bot player which randomly selects a legal move and
	plays it.
	"""

	def move(self, board: Shobu):
		candidates = board.move_gen()
		return random.choice(candidates)


class UserAgent(Agent):
	"""
	A `UserAgent` is a human user, inputting moves in "move notation" (see
	`ShobuMove.from_str` for more details) into the terminal. Moves are validated
	before they are actually played.
	"""

	def move(self, board: Shobu):
		while True:
			input_str = input("Input your next move: ")
			parsed = ShobuMove.from_str(input_str)
			if parsed is not None and board.is_legal(parsed):
				return parsed
			print("Illegal move. ", end="")


# TODO Flask agent, RL agent, etc
class RLAgent(Agent):
	"""
	A `RLAgent` is a bot player which uses a trained PPO model
	to select moves.
	"""
	def __init__(self, checkpoint_path: str):
		self.device = torch.device("cpu")   
		self.model = Shobu_PPO(self.device)
		self.model.to(self.device)
		self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
		self.model.eval()

	def move(self, board: Shobu, board_reps: list):
		with torch.no_grad():
			# check if we need to flip board
			was_moved = False
			if board.next_mover == Player.WHITE:
				board.flip()
				was_moved = True
			start_state = torch.concatenate(board_reps).unsqueeze(0)
			policy_output = self.model.get_policy(start_state)
			move, _, _, _, _, _, _ = model_action(policy_output, board, self.device)
			# check if we need to flip board and move
			if was_moved:
				board.flip()
				move.flip()
		return move

class MCTSAgent(Agent):
	"""
	A `RLAgent` is a bot player which uses a trained PPO model
	to select moves.
	"""
	def __init__(self, checkpoint_path: str):
		self.device = torch.device("cpu")   
		self.model = Shobu_MCTS(self.device)
		self.model.to(self.device)
		self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
		self.model.eval()

	def move(self, board: Shobu):
		with torch.no_grad():
			# check if we need to flip board
			was_moved = False
			if board.next_mover == Player.WHITE:
				board.flip()
				was_moved = True
			mcts = MCTree(self.model, board, self.device)
			rollout = mcts.search(800, noise=False)
			move = rollout.sample_move(0.2)
			# check if we need to flip board and move
			if was_moved:
				board.flip()
				move.flip()
		return move