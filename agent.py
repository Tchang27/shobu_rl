import abc
import random
from shobu import Shobu, ShobuMove, Player
from models import Shobu_PPO, Shobu_MCTS, Shobu_MCTS_Conv
from rl_utils import model_action
from mcts_simul import MCTree
from mcts_conv import MCTree_Conv
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
	def move(self, board: Shobu, half_ply: int) -> ShobuMove:
		"""
		Given the current board state, decide the next move that you wish to play.
		:param board: current board state
		:param half_ply: current move number, in half plies
		:return: your move
		"""

	@abc.abstractmethod
	def name(self) -> str:
		"""
		Returns the "name" of this agent. Choose something that is human readable
		and that uniquely identifies this agent.
		:return: str name
		"""


class RandomAgent(Agent):
	"""
	A `RandomAgent` is a bot player which randomly selects a legal move and
	plays it.
	"""

	def __init__(self, name="RandomAgent"):
		self.agent_name = name

	def move(self, board: Shobu, half_ply: int):
		candidates = board.move_gen()
		return random.choice(candidates)

	def name(self):
		return self.agent_name


class UserAgent(Agent):
	"""
	A `UserAgent` is a human user, inputting moves in "move notation" (see
	`ShobuMove.from_str` for more details) into the terminal. Moves are validated
	before they are actually played.
	"""

	def __init__(self, user_name="UserAgent"):
		self.agent_name = user_name

	def move(self, board: Shobu, half_ply: int):
		while True:
			input_str = input("Input your next move: ")
			parsed = ShobuMove.from_str(input_str)
			if parsed is not None and board.is_legal(parsed):
				return parsed
			print("Illegal move. ", end="")

	def name(self):
		return self.agent_name


# TODO Flask agent, RL agent, etc
class RLAgent(Agent):
	"""
	A `RLAgent` is a bot player which uses a trained PPO model
	to select moves.
	"""
	def __init__(self, checkpoint_path: str, name="PPOAgent"):
		self.device = torch.device("cpu")   
		self.model = Shobu_PPO(self.device)
		self.model.to(self.device)
		self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
		self.model.eval()
		self.agent_name = name

	def move(self, board: Shobu, board_reps: list):
		torch.set_num_threads(1)
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

	def name(self):
		return self.agent_name


class MCTSAgent(Agent):
	"""
	A `MCTSAgent` is a bot player which uses a trained MCTS model
	to select moves.
	"""
	def __init__(self, checkpoint_path: str, name=None):
		self.device = torch.device("cpu")   
		self.model = Shobu_MCTS(self.device)
		self.model.to(self.device)
		self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
		self.model.eval()
		if name is None:
			# choose a sensible default
			self.agent_name = checkpoint_path.split('/')[-1].split('.')[0]
		else:
			self.agent_name = name

	def move(self, board: Shobu, half_ply: int):
		#torch.set_num_threads(1)
		with torch.no_grad():
			# check if we need to flip board
			was_moved = False
			if board.next_mover == Player.WHITE:
				board.flip()
				was_moved = True
			mcts = MCTree(self.model, board, self.device)
			rollout = mcts.search(800, noise=False)
			#print(f'model thinks it has {((rollout.total_reward / rollout.num_visits + 1) * 50):.1f}% chance of winning')
			if half_ply < 0:
				move = rollout.sample_move(float('inf'))
			else:
				move = rollout.sample_move(0)
			# check if we need to flip board and move
			if was_moved:
				board.flip()
				move.flip()
		return move

	def name(self):
		return self.agent_name


class MCTSConvAgent(Agent):
	"""
	A `MCTSConvAgent` is a bot player which uses a trained MCTS model
	to select moves.
	"""
	def __init__(self, checkpoint_path: str, name=None):
		self.device = torch.device("cpu")   
		self.model = Shobu_MCTS_Conv(self.device)
		self.model.to(self.device)
		self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
		self.model.eval()
		if name is None:
			# choose a sensible default
			self.agent_name = checkpoint_path.split('/')[-1].split('.')[0]
		else:
			self.agent_name = name

	def move(self, board: Shobu, half_ply: int):
		#torch.set_num_threads(1)
		with torch.no_grad():
			# check if we need to flip board
			was_moved = False
			if board.next_mover == Player.WHITE:
				board.flip()
				was_moved = True
			mcts = MCTree_Conv(self.model, board, self.device)
			rollout = mcts.search(800, noise=False)
			if half_ply < 2:
				move = rollout.sample_move(float('inf'))
			else:
				move = rollout.sample_move(0)
			# check if we need to flip board and move
			if was_moved:
				board.flip()
				move.flip()
		return move

	def name(self):
		return self.agent_name
