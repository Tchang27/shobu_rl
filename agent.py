import abc
import random

from shobu import Shobu, ShobuMove


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
