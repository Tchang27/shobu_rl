import itertools
from enum import Enum
from typing import Union
from bitboard_utils import *
import sys

"""
Implements the Shobu game. See `main.py` for an example of how to use this
library. Ask Brandon if you have any questions.

Important functions include:

- Shobu::starting_position - get default starting position
- Shobu::from_str - parse a board from a string (for testing/debugging)
- Shobu::move_gen - generate legal moves
- Shobu::is_legal - check if a move is legal
- Shobu::apply - make a move on the board
- Shobu::flip - flip the board (rotate 180deg and flip colors)
- Shobu::check_winner - check if there's a winner in current position
- ShobuMove::from_str - parse a move from string (e.g. "a1f5n")
"""


class Player(Enum):
	"""
	This game has two players. BLACK plays first, and each player makes one move
	at a time in an alternating fashion.
	"""
	BLACK = False
	WHITE = True


class Direction(Enum):
	"""
	Represents one of the directions the piece can move in. Values are in the
	format (forward_direction, backward_direction) where each element is a
	function which shifts the board in the desired direction. See `bitboard_utils`
	for more implementation details on these N, NE, etc. functions
	"""
	N = (shift_N, shift_S)
	NE = (shift_NE, shift_SW)
	E = (shift_E, shift_W)
	SE = (shift_SE, shift_NW)
	S = (shift_S, shift_N)
	SW = (shift_SW, shift_NE)
	W = (shift_W, shift_E)
	NW = (shift_NW, shift_SE)

	def flip(self):
		if self == Direction.N: return Direction.S
		if self == Direction.NE: return Direction.SW
		if self == Direction.E: return Direction.W
		if self == Direction.SE: return Direction.NW
		if self == Direction.S: return Direction.N
		if self == Direction.SW: return Direction.NE
		if self == Direction.W: return Direction.E
		if self == Direction.NW: return Direction.SE


class ShobuSquare:
	"""
	We lay out a grid system for Shobu inspired by chess, where the bottom left
	square on the bottom left board is A1, the bottom right square on the bottom
	right board is H1, etc. (note then that the bottom-left sub-board is all
	squares between the coordinates A1 and D4)
	"""

	def __init__(self, x: int, y: int):
		"""
		A ShobuSquare is simply an x and y coordinate. Both x and y range from 0 to 7,
		where (x,y) = (0,0) correspond to the coordinate A1, and (x,y) = (7,7)
		corresponds to the coordinate H8.

		:param x: the x coordinate of this square
		:param y: the y coordinate of this square
		"""
		self.x = x
		self.y = y

	def flip(self):
		"""
		"Flip" this square, as if you were looking at the board rotated 180 degrees
		and applied the same coordinate system from that new reference point.
		For example, A1 becomes H8, G3 becomes B6
		"""
		self.x, self.y = 7 - self.x, 7 - self.y

	@staticmethod
	def from_str(s: str):
		"""
		Parse a string representation of the square, e.g. "A1" or "D4", into a
		ShobuSquare, e.g. ShobuSquare(0, 0) or ShobuSquare(4, 4)

		:param s: the string to parse
		:return: the corresponding ShobuSquare
		"""
		return ShobuSquare(ord(s[0]) - ord('A'), ord(s[1]) - ord('1'))

	def __str__(self):
		return chr(ord('a') + self.x) + chr(ord('1') + self.y)


class ShobuMove:
	"""
	A move in Shobu is represented by the coordinates of the passive piece to move,
	the coordinates of the aggressive piece to move, the direction to move it,
	and the number of steps to move it (either 1 or 2 steps). For example,
	given the following board state:

	8 ● ● ● ● | ● ● ● ●
	7 · · · · | · · · ·
	6 · · · · | · · · ·
	5 ○ ○ ○ ○ | ○ ○ ○ ○
		--------+--------
	4 ● ● ● ● | ● ● ● ●
	3 · · · · | · · · ·
	2 · · · · | · · · ·
	1 ○ ○ ○ ○ | ○ ○ ○ ○
		a b c d   e f g h

	and the following move: a1f5NE2

	the resulting board state will be:

	8 ● ● ● ● | ● ● ● ●
	7 · · · · | · · · ○
	6 · · · · | · · · ·
	5 ○ ○ ○ ○ | ○ · ○ ○
		--------+--------
	4 ● ● ● ● | ● ● ● ●
	3 · · ○ · | · · · ·
	2 · · · · | · · · ·
	1 · ○ ○ ○ | ○ ○ ○ ○
		a b c d   e f g h
	"""

	def __init__(self, passive_from: ShobuSquare, aggressive_from: ShobuSquare, direction: Direction, steps: int):
		"""
		:param passive_from: The starting coordinates of the passive piece to move
		:param aggressive_from: The starting coordinates of the aggressive piece to move
		:param direction: The direction to move these pieces
		:param steps: the number of steps to move these pieces
		"""
		self.passive_from = passive_from
		self.aggressive_from = aggressive_from
		self.direction = direction
		self.steps = steps

	def flip(self):
		"""
		Flip this move. See `ShobuSquare.flip` and `Shobu.flip` for why we care
		about doing this.
		"""
		self.passive_from.flip()
		self.aggressive_from.flip()
		self.direction = self.direction.flip()

	@staticmethod
	def from_str(s: str):
		"""
		Parse the move from the string representation.
		The string representation is expected to be a concatenation of:
		1. the starting coordinate of the passive piece to move.
		2. the starting coordinate of the aggressive piece to move.
		3. the direction to move.
		4. a '2' if we are moving two steps. Do not specify a number if we are
		   only moving 1 step

		For example, 'a1g2NE2' corresponds to moving pieces on a1 and g2 both
		2 steps to the northeast, where the piece on a1 is the passive move, and
		the piece on g2 is the aggressive move.

		:param s: The string representation of the move
		:return: The parsed ShobuMove
		"""
		try:
			s = s.upper()
			if s[-1] == '2':
				num_steps = 2
				s = s[:-1]
			else:
				num_steps = 1

			return ShobuMove(
				ShobuSquare.from_str(s[0:2]),
				ShobuSquare.from_str(s[2:4]),
				Direction[s[4:]],
				num_steps
			)
		except:
			return None

	def __str__(self):
		return f"{self.passive_from}{self.aggressive_from}{self.direction.name}{['', '2'][self.steps - 1]}"


class Shobu:
	"""
	Implements the Shobu game; board representation, move generation, move
	validity checks, game mechanics, and flipping.

	We implement Shobu as two 64-bit bitboards, one for black pieces and one for
	white pieces. There is an additional boolean value which indicates the next
	player. This makes each instance extremely memory-compact, so storing large
	numbers of Shobu instances for later retrieval should be very cheap.

	The bitboard format also allows us to rapidly and simultaneously compute
	legal moves. Everything is reduced down to simple bitwise operations on
	64-bit integers. This bitboard format can be then read back out into string
	or matrix representations depending on the necessary application.

	Some bitboard logic is implemented only from Black's perspective (e.g. see
	`_move_gen_black`, which is the basis of `move_gen`). This helps simplify
	logic. There are thoroughly-developed flipping facilities (see ShobuSquare::flip,
	ShobuMove::flip, ShobuBoard::flip) for "flipping" a board, i.e. rotating it
	by 180 degrees and inverting colors, so that we can play an entire game where
	both players believe they are playing with the black pieces simply by flipping
	the board and moves on each turn.
	"""

	def __init__(self, black=np.uint64(0), white=np.uint64(0), next_mover=Player.BLACK):
		self.black: np.uint64 = black
		self.white: np.uint64 = white
		self.next_mover: Player = next_mover

	@staticmethod
	def starting_position():
		return Shobu.from_str("""
		wwww wwww
		.... ....
		.... ....
		bbbb bbbb
		
		wwww wwww
		.... ....
		.... ....
		bbbb bbbb
		""")

	def move_gen(self) -> list[ShobuMove]:
		"""
		Generate all legal moves from the current position.

		:return: List of legal moves
		"""
		if self.next_mover == Player.BLACK:
			return self._move_gen_black()
		else:
			self.flip()
			moves = self._move_gen_black()
			self.flip()  # restore prior board state
			for move in moves:
				move.flip()
			return moves

	def _move_gen_black(self) -> list[ShobuMove]:
		moves: list[ShobuMove] = []

		for direction in Direction:
			moves += self._move_gen_single(direction)
			moves += self._move_gen_double(direction)

		return moves

	def _move_gen_single(self, direction: Direction) -> list[ShobuMove]:
		"""
		Generate all _single_ moves in a particular direction for BLACK, that
		is, pieces only move by one square from their starting position.

		:param direction: the direction to move in
		:return: List of all single-square moves for black from the current position
			in the given direction
		"""
		fwd, rev = direction.value
		all_pieces = self.black | self.white
		passive_moves = rev(fwd(self.black) & ~all_pieces)
		aggressive_moves = fwd(self.black) & ~self.black
		aggressive_moves = rev(aggressive_moves & ~(self.white & rev(all_pieces)))

		return Shobu._serialize_moves(passive_moves, aggressive_moves, direction, 1)

	def _move_gen_double(self, direction: Direction) -> list[ShobuMove]:
		"""
		Generate all _two step_ moves in a particular direction for BLACK, that is,
		pieces move two squares from their starting position

		:param direction: the direction to move in
		:return: List of all two-square moves for black from the current position in
			the given direction
		"""
		fwd, rev = direction.value
		all_pieces = self.black | self.white
		passive_move1 = fwd(self.black) & ~all_pieces
		passive_move2 = fwd(passive_move1) & ~all_pieces  # two-phase filter
		passive_moves = rev(rev(passive_move2))

		# behavior of 2-step aggressive move is quite a bit more complicated due to
		# all the pushing possibilities
		aggressive_move1 = fwd(self.black) & ~self.black
		pushing = aggressive_move1 & self.white  # whites that are pushed on first step
		aggressive_move1 &= ~(pushing & rev(all_pieces))  # x o o _ pattern
		aggressive_move2 = fwd(aggressive_move1) & ~self.black  # next step
		pushing = fwd(pushing) | (aggressive_move2 & self.white)
		aggressive_move2 &= ~(pushing & rev(all_pieces))  # x o _ o pattern
		aggressive_moves = rev(rev(aggressive_move2))

		return Shobu._serialize_moves(passive_moves, aggressive_moves, direction, 2)

	def is_legal(self, move: ShobuMove) -> bool:
		"""
		Determine whether, in the context of the current board state, the given
		ShobuMove is legal or not. This function checks a variety of conditions;
		both moved pieces must be `next_mover`'s pieces; the passive piece cannot
		push; the aggressive piece cannot push more than one.

		:param move: The move to check legality for
		:return: true if the move is legal, false otherwise
		"""
		passive_piece: np.uint64 = np.uint64(move.passive_from.y * 8 + move.passive_from.x)
		aggressive_piece: np.uint64 = np.uint64(move.aggressive_from.y * 8 + move.aggressive_from.x)

		passive_piece = one << passive_piece
		aggressive_piece = one << aggressive_piece

		if (self.next_mover == Player.BLACK and bottom_right(passive_piece)) or (self.next_mover == Player.WHITE and upper_right(passive_piece)):
			aggressive_piece = left(aggressive_piece)
		elif (self.next_mover == Player.BLACK and bottom_left(passive_piece)) or (self.next_mover == Player.WHITE and upper_left(passive_piece)):
			aggressive_piece = right(aggressive_piece)
		else:
			return False  # passive move is not on homeboard

		mover_pieces = self.black if self.next_mover == Player.BLACK else self.white
		if not (mover_pieces & passive_piece) or not (mover_pieces & aggressive_piece):
			# Either passive or aggressive piece is not the color of the current mover
			return False
		if move.steps == 1:
			return self._is_legal_single(passive_piece, aggressive_piece, move.direction)
		elif move.steps == 2:
			return self._is_legal_double(passive_piece, aggressive_piece, move.direction)

		# move is neither 1 nor 2 steps
		return False

	def _is_legal_single(self, passive_piece: np.uint64, aggressive_piece: np.uint64, direction: Direction) -> bool:
		all_pieces = self.black | self.white
		mover_pieces = self.black if self.next_mover == Player.BLACK else self.white
		fwd, _ = direction.value
		new_passive, new_aggressive = fwd(passive_piece), fwd(aggressive_piece)
		if not new_passive or not new_aggressive:
			# can't fall off the board voluntarily
			return False
		if new_passive & all_pieces:
			# passive can't push any other piece
			return False
		if new_aggressive & mover_pieces:
			# aggressive can't push your own pieces
			return False
		num_movers = bool(new_aggressive & all_pieces) + bool(fwd(new_aggressive) & all_pieces)
		if num_movers > 1:
			# can't push more than 1
			return False
		return True

	def _is_legal_double(self, passive_piece: np.uint64, aggressive_piece: np.uint64, direction: Direction) -> bool:
		if not self._is_legal_single(passive_piece, aggressive_piece, direction):
			# if a single step wasn't legal, the whole thing won't be legal
			return False

		all_pieces = self.black | self.white
		mover_pieces = self.black if self.next_mover == Player.BLACK else self.white
		fwd, _ = direction.value

		new_passive, new_aggressive = fwd(fwd(passive_piece)), fwd(fwd(aggressive_piece))
		if not new_passive or not new_aggressive:
			# can't fall off the board voluntarily
			return False
		if new_passive & all_pieces:
			# passive can't push any other piece
			return False
		if new_aggressive & mover_pieces:
			# aggressive can't push your own pieces
			return False
		num_movers = bool(fwd(aggressive_piece) & all_pieces) + bool(new_aggressive & all_pieces) + bool(fwd(new_aggressive) & all_pieces)
		if num_movers > 1:
			# Can't push more than 1
			return False
		return True

	def apply(self, move: ShobuMove, flip_next_mover: bool = True):
		"""
		"Applies" the given move, returning a new Shobu instance representing the
		board state after the move was made.

		**Important note**: This function DOES NOT validate the move before applying
		it. You can consider applying an illegal move to be undefined behavior. Call
		`is_legal` first if you aren't sure if the move is legal or not.

		:param move: The move to make
		:param flip_next_mover: Whether to flip the next_mover.
		:return: a new Shobu instance representing the board state after the move
		  is made.
		"""
		mover_pieces = self.black if self.next_mover == Player.BLACK else self.white
		other_pieces = self.black if self.next_mover == Player.WHITE else self.white
		passive_piece = np.uint64(move.passive_from.y * 8 + move.passive_from.x)
		aggressive_piece = np.uint64(move.aggressive_from.y * 8 + move.aggressive_from.x)
		passive_piece = one << passive_piece
		aggressive_piece = one << aggressive_piece
		fwd, _ = move.direction.value

		for _ in range(move.steps):
			# shift passive piece over by one: remove old pos, add new pos
			mover_pieces &= ~passive_piece
			passive_piece = fwd(passive_piece)
			mover_pieces |= passive_piece

			# shift aggressive piece over by one, then check if there's something to
			# push on the opponent's board
			mover_pieces &= ~aggressive_piece
			aggressive_piece = fwd(aggressive_piece)
			mover_pieces |= aggressive_piece

			if aggressive_piece & other_pieces:
				# pushing case.
				other_pieces = other_pieces & ~aggressive_piece | fwd(aggressive_piece)

		next_mover = Player(not self.next_mover.value) if flip_next_mover else self.next_mover
		if self.next_mover == Player.BLACK:
			return Shobu(mover_pieces, other_pieces, next_mover)
		else:
			return Shobu(other_pieces, mover_pieces, next_mover)

	@staticmethod
	def _serialize_moves(
					passive_moves: np.uint64,
					aggressive_moves: np.uint64,
					direction: Direction,
					num_moves: int) -> list[ShobuMove]:
		moves: list[ShobuMove] = []

		# "left passive" moves: Make the passive move on the left homeboard, and then
		# make the aggressive move on any of the two right boards
		left_passive = [iter_coords(bottom_left(passive_moves)), iter_coords(right(aggressive_moves))]
		for pas_coord, agg_coord in itertools.product(*left_passive):
			moves.append(ShobuMove(ShobuSquare(*pas_coord), ShobuSquare(*agg_coord), direction, num_moves))

		# "right passive" moves: Make the passive move on the right homeboard, and
		# then make the aggressive move on any of the two left boards
		right_passive = [iter_coords(bottom_right(passive_moves)), iter_coords(left(aggressive_moves))]
		for pas_coord, agg_coord in itertools.product(*right_passive):
			moves.append(ShobuMove(ShobuSquare(*pas_coord), ShobuSquare(*agg_coord), direction, num_moves))

		return moves

	def check_winner(self) -> Union[Player, None]:
		"""
		Checks the win condition (a player loses if they no longer have any pieces
		on any one of the sub-boards)
		:return: If this board has a winner, return the Player that won. Otherwise,
		return None.
		"""
		quadrants = [upper_left, upper_right, bottom_left, bottom_right]

		def any_empty_subboards(board):
			return any(map(lambda quadrant: not quadrant(board), quadrants))

		if any_empty_subboards(self.white):
			return Player.BLACK
		elif any_empty_subboards(self.black):
			return Player.WHITE

		return None

	def as_matrix(self) -> np.ndarray:
		"""
		Converts this Shobu instance into a matrix. -1 represents white pieces,
		1 represents black pieces, and 0 represents empty squares.
		"""
		def process_bitboard(board: np.uint64):
			as_bytes = np.array([board]).view(np.uint8)
			if not (as_bytes.dtype.byteorder == '>' or (as_bytes.dtype.byteorder == '=' and sys.byteorder == 'big')):
				as_bytes = as_bytes[::-1]  # try to get endianness right in case this is a problem
			bits = np.float32(np.unpackbits(np.expand_dims(as_bytes, axis=1), axis=1))
			bits = np.flip(bits, axis=1)
			return bits
		#board = process_bitboard(self.black) - process_bitboard(self.white)
		#return np.expand_dims(board, axis=0)
		blackboard = process_bitboard(self.black)
		whiteboard = process_bitboard(self.white)
		b1 = blackboard[:4,:4]
		b2 = blackboard[:4,4:]
		b3 = blackboard[4:,:4]
		b4 = blackboard[4:,4:]
		e1 = whiteboard[:4,:4]
		e2 = whiteboard[:4,4:]
		e3 = whiteboard[4:,:4]
		e4 = whiteboard[4:,4:]
		board = np.stack((b1,b2,b3,b4,e1,e2,e3,e4))
		return board

	def flip(self):
		"""
		Rotate the board by 180 degrees and swap out all black pieces for white,
		and vice-versa. Also flips the next mover. This is useful because we only
		generate moves from BLACK's POV. This mutates the board.

		:return: None
		"""
		self.next_mover = Player(not self.next_mover.value)
		self.white, self.black = bit_reverse(self.black), bit_reverse(self.white)

	def copy(self):
		return Shobu(self.black, self.white, self.next_mover)

	@staticmethod
	def from_str(s: str, next_mover=Player.BLACK):
		"""
		Converts a string sketch of the board into a Shobu board. All characters
		in the string are ignored except for "b", which represents black pieces,
		"w", which represents white pieces, and ".", which represents empty squares.
		It is expected that the first 4 characters represent the top row of the
		upper-left board, the next 4 characters represent the top row of the
		upper-right board, etc.

		For example, we can parse a text representation like so:

		b..b wbw.
		bbw. ....
		ww.. wbbw
		...w ...b

		bbbb bbbb
		.... ....
		.... ....
		wwww wwww

		(and the outcome would be as you expect.)

		:param s: The string to parse
		:param next_mover: The next mover
		:return: Shobu instance corresponding to the given input
		"""
		pieces = [c for c in s if c in ['w', 'b', '.', '●', '○', '·']][:64]
		white = np.uint64(0)
		black = np.uint64(0)
		for i, p in enumerate(pieces):
			x = i % 8
			y = 7 - (i // 8)
			idx = y*8 + x
			if p == 'b' or p == '●':
				black |= np.uint64(1) << np.uint64(idx)
			elif p == 'w' or p == '○':
				white |= np.uint64(1) << np.uint64(idx)

		return Shobu(black, white, next_mover)

	def __str__(self):
		"""
		Pretty-printing
		:return: String repr of the board
		"""
		black_piece = '●'
		white_piece = '○'
		empty_piece = '·'

		template = "\n".join([
			"8 · · · · | · · · ·",
			"7 · · · · | · · · ·",
			"6 · · · · | · · · ·",
			"5 · · · · | · · · ·",
			"  --------+--------",
			"4 · · · · | · · · ·",
			"3 · · · · | · · · ·",
			"2 · · · · | · · · ·",
			"1 · · · · | · · · ·",
			"  a b c d   e f g h",
			"",
			"Next to move: "
		]).split(empty_piece)

		pieces = [empty_piece] * 65
		pieces[-1] = black_piece if self.next_mover == Player.BLACK else white_piece

		for x, y in iter_coords(self.black):
			pieces[(7-y)*8 + x] = black_piece

		for x, y in iter_coords(self.white):
			pieces[(7-y)*8 + x] = white_piece

		return "".join(x for group in zip(template, pieces) for x in group)
