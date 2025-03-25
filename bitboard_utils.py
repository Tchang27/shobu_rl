from typing import Generator
import numpy as np

one = np.uint64(1)

# Rank and file masks
_rank_masks = [np.uint64(0xFF) << i * 8 for i in range(8)]
_file_masks = [np.uint64(0x0101010101010101) << i for i in range(8)]
_n_mask = ~(_rank_masks[3] | _rank_masks[7])
_e_mask = ~(_file_masks[3] | _file_masks[7])
_s_mask = ~(_rank_masks[0] | _rank_masks[4])
_w_mask = ~(_file_masks[0] | _file_masks[4])

# Sub-board masks for each of the 4 sub-boards
_upper_left_mask = np.uint64(0x0F0F0F0F00000000)
_upper_right_mask = _upper_left_mask << 4
_bottom_left_mask = _upper_left_mask >> 32
_bottom_right_mask = _bottom_left_mask << 4
_left_mask = _upper_left_mask | _bottom_left_mask
_right_mask = _upper_right_mask | _bottom_right_mask


def shift_N(board: np.uint64) -> np.uint64:
	return (board & _n_mask) << 8


def shift_E(board: np.uint64) -> np.uint64:
	return (board & _e_mask) << 1


def shift_S(board: np.uint64) -> np.uint64:
	return (board & _s_mask) >> 8


def shift_W(board: np.uint64) -> np.uint64:
	return (board & _w_mask) >> 1


def shift_NE(board: np.uint64) -> np.uint64:
	return shift_N(shift_E(board))


def shift_SE(board: np.uint64) -> np.uint64:
	return shift_S(shift_E(board))


def shift_SW(board: np.uint64) -> np.uint64:
	return shift_S(shift_W(board))


def shift_NW(board: np.uint64) -> np.uint64:
	return shift_N(shift_W(board))


def upper_left(board: np.uint64) -> np.uint64:
	return board & _upper_left_mask


def upper_right(board: np.uint64) -> np.uint64:
	return board & _upper_right_mask


def bottom_left(board: np.uint64) -> np.uint64:
	return board & _bottom_left_mask


def bottom_right(board: np.uint64) -> np.uint64:
	return board & _bottom_right_mask


def left(board: np.uint64) -> np.uint64:
	return board & _left_mask


def right(board: np.uint64) -> np.uint64:
	return board & _right_mask


def _ilsb(x: np.uint64) -> int:
	"""
	Get the index of the Least Significant Bit (LSB) in x
	"""
	return int((x & -x)).bit_length() - 1


def bit_reverse(x: np.uint64) -> np.uint64:
	"""
	Reverses all the bits in this number. For example, 0b1011 -> 0b1101
	Probably a smarter way to do this but I can't think that hard
	:param x: The number whose bits we wish to reverse
	:return: the results of the reverse
	"""
	for i in range(32):
		j = 63 - i
		z = ((x >> i) ^ (x >> j)) & one
		x ^= (z << i) | (z << j)
	return x


def _pick(x: np.uint64) -> tuple[int, np.uint64]:
	"""
	'Pick' the LSB out of x, returning (index of LSB, x with this bit unset)
	"""
	idx = _ilsb(x)
	return idx, x & ~(one << idx)


def _idx_to_coord(idx: int) -> tuple[int, int]:
	"""
	Converts a raw bitboard bit index to a "human"-understandable coordinate.
	e.g. bit 7 is H1, which corresponds to coordinate (7, 0); bit 56 is A8, which
	corresponds to coordinate (0, 7).
	:param idx: the index of the bitboard bit
	:return: the coordinate of the piece on the board
	"""
	return idx % 8, idx // 8


def iter_coords(board: np.uint64) -> Generator[tuple[int, int], None, None]:
	"""
	Given a bitboard, iterate through all coordinates for which there is a piece
	present at those coordinates.
	:param board: The bitboard to iterate through
	:return: Generates a new coordinate per iteration
	"""
	while board:
		piece_idx, board = _pick(board)
		yield _idx_to_coord(piece_idx)
