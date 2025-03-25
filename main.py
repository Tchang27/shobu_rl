from agent import Agent, UserAgent, RandomAgent
from shobu import Shobu, Player


def play_game(black_player: Agent, white_player: Agent, initial_board_state=Shobu.starting_position()):
	board = initial_board_state
	while True:
		print(board)

		if (winner := board.check_winner()) is not None:
			print(f"The winner is {winner}.")
			return
		if board.next_mover == Player.BLACK:
			move = black_player.move(board)
		else:
			move = white_player.move(board)
		board = board.apply(move)

		print(f"Next move is {move}")
		print("\n---------------------------------")


if __name__ == '__main__':
	play_game(UserAgent(), RandomAgent())
