from agent import UserAgent, RandomAgent, RLAgent, MCTSAgent
from eval import round_robin, n_game_match, play_game


if __name__ == '__main__':
	# Hopefully our players will be more interesting than this :)
	# _, elos = n_game_match(
	# 	RandomAgent(),
	# 	MCTSAgent("model_weights/mcts_checkpoint_312.pth"), 
	# 	n=30,
	# 	max_moves=100
	# )

	# _, elos = round_robin(
	# 	[RandomAgent(),
	# 	MCTSAgent("model_weights/mcts_checkpoint_500.pth")], 
	# 	num_rounds=30,
	# )

	# print(elos)
    
	play_game(
		MCTSAgent("model_weights/mcts_checkpoint_4300_v2.pth"), 
		MCTSAgent("model_weights/mcts_checkpoint_4100.pth"),
		max_moves = 64,
		print_info = True,
		)
