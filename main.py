from agent import UserAgent, RandomAgent, RLAgent, MCTSAgent
from eval import round_robin, n_game_match, play_game


if __name__ == '__main__':
	# Hopefully our players will be more interesting than this :)
	res = n_game_match(
		MCTSAgent("model_weights/mcts_checkpoint_8100_v2.pth"), 
		MCTSAgent("model_weights/mcts_checkpoint_6500_v2.pth"),
		#RandomAgent(), 
		n=50,
		max_moves=32
    )
	print(res[0])
	print(res[1])
	print(res[2])
	print(res[3])

	# _, elos = round_robin(
	# 	[RandomAgent(),
	# 	MCTSAgent("model_weights/mcts_checkpoint_500.pth")], 
	# 	num_rounds=30,
	# )

	# print(elos)
    
	# play_game(
	# 	MCTSAgent("model_weights/mcts_checkpoint_7900_v2.pth"), 
	# 	MCTSAgent("model_weights/mcts_checkpoint_7000_v2.pth"),
	# 	#UserAgent(),
	# 	max_moves = 32,
	# 	print_info = True,
	# 	)
