from agent import UserAgent, RandomAgent, RLAgent, MCTSAgent
from eval import round_robin, n_game_match, play_game


if __name__ == '__main__':
	# Hopefully our players will be more interesting than this :)
	# res = n_game_match(
	# 	MCTSAgent("model_weights/mcts_checkpoint_38100_explore_random.pth"), 
	# 	MCTSAgent("model_weights/mcts_checkpoint_46100_more_random.pth"),
	# 	#RandomAgent(), 
	# 	n=100,
	# 	max_moves=32
    # )
	# print(res[0])
	# print(res[1])
	# print(res[2])
	# print(res[3])

	# _, elos = round_robin(
	# 	[RandomAgent(),
	# 	MCTSAgent("model_weights/mcts_checkpoint_500.pth")], 
	# 	num_rounds=30,
	# )

	# print(elos)
    
	play_game(
		UserAgent(),
		MCTSAgent("model_weights/mcts_checkpoint_48700_more_random.pth"),
		#MCTSAgent("model_weights/mcts_checkpoint_31600_noisy_random.pth"),
		max_moves = 32,
		print_info = True,
		)
