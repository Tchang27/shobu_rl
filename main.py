from agent import UserAgent, RandomAgent, RLAgent, MCTSAgent, MCTSConvAgent
from eval import round_robin, n_game_match, play_game, tiered_arena

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
	# tiered_arena(
	# 	[RandomAgent("fred"), RandomAgent("george"), RandomAgent("matthew")],
	# 	[RandomAgent("jennifer"), RandomAgent("natalie"), RandomAgent("astrid")],
	# 	"."
	# )
    
	play_game(
		#UserAgent(),
		MCTSConvAgent("model_weights/mcts_conv_checkpoint_32000.pth"),
		MCTSAgent("model_weights/mcts_checkpoint_30400_noisy_random.pth"),
		#MCTSAgent("model_weights/mcts_checkpoint_63400_explore_random.pth",cpuct=1.1),
		print_info = True
	)
