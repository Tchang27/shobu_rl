from agent import UserAgent, RandomAgent, RLAgent, MCTSAgent
from eval import round_robin


if __name__ == '__main__':
	# Hopefully our players will be more interesting than this :)
	_, elos = round_robin([
		RandomAgent(),
		#RLAgent("model_weights/ppo_checkpoint_33000.pth"), 
		MCTSAgent("model_weights/mcts_checkpoint_10.pth"), 
	],
	num_rounds=10,
	max_moves_per_game=100
	)

	print(elos)
