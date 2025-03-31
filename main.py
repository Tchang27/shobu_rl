from agent import UserAgent, RandomAgent, RLAgent
from eval import round_robin


if __name__ == '__main__':
	# Hopefully our players will be more interesting than this :)
	_, elos = round_robin([
		RandomAgent(),
		RLAgent("model_weights/ppo_checkpoint_33000.pth"), 
		RLAgent("model_weights/ppo_checkpoint_50000.pth"), 
		RLAgent("model_weights/ppo_checkpoint_77000.pth"), 
	],
	num_rounds=20,
	max_moves_per_game=100
	)

	print(elos)
