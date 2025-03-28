from agent import UserAgent, RandomAgent, RLAgent
from eval import round_robin


if __name__ == '__main__':
	# Hopefully our players will be more interesting than this :)
	_, elos = round_robin([
		RandomAgent(),
		RLAgent("model_weights/ppo_checkpoint_2000.pth"), 
		RLAgent("model_weights/ppo_checkpoint_12000.pth"),
	])

	print(elos)
