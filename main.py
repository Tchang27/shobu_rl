from agent import UserAgent, RandomAgent, RLAgent, MCTSAgent
from eval import round_robin, n_game_match


if __name__ == '__main__':
	# Hopefully our players will be more interesting than this :)
	_, elos = n_game_match(
		RandomAgent(),
		MCTSAgent("model_weights/mcts_checkpoint_312.pth"), 
		n=30,
		max_moves=100
	)

	print(elos)
