from agent import UserAgent, RandomAgent
from eval import round_robin


if __name__ == '__main__':
	# Hopefully our players will be more interesting than this :)
	_, elos = round_robin([
		RandomAgent(),
		RandomAgent(),
		RandomAgent(),
		RandomAgent(),
		RandomAgent(),
		RandomAgent(),
		RandomAgent(),
		RandomAgent(),
		RandomAgent(),
		RandomAgent()
	])

	print(elos)
