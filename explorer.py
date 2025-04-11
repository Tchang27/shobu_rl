#!/usr/bin/env python

import sys
import os
import readline

import torch
from mcts_simul import MCTree
from models import Shobu_MCTS
from shobu import Player, Shobu

"""
This module defines an interactive REPL for exploring tree search results. It
allows you to run MCTS from any arbitrary position and traverse up and down the
resulting tree's branches, examining node states at any point along the way
and learning about stats such as (and not limited to)
	(1) How often did we consider a particular move? (more often = we think it's
	    a strong move)
	(2) Did we continue searching deeper past this particular board state?
	(3) What did the policy net initially think about this move?
	(4) What was the value net's evaluation of this position (without searching
	    deeper)?
	(5) Taking search into account, what is our overall evaluation of this
	    position?

To get started, run `./explorer.py <path-to-model-checkpoint>`, and type "h"
or "help" to view the help menu.
"""

# A collection of interesting positions for testing tree search, model eval, etc

STARTING = Shobu.starting_position() # The default starting position
BLACK_MATE_IN_1       = ...          # Black can play b2f6NE2 and win
BLACK_MATED_IN_1      = ...          # Black must play b3f1E2 to avoid losing
MATERIAL_ADVANTAGE    = ...          # Black has a large material advantage, but no immediate win
MATERIAL_DISADVANTAGE = ...          # Black has a large material disadvantage, but no immediate loss
CENTRALIZATION        = ...          # Equal material, but black's pieces are centralized and more mobile
MARGINALIZATION       = ...          # Equal material, but pieces are on edge, easier to be pushed off

CUSTOM = Shobu.from_str(             # Just make sure the position is legal
	"""
	.... ....
	.... ....
	.... ....
	wb.. wb..

	.... ....
	.... ....
	.... ....
	wb.. wb..
	""", next_mover=Player.WHITE)

flip_white_nodes = True
model = None
device = None
commands = None
tree = None
nodes = []
moves = []
cur_node = -1

def _args(f):
	return f.__code__.co_varnames[:f.__code__.co_argcount]

def do_help():
	for k, (d, f) in commands.items():
		print(f"{k}: {d}\n    Usage: {k} {' '.join(_args(f))}")

def do_board(predefined_board_name):
	global tree, nodes, cur_node, moves
	name = predefined_board_name.upper()
	if name in globals() and isinstance(globals()[name], Shobu):
		board = globals()[name]
	else:
		print(f"Unrecognized board name: {name}")
		return
	tree = MCTree(model, board, device)
	nodes = [tree.root]
	moves = []
	cur_node = 0

def _node_to_str(n):
	return f"{'B' if n.player == Player.BLACK else 'W'}(visits: {str(n.num_visits).rjust(3)}, avg_reward: {(n.total_reward / n.num_visits if n.num_visits != 0 else 0):+0.3f}, prior: {n.prior:+0.3f}, value: {n.value if n.value is None else f'{n.value:+0.3f}'}){'+' if n.is_expanded else ''}"

def do_print_node():
	if cur_node == -1:
		print("No nodes")
		return
	n = nodes[cur_node]
	print(_node_to_str(n))

def do_show_position():
	if cur_node == -1:
		print("No nodes")
		return
	n = nodes[cur_node]
	if flip_white_nodes and n.player == Player.WHITE: n.state.flip()
	print(n.state)
	if flip_white_nodes and n.player == Player.WHITE: n.state.flip()

def _do_search(num_simulations, use_noise):
	global nodes, cur_node, moves
	if cur_node == -1:
		print("No nodes")
		return

	num_simulations = int(num_simulations)
	nodes = [nodes[cur_node]]
	tree.root = nodes[0]
	moves = []
	cur_node = 0
	tree.root.num_visits = 0
	tree.root.total_reward = 0
	tree.root.children = {}
	tree.root.is_expanded = False
	tree.search(num_simulations, noise=use_noise)

def do_search(num_simulations):
	_do_search(num_simulations, False)

def do_dirichlet_search(num_simulations):
	_do_search(num_simulations, True)

def _lpad(x):
	return str(x).rjust(7)

def do_children():
	if cur_node == -1:
		print("No nodes")
		return
	n = nodes[cur_node]
	if not n.is_expanded:
		print("Node has no children. Consider executing a search from here.")
	children = list(n.children.items())
	children.sort(key=lambda e: e[1].num_visits, reverse=True)
	for k, v in children:
		if flip_white_nodes and n.player == Player.WHITE: k.flip()
		print(f"{_lpad(k)}: {_node_to_str(v)}")
		if flip_white_nodes and n.player == Player.WHITE: k.flip()

def do_trace():
	m = ["<root>"] + moves
	for i, (move, node) in enumerate(zip(m, nodes)):
		print('@' if i == cur_node else ' ', f"[{i}]", _lpad(move), "=>", _node_to_str(node))

def do_up():
	global cur_node
	if cur_node == -1:
		print("No nodes")
		return
	if cur_node == 0:
		print("Already at root")
		return
	cur_node -= 1

def do_down(move):
	global cur_node, nodes, moves
	if cur_node == -1:
		print("No nodes")
		return
	move = move.upper()
	if cur_node < len(moves) - 1 and moves[cur_node].upper() == move:
		cur_node += 1
		return
	n = nodes[cur_node]
	next_k, next_v = None, None
	for k, v in n.children.items():
		if flip_white_nodes and n.player == Player.WHITE: k.flip()
		strk = str(k)
		if flip_white_nodes and n.player == Player.WHITE: k.flip()
		if strk.upper() == move:
			next_k, next_v = strk, v
			break
	if next_k == None:
		print(f"{move} is not a child node.")
		return
	nodes = nodes[:cur_node + 1]
	moves = moves[:cur_node] # ???
	nodes.append(next_v)
	moves.append(next_k)
	cur_node += 1

def do_jump(trace_index):
	global cur_node
	if cur_node == -1:
		print("No nodes")
		return
	idx = int(trace_index)
	if idx < 0 or idx >= len(nodes):
		print(f"Jump target {idx} out of range.")
		return
	cur_node = idx

def do_sample(tau):
	if cur_node == -1:
		print("No nodes")
		return
	n = nodes[cur_node]
	tau = float(tau)
	move = n.sample_move(tau)
	if flip_white_nodes and n.player == Player.WHITE: move.flip()
	print(move)
	if flip_white_nodes and n.player == Player.WHITE: move.flip()

commands = {
	"board": ("Discards the current tree if it exists and sets the given board position as the current node.", do_board),
	"print_node": ("Print information about the current node.", do_print_node),
	"view_position": ("Print information about the current node.", do_show_position),
	"search": ("Sets the current position as the root node (discarding the current tree), and executes a tree search from the current position (DIRICHLET DISABLED).", do_search),
	"noise_search": ("Same thing as search, except searches with dirichlet noise enabled.", do_dirichlet_search),
	"children": ("List the children of this node", do_children),
	"trace": ("Show the currently explored path (all nodes from root to deepest explored child).", do_trace),
	"up": ("Go to the parent node of this node.", do_up),
	"down": ("Go to a child node of the current node, by specifying a move.", do_down),
	"jump": ("Jump to a particular node in the trace.", do_jump),
	"get_move": ("Sample the next move from this position. Choose tau=0 for strongest selection.", do_sample),
	"help": ("Prints this help message.", do_help)
}

def repl(checkpoint_path, initial_position=None, first_mover="black"):
	global device, model
	device = torch.device("cpu")
	model = Shobu_MCTS(device)
	model.to(device)
	model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
	model.eval()

	if initial_position is not None:
		global tree, nodes, cur_node, moves
		with open(initial_position, "r") as file:
				board = Shobu.from_str(file.read())
		tree = MCTree(model, board, device)
		nodes = [tree.root]
		moves = []
		cur_node = 0
	else:
		do_board("starting")

	while True:
		try:
			cmd = input("> ").strip().lower().split()
		except EOFError:
			print()
			return
		if len(cmd) == 0:
			continue
		if cmd[0] == "clear":
			os.system('cls' if os.name=='nt' else 'clear')
			continue
		candidates = [(k, f) for k, (_, f) in commands.items() if k.startswith(cmd[0])]
		if len(candidates) > 1:
			print(f"Ambiguous command: {', '.join(k for (k, _) in candidates)}")
		elif len(candidates) == 0:
			print(f"Unrecognized command: {cmd[0]}")
		else:
			f = candidates[0][1]
			args = _args(f)
			if len(args) != len(cmd[1:]):
				print(f"Usage: {candidates[0][0]} {' '.join(args)}")
			else:
				try: f(*cmd[1:])
				except Exception as e: print(e)


# If you are editing / reading this file, you can fold this if statement to
# hide all of the lengthy board representations
if True:
	BLACK_MATE_IN_1 = Shobu.from_str(
		"""
		wwww ....
		.... ..w.
		.... .b..
		b... .b..

		w... bbbb
		b... ....
		.b.b .w.w
		.ww. ..w.
		""", next_mover=Player.BLACK)

	BLACK_MATED_IN_1 = Shobu.from_str(
		"""
		ww.. ww..
		.... .w..
		.... ....
		bbbb bbbb

		.... ..ww
		.b.. ....
		..b. .w..
		wwww .b..
		""", next_mover=Player.BLACK)
	
	TEST = Shobu.from_str(
		"""
		...w ....
		.... ....
		.... ...w
		b... b...

		..b. ww..
		.b.. ....
		w... b...
		b..w .b.w
		""", next_mover=Player.BLACK)

	MATERIAL_ADVANTAGE = Shobu.from_str(
		"""
		w... w...
		.... ....
		.... ....
		bbbb bbbb

		w... w...
		.... ....
		.... ....
		bbbb bbbb
		""", next_mover=Player.BLACK)

	MATERIAL_DISADVANTAGE = Shobu.from_str(
		"""
		wwww wwww
		.... ....
		.... ....
		b... b...

		wwww wwww
		.... ....
		.... ....
		b... b...
		""", next_mover=Player.BLACK)

	CENTRALIZATION = Shobu.from_str(
		"""
		..w. ..w.
		.... ....
		.b.. .b..
		.... ....

		.... ....
		.... ....
		.b.. ..b.
		...w w...
		""", next_mover=Player.BLACK)

	MARGINALIZATION = Shobu.from_str(
		"""
		..b. ..b.
		.... ....
		.w.. .w..
		.... ....

		.... ....
		.... ....
		.w.. ..w.
		...b b...
		""", next_mover=Player.BLACK)

if __name__ == "__main__":
	if 2 <= len(sys.argv) <= 4: repl(*sys.argv[1:])
	else: print(f"Usage: {sys.argv[0]} <path-to-model-checkpoint> [<path-to-initial-position>] [<first-mover>]")
