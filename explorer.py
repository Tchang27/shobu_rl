#!/usr/bin/env python

import sys
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

Here's how a session generally may look ('>' means user input):

> b starting
> v
8 ○○○○|○○○○
7 · · · · | · · · ·
6 · · · · | · · · ·
5 ●●●●|●●●●
  --------+--------
4 ○○○○|○○○○
3 · · · · | · · · ·
2 · · · · | · · · ·
1 ●●●●|●●●●
  a b c d   e f g h

Next to move: ●
> s 800
> c
 f1b5N2: B(visits:  42, avg_reward: -0.027, prior: +0.004, value: -0.083)+
 g1b5NW: B(visits:  34, avg_reward: -0.025, prior: +0.006, value: -0.182)+
 g1b5NE: B(visits:  25, avg_reward: -0.021, prior: +0.005, value: -0.235)+
 g1b5N2: B(visits:  19, avg_reward: -0.019, prior: +0.005, value: -0.197)+
 a1f1N2: B(visits:  18, avg_reward: -0.022, prior: +0.003, value: -0.211)+
 f1a1N2: B(visits:  18, avg_reward: -0.022, prior: +0.004, value: -0.211)+
 c1f5NW: B(visits:  17, avg_reward: -0.015, prior: +0.005, value: -0.181)+
 ...output truncated...
> down a1f1n2
> t
  [0]  <root> => B(visits: 800, avg_reward: +0.024, prior: +0.000, value: +0.010)+
@ [1]  a1f1N2 => B(visits:  18, avg_reward: -0.022, prior: +0.003, value: -0.211)+
> c
  g1a1N: B(visits:  14, avg_reward: -0.026, prior: +0.008, value: -0.086)+
  e1a1N: B(visits:   2, avg_reward: +0.079, prior: +0.008, value: -0.039)+
  h1a1N: B(visits:   1, avg_reward: +0.028, prior: +0.008, value: +0.028)+
	...output truncated...
> do g1a1N
> v
8 ○. ○○|○○○ ·
7 · ○· · | · · · ○
6 · · · · | · · · ·
5 ●●●●|●●●●
  --------+--------
4 ○○○○|○○○○
3 ●· · · | · ●· ·
2 · · · · | · · · ·
1 · ●●●|●· ●●
  a b c d   e f g h

Next to move: ●
> t
  [0]  <root> => B(visits: 800, avg_reward: +0.024, prior: +0.000, value: +0.010)+
  [1]  a1f1N2 => B(visits:  18, avg_reward: -0.022, prior: +0.003, value: -0.211)+
@ [2]   g1a1N => B(visits:  14, avg_reward: -0.026, prior: +0.008, value: -0.086)+
> j 0
> t
@ [0]  <root> => B(visits: 800, avg_reward: +0.024, prior: +0.000, value: +0.010)+
  [1]  a1f1N2 => B(visits:  18, avg_reward: -0.022, prior: +0.003, value: -0.211)+
  [2]   g1a1N => B(visits:  14, avg_reward: -0.026, prior: +0.008, value: -0.086)+
"""

# A collection of interesting positions for testing tree search, model eval, etc

STARTING = Shobu.starting_position() # The default starting position
BLACK_MATE_IN_1  = ...               # Black can play b2f6NE2 and win
BLACK_MATED_IN_1 = ...               # Black must play b3f1E2 to avoid losing

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
	return f"{'B' if n.player == Player.BLACK else 'W'}(visits: {str(n.num_visits).rjust(3)}, avg_reward: {(n.total_reward / n.num_visits if n.num_visits != 0 else 0):+0.3f}, prior: {n.prior:+0.3f}, value: {n.value:+0.3f}){'+' if n.is_expanded else ''}"

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
	print(n.state)

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
		print(f"{_lpad(k)}: {_node_to_str(v)}")

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
		if str(k).upper() == move:
			next_k, next_v = k, v
			break
	if next_k == None:
		print(f"{move} is not a child node.")
		return
	nodes = nodes[:cur_node + 1]
	moves = moves[:cur_node] # ???
	nodes.append(next_v)
	moves.append(str(next_k))
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

commands = {
	"board": ("Discards the current tree if it exists and sets the given board position as the current node.", do_board),
	"print_node": ("Print information about the current node.", do_print_node),
	"view_position": ("Print information about the current node.", do_show_position),
	"search": ("Sets the current position as the root node (discarding the current tree), and executes a tree search from the current position (DIRICHLET DISABLED).", do_search),
	"dsearch": ("Same thing as search, except searches with dirichlet noise enabled.", do_dirichlet_search),
	"children": ("List the children of this node", do_children),
	"trace": ("Show the currently explored path (all nodes from root to deepest explored child).", do_trace),
	"up": ("Go to the parent node of this node.", do_up),
	"down": ("Go to a child node of the current node, by specifying a move.", do_down),
	"jump": ("Jump to a particular node in the trace.", do_jump),
	"help": ("Prints this help message.", do_help)
}

def repl(checkpoint_path):
	global device, model
	device = torch.device("cpu")
	model = Shobu_MCTS(device)
	model.to(device)
	model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
	model.eval()

	while True:
		try:
			cmd = input("> ").strip().lower().split()
		except EOFError:
			print()
			return
		if len(cmd) == 0:
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
		wwww ww..
		.... .w..
		.... ....
		bbbb bbbb

		.... ..ww
		.b.. ....
		..b. .w..
		wwww .b..
		""", next_mover=Player.BLACK)

if __name__ == "__main__":
	if len(sys.argv) == 2: repl(sys.argv[1])
	else: print(f"Usage: {sys.argv[0]} <path-to-model-checkpoint>")
