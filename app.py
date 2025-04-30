from flask import Flask, request, send_file
from agent import MCTSAgent, RandomAgent
from shobu import Player, Shobu
from flask_cors import CORS, cross_origin

build_path = "./shobu_ui/build"
app = Flask(__name__, static_folder=f"{build_path}/static")
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

started_procs = False

agents = {
    "random": RandomAgent(),
    "1k": MCTSAgent("mcts_checkpoint_1000.pth", cpuct=1.0),
    "20k": MCTSAgent("mcts_checkpoint_20000_noisy_random.pth", cpuct=1.0),
    "40k": MCTSAgent("mcts_checkpoint_40000_explore_random.pth", cpuct=1.0),
    "84k": MCTSAgent("mcts_checkpoint_84000_explore_random.pth", cpuct=1.1)
}

def do_work(board, user_color, desired_agent):
    move = None
    legal_moves_str = []
    winner = board.check_winner()
    if winner is not None:
        return winner, None, []
    if board.next_mover != user_color:
        move = agents[desired_agent].move(board, 1000) # half ply to some big number so i don't accidentally enable temp
        board = board.apply(move)
    legal_moves = board.move_gen()
    legal_moves_str = [str(move) for move in legal_moves]
    str_move = move if move is None else str(move)
    winner = board.check_winner()
    return winner, str_move, legal_moves_str

@app.route("/")
@cross_origin()
def entry():
    print("user loaded /")
    return send_file(f"{build_path}/index.html")

@app.route("/game_start")
@cross_origin()
def game_start():
    user_playing = request.args.get('playing')
    board_str = request.args.get('position')
    user_color = Player.BLACK
    if user_playing == 'w':
        user_color = Player.WHITE
    desired_agent = request.args.get('agent')
    if user_playing is None or desired_agent is None or board_str is None:
        return {"success": False}
    if desired_agent not in agents:
        return {"success": False}

    print(f"User joined, playing {user_color} against {desired_agent}")
    board = Shobu.from_str(board_str)
    winner, move_str, legal_moves_str = do_work(board, user_color, desired_agent)
    return {"success": True, "server_move": move_str, "legal_moves": legal_moves_str, "winner": winner if winner is None else "black" if winner == Player.BLACK else "white"}

@app.route("/player_move")
@cross_origin()
def player_move():
    board_str = request.args.get('position')
    user_playing = request.args.get('playing')
    desired_agent = request.args.get('agent')
    user_color = Player.BLACK
    if user_playing == 'w':
        user_color = Player.WHITE
    board = Shobu.from_str(board_str)
    board.next_mover = Player(not user_color.value)
    winner, move_str, legal_moves_str = do_work(board, user_color, desired_agent)
    return {"success": True, "server_move": move_str, "legal_moves": legal_moves_str, "winner": winner if winner is None else "black" if winner == Player.BLACK else "white"}
