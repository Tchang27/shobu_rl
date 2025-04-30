import React, { useEffect, useState } from 'react';
import './App.css';
import Board from './Board';

let dirmapping: any = {
  "n": [0, 1],
  "ne": [1, 1],
  "e": [1, 0],
  "se": [1, -1],
  "s": [0, -1],
  "sw": [-1, -1],
  "w": [-1, 0],
  "nw": [-1, 1],
  "n2": [0, 2],
  "ne2": [2, 2],
  "e2": [2, 0],
  "se2": [2, -2],
  "s2": [0, -2],
  "sw2": [-2, -2],
  "w2": [-2, 0],
  "nw2": [-2, 2]
};

function App() {
  const boardBoundingRect = useBoardBoundingRect();

  // somehow acquired from flask
  // let legalMoves = [
  //   "a1f2n2",
  //   "a1f2n",
  //   "a1f5ne2",
  //   "a1f5ne",
  //   "f8b8s"
  // ];
  let [legalMoves, setLegalMoves]: any = useState([]);

  // we need legalMoves for:
  // 1. given a passive piece, determine which agg pieces you can move
  // 2. given passive+agg, determine which squares you can move to.
  let passToAggMap: any = {};
  let paToSquaresAggMap: any = {};
  let paToSquaresPassMap: any = {};
  for (let move of legalMoves) {
    move = move.toLowerCase();
    let passMove = move.slice(0, 2);
    if (!(passMove in passToAggMap)) {
      passToAggMap[passMove] = [];
    }
    let aggMove = move.slice(2, 4);
    passToAggMap[passMove].push(aggMove);
    let paMove = move.slice(0, 4);
    if (!(paMove in paToSquaresAggMap)) {
      paToSquaresAggMap[paMove] = [];
      paToSquaresPassMap[paMove] = [];
    }
    let dir = dirmapping[move.slice(4)];
    paToSquaresAggMap[paMove].push(applyDir(dir, aggMove));
    paToSquaresPassMap[paMove].push(applyDir(dir, passMove));
  }

  let [interactionState, setInteractionState] = useState("entry"); // "starting", "selected_passive", "dragging", "waiting", "entry", "ended"
  let [selectedPassiveSquare, setSelectedPassiveSquare] = useState("");
  let [selectedAggressiveSquare, setSelectedAggressiveSquare] = useState("");
  let [aggHints, setAggHints] = useState([]);
  let [moveHints, setMoveHints] = useState([]);
  let [allowedMoves, setAllowedMoves] = useState([]);
  let [dragCoords, setDragCoords] = useState([0, 0]);
  let [winner, setWinner] = useState("");

  let [board_state, setBoardState] = useState([
    ["w0", "w1",  "w2",  "w3",  "w4",  "w5",  "w6",  "w7"],
    ["",     "",    "",    "",    "",    "",    "",    ""],
    ["",     "",    "",    "",    "",    "",    "",    ""],
    ["b8", "b9", "b10", "b11", "b12", "b13", "b14", "b15"],
    ["w8", "w9", "w10", "w11", "w12", "w13", "w14", "w15"],
    ["",     "",    "",    "",    "",    "",    "",    ""],
    ["",     "",    "",    "",    "",    "",    "",    ""],
    ["b0", "b1",  "b2",  "b3",  "b4",  "b5",  "b6",  "b7"],
  ]);
  let [changedSquares, setChangedSquares] = useState([]);
  let [im_playing_as, setImPlayingAs] = useState("b");
  let [playingAgainst, setPlayingAgainst] = useState("random"); // random, 1k, 20k, 40k, 84k
  let flipped = im_playing_as === "w";

  let width = boardBoundingRect.width;
  let square_centers = [...Array(8)].map(e => Array(8));
  let padding_size = width / 40;
  let gutter_size = (width * 37 / 40 - 4) / 150;
  let square_size = (width - (3 * (padding_size + 1) + 1) - 10 * gutter_size) / 8;

  for (let i = 0; i < 8; i++) {
    for (let j = 0; j < 8; j++) {
      square_centers[flipped ? i : 8 - i - 1][flipped ? 8 - j - 1 : j] = [boardBoundingRect.y + padding_size + 1 + (i + 1) * gutter_size + square_size * i + square_size / 2, boardBoundingRect.x + padding_size + 1 + (j + 1) * gutter_size  + square_size * j + square_size / 2];
      if (i > 3) square_centers[flipped ? i : 8 - i - 1][flipped ? 8 - j - 1 : j][0] += padding_size + gutter_size + 2;
      if (j > 3) square_centers[flipped ? i : 8 - i - 1][flipped ? 8 - j - 1 : j][1] += padding_size + gutter_size + 2;
    }
  }

  let piece_size = width / 12;
  let pieces = [];
  for (let i = 0; i < 8; i++) {
    for (let j = 0; j < 8; j++) {
      let sq = coordsToSquare([j, i]);
      if (board_state[8 - i - 1][j].startsWith("w")) {
        if (interactionState === "dragging" && (selectedPassiveSquare === sq || selectedAggressiveSquare === sq)) {
          pieces.push(<div id={board_state[8 - i - 1][j]} style={{ cursor: "pointer", height: piece_size + "px", position: "absolute", top: square_centers[i][j][0] - piece_size / 2 + 'px', left: square_centers[i][j][1] - piece_size / 2 + 'px', width: piece_size + "px", borderRadius: "50%", backgroundColor: "#658594", border: "solid #16161D", borderWidth: width / 120 + 'px', opacity: "50%" }}></div>);
        } else {
          pieces.push(<div id={board_state[8 - i - 1][j]} style={{ cursor: "pointer", height: piece_size + "px", position: "absolute", top: square_centers[i][j][0] - piece_size / 2 + 'px', left: square_centers[i][j][1] - piece_size / 2 + 'px', width: piece_size + "px", borderRadius: "50%", backgroundColor: "#658594", border: "solid #16161D", borderWidth: width / 120 + 'px' }}></div>);
        }
      } else if (board_state[8 - i - 1][j].startsWith("b")) {
        if (interactionState === "dragging" && (selectedPassiveSquare === sq || selectedAggressiveSquare === sq)) {
          pieces.push(<div id={board_state[8 - i - 1][j]} style={{ cursor: "pointer", height: piece_size + "px", position: "absolute", top: square_centers[i][j][0] - piece_size / 2 + 'px', left: square_centers[i][j][1] - piece_size / 2 + 'px', width: piece_size + "px", borderRadius: "50%", backgroundColor: "#223249", border: "solid #16161D", borderWidth: width / 120 + 'px', opacity: "50%" }}></div>);
        } else {
          pieces.push(<div id={board_state[8 - i - 1][j]} style={{ cursor: "pointer", height: piece_size + "px", position: "absolute", top: square_centers[i][j][0] - piece_size / 2 + 'px', left: square_centers[i][j][1] - piece_size / 2 + 'px', width: piece_size + "px", borderRadius: "50%", backgroundColor: "#223249", border: "solid #16161D", borderWidth: width / 120 + 'px' }}></div>);
        }
      }
    }
  }

  let square_markers = [];
  for (let i = 0; i < 8; i++) {
    for (let j = 0; j < 8; j++) {
      let sq = coordsToSquare([j, i]);
      if (changedSquares.slice(0, 2).includes(sq as never)) {
        square_markers.push(<div style={{ height: square_size + "px", position: "absolute", top: square_centers[i][j][0] - square_size / 2 + 'px', left: square_centers[i][j][1] - square_size / 2 + 'px', width: square_size + "px", backgroundColor: "#98BB6C", opacity: "40%", borderRadius: width / 180 + 'px'}}></div>)
      } else if (changedSquares.slice(2, 4).includes(sq as never)) {
        square_markers.push(<div style={{ height: square_size + "px", position: "absolute", top: square_centers[i][j][0] - square_size / 2 + 'px', left: square_centers[i][j][1] - square_size / 2 + 'px', width: square_size + "px", backgroundColor: "#76946A", opacity: "80%", borderRadius: width / 180 + 'px'}}></div>)
      }
      if (interactionState === "starting") continue;
      else if (interactionState === "selected_passive") {
        if (sq === selectedPassiveSquare) {
          square_markers.push(<div style={{ height: square_size + "px", position: "absolute", top: square_centers[i][j][0] - square_size / 2 + 'px', left: square_centers[i][j][1] - square_size / 2 + 'px', width: square_size + "px", backgroundColor: "#98BB6C", opacity: "80%", borderRadius: width / 180 + 'px'}}></div>)
        } else if (aggHints !== undefined && aggHints.includes(sq as never)) {
          square_markers.push(<div className='cornerborder' style={{ height: square_size + "px", position: "absolute", top: square_centers[i][j][0] - square_size / 2 + 'px', left: square_centers[i][j][1] - square_size / 2 + 'px', width: square_size + "px", borderRadius: width / 180 + 'px'}}></div>);
        }
      } else if (interactionState === "dragging") {
        if (sq === selectedPassiveSquare || sq === selectedAggressiveSquare) {
          square_markers.push(<div style={{ height: square_size + "px", position: "absolute", top: square_centers[i][j][0] - square_size / 2 + 'px', left: square_centers[i][j][1] - square_size / 2 + 'px', width: square_size + "px", backgroundColor: "#98BB6C", opacity: "80%", borderRadius: width / 180 + 'px'}}></div>);
        } else if (moveHints !== undefined && moveHints.includes(sq as never)) {
          if (board_state[8 - i - 1][j]) {
            square_markers.push(<div className='cornerborder2' style={{ height: square_size + "px", position: "absolute", top: square_centers[i][j][0] - square_size / 2 + 'px', left: square_centers[i][j][1] - square_size / 2 + 'px', width: square_size + "px", borderRadius: width / 180 + 'px'}}></div>);
          } else {
            square_markers.push(<div style={{position: "absolute", width: width / 60 + "px", height: width / 60 + "px", borderRadius: "50%", backgroundColor: "#76946A", top: square_centers[i][j][0] - width / 120 + 'px', left: square_centers[i][j][1] - width / 120 + 'px'}}></div>);
          }
        }
      }
    }
  }

  let drag_pieces = [];
  if (interactionState === "dragging") {
    let color = im_playing_as === "b" ? "#223249" : "#658594";
    // compute offset
    let coordsP = squareToCoords(selectedPassiveSquare);
    let coordsA = squareToCoords(selectedAggressiveSquare);
    let idP = board_state[8 - coordsP[1] - 1][coordsP[0]];
    let idA = board_state[8 - coordsA[1] - 1][coordsA[0]];
    let rectP = document.getElementById(idP)?.getBoundingClientRect();
    let rectA = document.getElementById(idA)?.getBoundingClientRect();
    if (rectA && rectP) {
    let offsetY = rectP.top - rectA.top;
    let offsetX = rectP.left - rectA.left;
    drag_pieces.push(<div style={{ cursor: "pointer", height: piece_size + "px", position: "absolute", top: dragCoords[1] - piece_size / 2 + 'px', left: dragCoords[0] - piece_size / 2 + 'px', width: piece_size + "px", borderRadius: "50%", backgroundColor: color, border: "solid #16161D", borderWidth: width / 120 + 'px' }}></div>);
    drag_pieces.push(<div style={{ cursor: "pointer", height: piece_size + "px", position: "absolute", top: offsetY + dragCoords[1] - piece_size / 2 + 'px', left: offsetX + dragCoords[0] - piece_size / 2 + 'px', width: piece_size + "px", borderRadius: "50%", backgroundColor: color, border: "solid #16161D", borderWidth: width / 120 + 'px' }}></div>);
    }
  }

  useEffect(() => {
    if (interactionState === "entry") return;
    function resolve(mouseX: number, mouseY: number) {
      for (let i = 0; i < 8; i++) {
        for (let j = 0; j < 8; j++) {
          let minY = square_centers[i][j][0] - square_size / 2;
          let maxY = square_centers[i][j][0] + square_size / 2;
          let minX = square_centers[i][j][1] - square_size / 2;
          let maxX = square_centers[i][j][1] + square_size / 2;
          if (mouseX >= minX && mouseX <= maxX && mouseY >= minY && mouseY <= maxY) {
            return [j, i];
          }
        }
      }
      return null;
    }

    function doMouseMove(x: any, y: any) {
      if (interactionState !== "dragging") return;
      setDragCoords([x, y]);
    }
    function handleMouseMove(e: MouseEvent) {
      e.preventDefault();
      doMouseMove(e.clientX, e.clientY);
    }
    function handleTouchMove(e: TouchEvent) {
      e.preventDefault();
      doMouseMove(e.targetTouches[0].clientX, e.targetTouches[0].clientY);
    }
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("touchmove", handleTouchMove);

    function doMouseDown(x: any, y: any) {
      let coords = resolve(x, y);
      if (coords == null) {
        if (interactionState === "waiting" || interactionState === "ended") return;
        setInteractionState("starting");
        return;
      }
      let sq = coordsToSquare(coords);
      if (interactionState === "starting" || interactionState === "selected_passive") {
        if (interactionState !== "starting" && aggHints !== undefined && aggHints.includes(sq as never)) {
          setSelectedAggressiveSquare(sq);
          setDragCoords([x, y]);
          let pa = selectedPassiveSquare + sq;
          setMoveHints((paToSquaresAggMap[pa] || []).concat(paToSquaresPassMap[pa] || []));
          setAllowedMoves(paToSquaresAggMap[pa] || []);
          setInteractionState("dragging");
        } else if (((!flipped && coords[1] < 4) || (flipped && coords[1] >= 4)) && board_state[8 - coords[1] - 1][coords[0]].startsWith(im_playing_as)) {
          setSelectedPassiveSquare(sq);
          setAggHints(passToAggMap[sq]);
          setInteractionState("selected_passive");
        } else {
          setInteractionState("starting");
        }
      }
    }

    function handleMouseDown(e: MouseEvent) {
      e.preventDefault();
      doMouseDown(e.clientX, e.clientY);
    }
    function handleTouchDown(e: TouchEvent) {
      e.preventDefault();
      doMouseDown(e.targetTouches[0].clientX, e.targetTouches[0].clientY);
    }
    window.addEventListener("mousedown", handleMouseDown);
    window.addEventListener("touchstart", handleTouchDown);

    function doMouseUp() {
      if (interactionState !== "dragging") return;
      let coords = resolve(dragCoords[0], dragCoords[1]);
      if (coords == null) {
        setInteractionState("selected_passive");
        return;
      }
      let sq = coordsToSquare(coords);
      if (allowedMoves.includes(sq as never)) {
        // figure out move notation
        setInteractionState("waiting");
        let aggCoords = squareToCoords(selectedAggressiveSquare);
        let xdiff = coords[0] - aggCoords[0];
        let ydiff = coords[1] - aggCoords[1];
        let dir = ((ydiff < 0) ? "s" : ((ydiff > 0) ? "n" : "")) + ((xdiff < 0) ? "w" : ((xdiff > 0) ? "e" : ""));
        if (Math.abs(xdiff) === 2 || Math.abs(ydiff) === 2) dir += "2";
        let move = selectedPassiveSquare + selectedAggressiveSquare + dir;
        // TODO applyMove + send req + setInteractionState("waiting")
        let [newBoardState, changedSquares] = applyMove(board_state, move);
        setBoardState(newBoardState);
        setChangedSquares(changedSquares);
        fetch(`/player_move?position=${boardToString(newBoardState)}&playing=${im_playing_as}&agent=${playingAgainst}`, { method: 'get', mode: 'cors' }).then((response) => {
          if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
          return response.json();
        })
        .then((response) => {
          if (!response["success"]) {
            console.log("UH OH");
          }
          if (response["server_move"]) {
            let [newBoardState, changedSquares] = applyMove(board_state, response["server_move"]);
            setBoardState(newBoardState);
            setChangedSquares(changedSquares);
          }
          if (response["winner"]) {
            setWinner(response["winner"]);
            setInteractionState("ended");
          } else {
            setLegalMoves(response["legal_moves"]);
            setInteractionState("starting");
          }
        });
      } else {
        setInteractionState("selected_passive");
        return;
      }
    }

    function handleMouseUp(e: MouseEvent) {
      e.preventDefault();
      doMouseUp();
    }
    function handleTouchUp(e: TouchEvent) {
      e.preventDefault();
      doMouseUp();
    }
    window.addEventListener("mouseup", handleMouseUp);
    window.addEventListener("touchend", handleTouchUp);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("touchmove", handleTouchMove);
      window.removeEventListener("mousedown", handleMouseDown);
      window.removeEventListener("touchstart", handleTouchDown);
      window.removeEventListener("mouseup", handleMouseUp);
      window.removeEventListener("touchend", handleTouchUp);
    }
  }, [im_playing_as, board_state, interactionState, square_centers, square_size, passToAggMap, aggHints, paToSquaresAggMap, paToSquaresPassMap, selectedPassiveSquare, allowedMoves, dragCoords, selectedAggressiveSquare, flipped, playingAgainst]);

  let interactionHint = "Hello!";
  switch (interactionState) {
    case "starting":
      interactionHint = "Click/tap on a passive piece on either of the two lower boards.";
      break;
    case "selected_passive":
      interactionHint = "Drag one of the highlighted aggressive pieces to make your move!";
      break;
    case "dragging":
      interactionHint = "Legal moves are marked on the board.";
      break;
    case "waiting":
      interactionHint = "Server is thinking...";
      break;
    case "ended":
      interactionHint = `Game ended; ${winner} won! Refresh this page to play again.`;
      break;
    default:
      break;
  }
  return (
    <div className="App">
      <h1>shōbu</h1>

      { interactionState === "entry" && (
        <div id="entry-box">
          <h1>shōbu</h1>
          <div id="entry-options">
            <h2 style={{marginTop: "0px"}}>Play as:</h2>
            <div style={{display: "flex"}}>
              <div className={"entry-btn " + ((im_playing_as === "b") ? "selected" : "")} onClick={() => setImPlayingAs("b")}>Black</div>
              <div className={"entry-btn " + ((im_playing_as === "w") ? "selected" : "")} onClick={() => setImPlayingAs("w")}>White</div>
            </div>
            <h2>Play against:</h2>
            <div className={"entry-btn " + ((playingAgainst === "random") ? "selected" : "")} onClick={() => setPlayingAgainst("random")}>Random Bot</div>
            <div className={"entry-btn " + ((playingAgainst === "1k") ? "selected" : "")} onClick={() => setPlayingAgainst("1k")}>ShabuShabu-1k</div>
            <div className={"entry-btn " + ((playingAgainst === "20k") ? "selected" : "")} onClick={() => setPlayingAgainst("20k")}>ShabuShabu-20k</div>
            <div className={"entry-btn " + ((playingAgainst === "40k") ? "selected" : "")} onClick={() => setPlayingAgainst("40k")}>ShabuShabu-40k</div>
            <div className={"entry-btn " + ((playingAgainst === "84k") ? "selected" : "")} onClick={() => setPlayingAgainst("84k")}>ShabuShabu-84k</div>

            <h2>Ready?</h2>
            <div className={"entry-submit-btn"} onClick={() => {
              document.addEventListener('touchmove', function(e) { e.preventDefault(); }, { passive:false });
              setInteractionState("waiting");
              window.scroll(0, 0);
              fetch(`/game_start?playing=${im_playing_as}&agent=${playingAgainst}&position=${boardToString(board_state)}`, { method: 'get', mode: 'cors' }).then((response) => {
                if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
                return response.json();
              })
              .then((response) => {
                if (!response["success"]) {
                  console.log("UH OH");
                }
                if (response["server_move"]) {
                  let [newBoardState, changedSquares] = applyMove(board_state, response["server_move"]);
                  setBoardState(newBoardState);
                  setChangedSquares(changedSquares);
                }
                if (response["winner"]) {
                  setWinner(response["winner"]);
                  setInteractionState("ended");
                } else {
                  setLegalMoves(response["legal_moves"]);
                  setInteractionState("starting");
                }
              });
            }}>Start game</div>
          </div>
        </div>
      )}

      <div id="board-container">
        <Board width={boardBoundingRect.width} flipped={flipped}/>
      </div>

      {/* {square_centers.flat().map(e => <div style={{position: "absolute", width: "10px", height: "10px", borderRadius: "50%", backgroundColor: "red", top: e[0] - 5 + 'px', left: e[1] - 5 + 'px'}}></div>)} */}
      {square_markers}
      {pieces}
      {drag_pieces}

      <p style={{fontSize: "18px", fontStyle: "italic", padding: "20px"}}>{interactionHint}</p>
    </div>
  );
}

// this function assumes the move is legal. all bets are off if not
function applyMove(boardState: any, moveStr: string) {
  moveStr = moveStr.toLowerCase();
  let pasCoordBegin = squareToCoords(moveStr.slice(0, 2));
  let aggCoordBegin = squareToCoords(moveStr.slice(2, 4));
  let [diffX, diffY] = dirmapping[moveStr.slice(4)];
  let pasCoordEnd = [pasCoordBegin[0] + diffX, pasCoordBegin[1] + diffY];
  // reassign passCoord
  boardState[8 - pasCoordEnd[1] - 1][pasCoordEnd[0]] = boardState[8 - pasCoordBegin[1] - 1][pasCoordBegin[0]];
  boardState[8 - pasCoordBegin[1] - 1][pasCoordBegin[0]] = "";

  // aggCoord need to figure out pushes
  let dir = moveStr.slice(4);
  let twice = false;
  if (dir.endsWith('2')) {
    dir = dir.slice(0, -1);
    twice = true;
  }
  [diffX, diffY] = dirmapping[dir];

  let aggCordEnd = [aggCoordBegin[0] + diffX, aggCoordBegin[1] + diffY];
  let pushedPiece = boardState[8 - aggCordEnd[1] - 1][aggCordEnd[0]];
  boardState[8 - aggCordEnd[1] - 1][aggCordEnd[0]] = "";
  if (twice) {
    aggCordEnd = [aggCordEnd[0] + diffX, aggCordEnd[1] + diffY];
    pushedPiece = (pushedPiece || boardState[8 - aggCordEnd[1] - 1][aggCordEnd[0]]);
    boardState[8 - aggCordEnd[1] - 1][aggCordEnd[0]] = "";
  }
  boardState[8 - aggCordEnd[1] - 1][aggCordEnd[0]] = boardState[8 - aggCoordBegin[1] - 1][aggCoordBegin[0]];
  boardState[8 - aggCoordBegin[1] - 1][aggCoordBegin[0]] = "";
  let pushedPieceLoc = [aggCordEnd[0] + diffX, aggCordEnd[1] + diffY];
  let skip = false;
  if (aggCordEnd[0] === 0 && pushedPieceLoc[0] < 0) skip = true;
  else if (aggCordEnd[0] === 3 && pushedPieceLoc[0] > 3) skip = true;
  else if (aggCordEnd[0] === 4 && pushedPieceLoc[0] < 4) skip = true;
  else if (aggCordEnd[0] === 7 && pushedPieceLoc[0] > 7) skip = true;
  else if (aggCordEnd[1] === 0 && pushedPieceLoc[1] < 0) skip = true;
  else if (aggCordEnd[1] === 3 && pushedPieceLoc[1] > 3) skip = true;
  else if (aggCordEnd[1] === 4 && pushedPieceLoc[1] < 4) skip = true;
  else if (aggCordEnd[1] === 7 && pushedPieceLoc[1] > 7) skip = true;

  if (!skip) {
    boardState[8 - pushedPieceLoc[1] - 1][pushedPieceLoc[0]] = boardState[8 - pushedPieceLoc[1] - 1][pushedPieceLoc[0]] || pushedPiece;
  }

  let changedSquares = [coordsToSquare(pasCoordBegin), coordsToSquare(aggCoordBegin), coordsToSquare(pasCoordEnd), coordsToSquare(aggCordEnd)];
  return [boardState, changedSquares];
}

// Thanks StackOverflow!
function useBoardBoundingRect() {
  const [boardBoundingRect, setBoardBoundingRect] = useState({
    height: 0,
    width: 0,
    x: 0,
    y: 0
  });
  useEffect(() => {
    function handleResize() {
      setBoardBoundingRect(document.querySelector("#board-container")!.getBoundingClientRect());
    }
    window.addEventListener("resize", handleResize);
    handleResize();
    // document.addEventListener('touchmove', function(e) { e.preventDefault(); }, { passive:false });
    return () => window.removeEventListener("resize", handleResize);
  }, []);
  return boardBoundingRect;
}

function applyDir(dir: any, coord: any) {
  let [x, y] = dir;
  let xc = String.fromCodePoint(coord.codePointAt(0) + x);
  let yc = String.fromCodePoint(coord.codePointAt(1) + y);
  return xc + yc;
}

function coordsToSquare(coords: any) {
  let xc = String.fromCodePoint(coords[0] + 'a'.codePointAt(0));
  let yc = String.fromCodePoint(coords[1] + '1'.codePointAt(0));
  return xc + yc;
}

function squareToCoords(sq: any) {
  let xc = sq.codePointAt(0) - 'a'.codePointAt(0)!;
  let yc = sq.codePointAt(1) - '1'.codePointAt(0)!;
  return [xc, yc];
}

function boardToString(board: any) {
  let res_str = "";
  for (let i = 0; i < 8; i++) {
    for (let j = 0; j < 8; j++) {
      if (board[i][j].startsWith('b')) res_str += "b";
      else if (board[i][j].startsWith('w')) res_str += "w";
      else if (!board[i][j]) res_str += ".";
    }
  }
  return res_str;
}


export default App;
