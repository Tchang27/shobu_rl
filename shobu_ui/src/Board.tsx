import React from 'react';
import './App.css';

interface BoardProps { width: number, flipped: boolean }
function Board({ width, flipped }: BoardProps) {
	let color_a = flipped ? "light" : "dark";
	let color_b = flipped ? "dark" : "light";
	let coords = flipped ? [
		['1', '2', '3', '4'],
		['5', '6', '7', '8', 'h', 'g', 'f', 'e'],
		['', '', '', '', 'd', 'c', 'b', 'a']
	] : [
		['8', '7', '6', '5'],
		['4', '3', '2','1', 'a', 'b', 'c', 'd'],
		['', '', '', '', 'e', 'f', 'g', 'h']
	];
	return (
		<div className="board">
			<table style={{ borderSpacing: width / 40 + 'px' }}>
				<tbody>
					<tr>
						<td><SubBoard color={color_a} width={width * 37 / 40 - 4} cornerText={coords[0]}/></td>
						<td><SubBoard color={color_b} width={width * 37 / 40 - 4} cornerText={[]}/></td>
					</tr>
					<tr>
						<td><SubBoard color={color_a} width={width * 37 / 40 - 4}  cornerText={coords[1]}/></td>
						<td><SubBoard color={color_b} width={width * 37 / 40 - 4}  cornerText={coords[2]}/></td>
					</tr>
				</tbody>
			</table>
		</div>
	);
}


interface SubBoardProps { color: string, width: number, cornerText: string[] }
function SubBoard({ color, width, cornerText }: SubBoardProps) {
	return (
		<div className={color === "dark" ? "subboard dark" : "subboard light"} style={{
			height: width / 2 + 'px',
			width: width / 2 + 'px',
			borderRadius: width / 120 + 'px',
			position: "relative",
			fontSize: width / 40 + 'px'
		}}>
			<table style={{ width: "100%", height: "100%", borderSpacing: width / 150 + 'px' }}>
				<tbody>
					<tr>
						<td>
						<table style={{ width: "100%", height: "100%", padding: "3%" }}>
								<tbody>
									<tr>
										<td style={{ lineHeight: 1, textAlign: "left", verticalAlign: "top" }}>
											{cornerText[0]}
										</td>
									</tr>
									<tr>
										<td></td>
									</tr>
								</tbody>
							</table>
						</td>
						<td></td>
						<td></td>
						<td></td>
					</tr>
					<tr>
						<td>
						<table style={{ width: "100%", height: "100%", padding: "3%" }}>
								<tbody>
									<tr>
										<td style={{ lineHeight: 1, textAlign: "left", verticalAlign: "top" }}>
											{cornerText[1]}
										</td>
									</tr>
									<tr>
										<td></td>
									</tr>
								</tbody>
							</table>
						</td>
						<td></td>
						<td></td>
						<td></td>
					</tr>
					<tr>
						<td>
						<table style={{ width: "100%", height: "100%", padding: "3%" }}>
								<tbody>
									<tr>
										<td style={{ lineHeight: 1, textAlign: "left", verticalAlign: "top" }}>
											{cornerText[2]}
										</td>
									</tr>
									<tr>
										<td></td>
									</tr>
								</tbody>
							</table>
						</td>
						<td></td>
						<td></td>
						<td></td>
					</tr>
					<tr>
						<td>
						<table style={{ width: "100%", height: "100%", padding: "3%" }}>
								<tbody>
									<tr>
										<td style={{ lineHeight: 1, textAlign: "left", verticalAlign: "top" }}>
											{cornerText[3]}
										</td>
									</tr>
									<tr>
									<td style={{ lineHeight: 1, textAlign: "right", verticalAlign: "bottom" }}>
											{cornerText[4]}
										</td>
									</tr>
								</tbody>
							</table>
						</td>
						<td><table style={{ width: "100%", height: "100%", padding: "3%" }}>
								<tbody>
									<tr>
										<td></td>
									</tr>
									<tr>
									<td style={{ lineHeight: 1, textAlign: "right", verticalAlign: "bottom" }}>
											{cornerText[5]}
										</td>
									</tr>
								</tbody>
							</table>
							</td>
						<td><table style={{ width: "100%", height: "100%", padding: "3%" }}>
								<tbody>
									<tr>
										<td></td>
									</tr>
									<tr>
									<td style={{ lineHeight: 1, textAlign: "right", verticalAlign: "bottom" }}>
											{cornerText[6]}
										</td>
									</tr>
								</tbody>
							</table></td>
						<td><table style={{ width: "100%", height: "100%", padding: "3%" }}>
								<tbody>
									<tr>
										<td></td>
									</tr>
									<tr>
									<td style={{ lineHeight: 1, textAlign: "right", verticalAlign: "bottom" }}>
											{cornerText[7]}
										</td>
									</tr>
								</tbody>
							</table></td>
					</tr>
				</tbody>
			</table>
		</div>
	);
}

export default Board;
