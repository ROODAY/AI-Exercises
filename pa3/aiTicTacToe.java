import java.util.*;
import java.util.stream.Collectors;

class aiTicTacToe {
	private int considered = 0;
	private int player; // 1 for player 1 and 2 for player 2
	private int MAX_DEPTH; // Controls how far the AI will lookahead (depth starts at 0)
	private List<List<positionTicTacToe>> winningLines;

	// Helper function to get state of a certain position in the Tic-Tac-Toe board by given position TicTacToe
	private int getStateOfPositionFromBoard(positionTicTacToe position, List<positionTicTacToe> board) {
		int index = position.x*16+position.y*4+position.z;
		return board.get(index).state;
	}

	// Custom heuristic function to score a given game board
	private int heuristic(List<positionTicTacToe> board) {
		int other = player == 1 ? 2 : 1;
		int boardScore = 0;
		for (List<positionTicTacToe> winningLine : winningLines) {
			// Map positions into states
			List<Integer> states = winningLine.stream().map(pos -> getStateOfPositionFromBoard(pos, board)).collect(Collectors.toList());

			// If the current position contains opponent states, play defensively (block opponent), otherwise offensively (get 4 in a row)
			boolean offensive = true;
			if (states.contains(other)) {
				offensive = false;
			}

			// Values assigned based on how many positions are taken in the line
			// Getting 4 in a row (winning) is more valued than blocking the opponent
			if (offensive) {
				int playerSpots = (int) states.stream().filter(state -> state == player).count();
				switch (playerSpots) {
					case 4: // If we win this board, we want it the most
						boardScore += Integer.MAX_VALUE;
						break;
					case 3:
						boardScore += 1000000;
						break;
					case 2:
						boardScore += 10000;
						break;
					case 1:
						boardScore += 100;
						break;
				}
			} else {
				int otherSpots = (int) states.stream().filter(state -> state == other).count();
				switch (otherSpots) {
					case 4: // If the opponent wins this board, we want it the least
						boardScore += Integer.MIN_VALUE;
						break;
					case 3:
						boardScore += 100000;
						break;
					case 2:
						boardScore += 1000;
						break;
					case 1:
						boardScore += 10;
						break;
				}
			}
		}

		return boardScore;
	}

	// Implements minimax algorithm (pruning based on move ordering rather than alpha-beta)
	private int minimax(List<positionTicTacToe> board, int curPlayer, int depth) {
		considered++;
		// If at max depth, return value of the board
		if (depth >= MAX_DEPTH) {
			return heuristic(board);
		}

		// Get available moves and score each one
		List<positionTicTacToe> availableMoves = getAvailableMoves(board);
		List<Integer> scores = availableMoves.stream().map(move -> {
			List<positionTicTacToe> b = deepCopyATicTacToeBoard(board);
			makeMove(move, curPlayer, b);
			return heuristic(b);
		}).collect(Collectors.toList());

		// Perform best move, depending on maximizer or minimizer (maximizer is 'player')
		int bestMoveIndex = curPlayer == player ? scores.indexOf(Collections.max(scores)) : scores.indexOf(Collections.min(scores));
		List<positionTicTacToe> bestPath = deepCopyATicTacToeBoard(board);
		makeMove(availableMoves.get(bestMoveIndex), curPlayer, bestPath);

		// Recurse down the appropriate path, depending on maximizer or minimizer (maximizer is 'player')
		int other = player == 1 ? 2 : 1;
		if (curPlayer == player) {
			if (isEnded(bestPath)) return 1000000; // Greatly value boards where maximizer wins
			return Math.max(Collections.max(scores), minimax(bestPath, other, depth+1));
		} else {
			if (isEnded(bestPath)) return Integer.MIN_VALUE; // Avoid boards where minimizer wins
			return Math.min(Collections.min(scores), minimax(bestPath, other, depth+1));
		}
	}

	// Base function that calls minimax algorithm
	positionTicTacToe myAIAlgorithm(List<positionTicTacToe> board, int player) {
		considered = 0;
		positionTicTacToe myNextMove = null;
		int other = player == 1 ? 2 : 1;
		int bestScore = Integer.MIN_VALUE;

		/*
		 * Evaluate each possible move.
		 * If move ends the game, play it.
		 * Otherwise, score move based on minimax.
		 * If score is new best, update bestScore and myNextMove.
		 */
		for (positionTicTacToe move : getAvailableMoves(board)) {
			List<positionTicTacToe> b = deepCopyATicTacToeBoard(board);
			makeMove(move, player, b);
			if (isEnded(b)) {
				myNextMove = move;
				break;
			}
			int score = minimax(b, other, 1);
			if (score >= bestScore) {
				bestScore = score;
				myNextMove = move;
			}
		}

		System.out.println("Player " + player + " considered " + considered + " moves.");
		return myNextMove;
	}

	// Helper function to initialize winning lines (used for scoring potential game boards)
	private List<List<positionTicTacToe>> initializeWinningLines() {
		//create a list of winning line so that the game will "brute-force" check if a player satisfied any 	winning condition(s).
		List<List<positionTicTacToe>> winningLines = new ArrayList<>();
		
		//48 straight winning lines
		//z axis winning lines
		for(int i = 0; i<4; i++)
			for(int j = 0; j<4;j++) {
				List<positionTicTacToe> oneWinCondition = new ArrayList<>();
				oneWinCondition.add(new positionTicTacToe(i,j,0,-1));
				oneWinCondition.add(new positionTicTacToe(i,j,1,-1));
				oneWinCondition.add(new positionTicTacToe(i,j,2,-1));
				oneWinCondition.add(new positionTicTacToe(i,j,3,-1));
				winningLines.add(oneWinCondition);
			}
		//y axis winning lines
		for(int i = 0; i<4; i++)
			for(int j = 0; j<4;j++) {
				List<positionTicTacToe> oneWinCondition = new ArrayList<>();
				oneWinCondition.add(new positionTicTacToe(i,0,j,-1));
				oneWinCondition.add(new positionTicTacToe(i,1,j,-1));
				oneWinCondition.add(new positionTicTacToe(i,2,j,-1));
				oneWinCondition.add(new positionTicTacToe(i,3,j,-1));
				winningLines.add(oneWinCondition);
			}
		//x axis winning lines
		for(int i = 0; i<4; i++)
			for(int j = 0; j<4;j++) {
				List<positionTicTacToe> oneWinCondition = new ArrayList<>();
				oneWinCondition.add(new positionTicTacToe(0,i,j,-1));
				oneWinCondition.add(new positionTicTacToe(1,i,j,-1));
				oneWinCondition.add(new positionTicTacToe(2,i,j,-1));
				oneWinCondition.add(new positionTicTacToe(3,i,j,-1));
				winningLines.add(oneWinCondition);
			}
		
		//12 main diagonal winning lines
		//xz plane-4
		for (int i = 0; i<4; i++) {
			List<positionTicTacToe> oneWinCondition = new ArrayList<>();
			oneWinCondition.add(new positionTicTacToe(0,i,0,-1));
			oneWinCondition.add(new positionTicTacToe(1,i,1,-1));
			oneWinCondition.add(new positionTicTacToe(2,i,2,-1));
			oneWinCondition.add(new positionTicTacToe(3,i,3,-1));
			winningLines.add(oneWinCondition);
		}
		//yz plane-4
		for (int i = 0; i<4; i++) {
			List<positionTicTacToe> oneWinCondition = new ArrayList<>();
			oneWinCondition.add(new positionTicTacToe(i,0,0,-1));
			oneWinCondition.add(new positionTicTacToe(i,1,1,-1));
			oneWinCondition.add(new positionTicTacToe(i,2,2,-1));
			oneWinCondition.add(new positionTicTacToe(i,3,3,-1));
			winningLines.add(oneWinCondition);
		}
		//xy plane-4
		for (int i = 0; i<4; i++) {
			List<positionTicTacToe> oneWinCondition = new ArrayList<>();
			oneWinCondition.add(new positionTicTacToe(0,0,i,-1));
			oneWinCondition.add(new positionTicTacToe(1,1,i,-1));
			oneWinCondition.add(new positionTicTacToe(2,2,i,-1));
			oneWinCondition.add(new positionTicTacToe(3,3,i,-1));
			winningLines.add(oneWinCondition);
		}
		
		//12 anti diagonal winning lines
		//xz plane-4
		for (int i = 0; i<4; i++) {
			List<positionTicTacToe> oneWinCondition = new ArrayList<>();
			oneWinCondition.add(new positionTicTacToe(0,i,3,-1));
			oneWinCondition.add(new positionTicTacToe(1,i,2,-1));
			oneWinCondition.add(new positionTicTacToe(2,i,1,-1));
			oneWinCondition.add(new positionTicTacToe(3,i,0,-1));
			winningLines.add(oneWinCondition);
		}
		//yz plane-4
		for (int i = 0; i<4; i++) {
			List<positionTicTacToe> oneWinCondition = new ArrayList<>();
			oneWinCondition.add(new positionTicTacToe(i,0,3,-1));
			oneWinCondition.add(new positionTicTacToe(i,1,2,-1));
			oneWinCondition.add(new positionTicTacToe(i,2,1,-1));
			oneWinCondition.add(new positionTicTacToe(i,3,0,-1));
			winningLines.add(oneWinCondition);
		}
		//xy plane-4
		for (int i = 0; i<4; i++) {
			List<positionTicTacToe> oneWinCondition = new ArrayList<>();
			oneWinCondition.add(new positionTicTacToe(0,3,i,-1));
			oneWinCondition.add(new positionTicTacToe(1,2,i,-1));
			oneWinCondition.add(new positionTicTacToe(2,1,i,-1));
			oneWinCondition.add(new positionTicTacToe(3,0,i,-1));
			winningLines.add(oneWinCondition);
		}
		
		//4 additional diagonal winning lines
		List<positionTicTacToe> oneWinCondition = new ArrayList<>();
		oneWinCondition.add(new positionTicTacToe(0,0,0,-1));
		oneWinCondition.add(new positionTicTacToe(1,1,1,-1));
		oneWinCondition.add(new positionTicTacToe(2,2,2,-1));
		oneWinCondition.add(new positionTicTacToe(3,3,3,-1));
		winningLines.add(oneWinCondition);
		
		oneWinCondition = new ArrayList<>();
		oneWinCondition.add(new positionTicTacToe(0,0,3,-1));
		oneWinCondition.add(new positionTicTacToe(1,1,2,-1));
		oneWinCondition.add(new positionTicTacToe(2,2,1,-1));
		oneWinCondition.add(new positionTicTacToe(3,3,0,-1));
		winningLines.add(oneWinCondition);
		
		oneWinCondition = new ArrayList<>();
		oneWinCondition.add(new positionTicTacToe(3,0,0,-1));
		oneWinCondition.add(new positionTicTacToe(2,1,1,-1));
		oneWinCondition.add(new positionTicTacToe(1,2,2,-1));
		oneWinCondition.add(new positionTicTacToe(0,3,3,-1));
		winningLines.add(oneWinCondition);
		
		oneWinCondition = new ArrayList<>();
		oneWinCondition.add(new positionTicTacToe(0,3,0,-1));
		oneWinCondition.add(new positionTicTacToe(1,2,1,-1));
		oneWinCondition.add(new positionTicTacToe(2,1,2,-1));
		oneWinCondition.add(new positionTicTacToe(3,0,3,-1));
		winningLines.add(oneWinCondition);	
		
		return winningLines;
	}

	// Helper function to inform when game is complete
	private boolean isEnded(List<positionTicTacToe> board) {
		boolean gameover = false;
		for (List<positionTicTacToe> winningLine : winningLines) {

			positionTicTacToe p0 = winningLine.get(0);
			positionTicTacToe p1 = winningLine.get(1);
			positionTicTacToe p2 = winningLine.get(2);
			positionTicTacToe p3 = winningLine.get(3);

			int state0 = getStateOfPositionFromBoard(p0, board);
			int state1 = getStateOfPositionFromBoard(p1, board);
			int state2 = getStateOfPositionFromBoard(p2, board);
			int state3 = getStateOfPositionFromBoard(p3, board);

			//if they have the same state (marked by same player) and they are not all marked.
			if (state0 == state1 && state1 == state2 && state2 == state3 && state0 != 0) {
				gameover = true;
				break;
			}
		}
		return gameover;
	}

	// Helper function to assist with evaluating potential moves
	private void makeMove(positionTicTacToe position, int player, List<positionTicTacToe> targetBoard) {
		for (positionTicTacToe positionTicTacToe : targetBoard) {
			if (positionTicTacToe.x == position.x && positionTicTacToe.y == position.y && positionTicTacToe.z == position.z) {
				if (positionTicTacToe.state == 0) {
					positionTicTacToe.state = player;
					return;
				} else {
					System.out.println("Error: this is not a valid move.");
				}
			}
		}
	}

	// Helper function to assist with evaluating potential moves
	private List<positionTicTacToe> deepCopyATicTacToeBoard(List<positionTicTacToe> board) {
		//deep copy of game boards
		List<positionTicTacToe> copiedBoard = new ArrayList<positionTicTacToe>();
		for (positionTicTacToe positionTicTacToe : board) {
			copiedBoard.add(new positionTicTacToe(positionTicTacToe.x, positionTicTacToe.y, positionTicTacToe.z, positionTicTacToe.state));
		}
		return copiedBoard;
	}

	// Filters board to return only empty positions
	private List<positionTicTacToe> getAvailableMoves(List<positionTicTacToe> board) {
		return board.stream().filter(pos -> pos.state == 0).collect(Collectors.toList());
	}

	// Initialize agent with depth
	aiTicTacToe(int setPlayer, int setDepth) {
		player = setPlayer;
		winningLines = initializeWinningLines();
		MAX_DEPTH = setDepth;
	}

	// Initialize agent with default depth
	aiTicTacToe(int setPlayer) {
		player = setPlayer;
		winningLines = initializeWinningLines();
		MAX_DEPTH = 3;
	}
}
