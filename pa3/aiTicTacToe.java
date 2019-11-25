import java.util.*;
import java.util.stream.Collectors;

class aiTicTacToe {

	private int player; //1 for player 1 and 2 for player 2
	private List<List<positionTicTacToe>> winningLines;
	private int getStateOfPositionFromBoard(positionTicTacToe position, List<positionTicTacToe> board)
	{
		//a helper function to get state of a certain position in the Tic-Tac-Toe board by given position TicTacToe
		int index = position.x*16+position.y*4+position.z;
		return board.get(index).state;
	}
	private int heuristic(List<positionTicTacToe> board, int player) {
		int other = player == 1 ? 2 : 1;
		int playerScore = 0;
		int otherScore = 0;
		for (List<positionTicTacToe> winningLine : winningLines) {
			List<Integer> states = new ArrayList<>();
			for (positionTicTacToe pos : winningLine) {
				states.add(getStateOfPositionFromBoard(pos,board));
			}

			int playerSpots = (int) states.stream().filter(state -> state == player).count();
			switch (playerSpots) {
				case 4:
					playerScore += 1000;
					break;
				case 3:
					playerScore += 100;
					break;
				case 2:
					playerScore += 10;
					break;
				case 1:
					playerScore += 1;
					break;
			}

			int otherSpots = (int) states.stream().filter(state -> state == other).count();
			switch (otherSpots) {
				case 4:
					otherScore += 1000;
					break;
				case 3:
					otherScore += 100;
					break;
				case 2:
					otherScore += 10;
					break;
				case 1:
					otherScore += 1;
					break;
			}
		}

		return playerScore - otherScore;
	}
	positionTicTacToe myAIAlgorithm(List<positionTicTacToe> board, int player)
	{
		positionTicTacToe myNextMove = null;
		
		int bestMove = Integer.MIN_VALUE;
		List<positionTicTacToe> availableMoves = getAvailableMoves(board);
		for (positionTicTacToe move : availableMoves) {
			List<positionTicTacToe> b = deepCopyATicTacToeBoard(board);
			makeMove(move, player, b);
			int score = heuristic(b, player);
			if (score > bestMove) {
				bestMove = score;
				myNextMove = move;
			}
		}
		return myNextMove;
	}
	private List<List<positionTicTacToe>> initializeWinningLines()
	{
		//create a list of winning line so that the game will "brute-force" check if a player satisfied any 	winning condition(s).
		List<List<positionTicTacToe>> winningLines = new ArrayList<>();
		
		//48 straight winning lines
		//z axis winning lines
		for(int i = 0; i<4; i++)
			for(int j = 0; j<4;j++)
			{
				List<positionTicTacToe> oneWinCondition = new ArrayList<>();
				oneWinCondition.add(new positionTicTacToe(i,j,0,-1));
				oneWinCondition.add(new positionTicTacToe(i,j,1,-1));
				oneWinCondition.add(new positionTicTacToe(i,j,2,-1));
				oneWinCondition.add(new positionTicTacToe(i,j,3,-1));
				winningLines.add(oneWinCondition);
			}
		//y axis winning lines
		for(int i = 0; i<4; i++)
			for(int j = 0; j<4;j++)
			{
				List<positionTicTacToe> oneWinCondition = new ArrayList<>();
				oneWinCondition.add(new positionTicTacToe(i,0,j,-1));
				oneWinCondition.add(new positionTicTacToe(i,1,j,-1));
				oneWinCondition.add(new positionTicTacToe(i,2,j,-1));
				oneWinCondition.add(new positionTicTacToe(i,3,j,-1));
				winningLines.add(oneWinCondition);
			}
		//x axis winning lines
		for(int i = 0; i<4; i++)
			for(int j = 0; j<4;j++)
			{
				List<positionTicTacToe> oneWinCondition = new ArrayList<>();
				oneWinCondition.add(new positionTicTacToe(0,i,j,-1));
				oneWinCondition.add(new positionTicTacToe(1,i,j,-1));
				oneWinCondition.add(new positionTicTacToe(2,i,j,-1));
				oneWinCondition.add(new positionTicTacToe(3,i,j,-1));
				winningLines.add(oneWinCondition);
			}
		
		//12 main diagonal winning lines
		//xz plane-4
		for(int i = 0; i<4; i++)
			{
				List<positionTicTacToe> oneWinCondition = new ArrayList<>();
				oneWinCondition.add(new positionTicTacToe(0,i,0,-1));
				oneWinCondition.add(new positionTicTacToe(1,i,1,-1));
				oneWinCondition.add(new positionTicTacToe(2,i,2,-1));
				oneWinCondition.add(new positionTicTacToe(3,i,3,-1));
				winningLines.add(oneWinCondition);
			}
		//yz plane-4
		for(int i = 0; i<4; i++)
			{
				List<positionTicTacToe> oneWinCondition = new ArrayList<>();
				oneWinCondition.add(new positionTicTacToe(i,0,0,-1));
				oneWinCondition.add(new positionTicTacToe(i,1,1,-1));
				oneWinCondition.add(new positionTicTacToe(i,2,2,-1));
				oneWinCondition.add(new positionTicTacToe(i,3,3,-1));
				winningLines.add(oneWinCondition);
			}
		//xy plane-4
		for(int i = 0; i<4; i++)
			{
				List<positionTicTacToe> oneWinCondition = new ArrayList<>();
				oneWinCondition.add(new positionTicTacToe(0,0,i,-1));
				oneWinCondition.add(new positionTicTacToe(1,1,i,-1));
				oneWinCondition.add(new positionTicTacToe(2,2,i,-1));
				oneWinCondition.add(new positionTicTacToe(3,3,i,-1));
				winningLines.add(oneWinCondition);
			}
		
		//12 anti diagonal winning lines
		//xz plane-4
		for(int i = 0; i<4; i++)
			{
				List<positionTicTacToe> oneWinCondition = new ArrayList<>();
				oneWinCondition.add(new positionTicTacToe(0,i,3,-1));
				oneWinCondition.add(new positionTicTacToe(1,i,2,-1));
				oneWinCondition.add(new positionTicTacToe(2,i,1,-1));
				oneWinCondition.add(new positionTicTacToe(3,i,0,-1));
				winningLines.add(oneWinCondition);
			}
		//yz plane-4
		for(int i = 0; i<4; i++)
			{
				List<positionTicTacToe> oneWinCondition = new ArrayList<>();
				oneWinCondition.add(new positionTicTacToe(i,0,3,-1));
				oneWinCondition.add(new positionTicTacToe(i,1,2,-1));
				oneWinCondition.add(new positionTicTacToe(i,2,1,-1));
				oneWinCondition.add(new positionTicTacToe(i,3,0,-1));
				winningLines.add(oneWinCondition);
			}
		//xy plane-4
		for(int i = 0; i<4; i++)
			{
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
	private int isEnded(List<List<positionTicTacToe>> winningLines, List<positionTicTacToe> board)
	{
		//test whether the current game is ended

		//brute-force
		for(int i=0;i<winningLines.size();i++)
		{

			positionTicTacToe p0 = winningLines.get(i).get(0);
			positionTicTacToe p1 = winningLines.get(i).get(1);
			positionTicTacToe p2 = winningLines.get(i).get(2);
			positionTicTacToe p3 = winningLines.get(i).get(3);

			int state0 = getStateOfPositionFromBoard(p0,board);
			int state1 = getStateOfPositionFromBoard(p1,board);
			int state2 = getStateOfPositionFromBoard(p2,board);
			int state3 = getStateOfPositionFromBoard(p3,board);

			//if they have the same state (marked by same player) and they are not all marked.
			if(state0 == state1 && state1 == state2 && state2 == state3 && state0!=0)
			{
				//someone wins
				p0.state = state0;
				p1.state = state1;
				p2.state = state2;
				p3.state = state3;

				//print the satisified winning line (one of them if there are several)
				p0.printPosition();
				p1.printPosition();
				p2.printPosition();
				p3.printPosition();
				return state0;
			}
		}
		for(int i=0;i<board.size();i++)
		{
			if(board.get(i).state==0)
			{
				//game is not ended, continue
				return 0;
			}
		}
		return -1; //call it a draw
	}
	private void makeMove(positionTicTacToe position, int player, List<positionTicTacToe> targetBoard)
	{
		//make move on Tic-Tac-Toe board, given position and player
		//player 1 = 1, player 2 = 2

		//brute force (obviously not a wise way though)
		for(int i=0;i<targetBoard.size();i++)
		{
			if(targetBoard.get(i).x==position.x && targetBoard.get(i).y==position.y && targetBoard.get(i).z==position.z) //if this is the position
			{
				if(targetBoard.get(i).state==0)
				{
					targetBoard.get(i).state = player;
					return;
				}
				else
				{
					System.out.println("Error: this is not a valid move.");
				}
			}

		}
	}
	private List<positionTicTacToe> deepCopyATicTacToeBoard(List<positionTicTacToe> board)
	{
		//deep copy of game boards
		List<positionTicTacToe> copiedBoard = new ArrayList<positionTicTacToe>();
		for (positionTicTacToe positionTicTacToe : board) {
			copiedBoard.add(new positionTicTacToe(positionTicTacToe.x, positionTicTacToe.y, positionTicTacToe.z, positionTicTacToe.state));
		}
		return copiedBoard;
	}
	private List<positionTicTacToe> getAvailableMoves(List<positionTicTacToe> board) {
		return board.stream().filter(pos -> pos.state == 0).collect(Collectors.toList());
	}
	aiTicTacToe(int setPlayer)
	{
		player = setPlayer;
		winningLines = initializeWinningLines();
	}
}
