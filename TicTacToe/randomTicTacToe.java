import java.util.*;
class randomTicTacToe {

	private int player; //1 for player 1 and 2 for player 2
	private int getStateOfPositionFromBoard(positionTicTacToe position, List<positionTicTacToe> board)
	{
		//a helper function to get state of a certain position in the Tic-Tac-Toe board by given position TicTacToe
		int index = position.x*16+position.y*4+position.z;
		return board.get(index).state;
	}
	positionTicTacToe myAIAlgorithm(List<positionTicTacToe> board, int player)
	{

		//TODO: this is where you are going to implement your AI algorithm to win the game. The default is an AI randomly choose any available move.
		positionTicTacToe myNextMove;
		
		do
			{
				Random rand = new Random();
				int x = rand.nextInt(4);
				int y = rand.nextInt(4);
				int z = rand.nextInt(4);
				myNextMove = new positionTicTacToe(x,y,z);
			} while(getStateOfPositionFromBoard(myNextMove,board)!=0);
		return myNextMove;
			
		
	}
	private List<List<positionTicTacToe>> initializeWinningLines()
	{
		//create a list of winning line so that the game will "brute-force" check if a player satisfied any 	winning condition(s).
		List<List<positionTicTacToe>> winningLines = new ArrayList<List<positionTicTacToe>>();
		
		//48 straight winning lines
		//z axis winning lines
		for(int i = 0; i<4; i++)
			for(int j = 0; j<4;j++)
			{
				List<positionTicTacToe> oneWinCondition = new ArrayList<positionTicTacToe>();
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
				List<positionTicTacToe> oneWinCondition = new ArrayList<positionTicTacToe>();
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
				List<positionTicTacToe> oneWinCondition = new ArrayList<positionTicTacToe>();
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
				List<positionTicTacToe> oneWinCondition = new ArrayList<positionTicTacToe>();
				oneWinCondition.add(new positionTicTacToe(0,i,0,-1));
				oneWinCondition.add(new positionTicTacToe(1,i,1,-1));
				oneWinCondition.add(new positionTicTacToe(2,i,2,-1));
				oneWinCondition.add(new positionTicTacToe(3,i,3,-1));
				winningLines.add(oneWinCondition);
			}
		//yz plane-4
		for(int i = 0; i<4; i++)
			{
				List<positionTicTacToe> oneWinCondition = new ArrayList<positionTicTacToe>();
				oneWinCondition.add(new positionTicTacToe(i,0,0,-1));
				oneWinCondition.add(new positionTicTacToe(i,1,1,-1));
				oneWinCondition.add(new positionTicTacToe(i,2,2,-1));
				oneWinCondition.add(new positionTicTacToe(i,3,3,-1));
				winningLines.add(oneWinCondition);
			}
		//xy plane-4
		for(int i = 0; i<4; i++)
			{
				List<positionTicTacToe> oneWinCondition = new ArrayList<positionTicTacToe>();
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
				List<positionTicTacToe> oneWinCondition = new ArrayList<positionTicTacToe>();
				oneWinCondition.add(new positionTicTacToe(0,i,3,-1));
				oneWinCondition.add(new positionTicTacToe(1,i,2,-1));
				oneWinCondition.add(new positionTicTacToe(2,i,1,-1));
				oneWinCondition.add(new positionTicTacToe(3,i,0,-1));
				winningLines.add(oneWinCondition);
			}
		//yz plane-4
		for(int i = 0; i<4; i++)
			{
				List<positionTicTacToe> oneWinCondition = new ArrayList<positionTicTacToe>();
				oneWinCondition.add(new positionTicTacToe(i,0,3,-1));
				oneWinCondition.add(new positionTicTacToe(i,1,2,-1));
				oneWinCondition.add(new positionTicTacToe(i,2,1,-1));
				oneWinCondition.add(new positionTicTacToe(i,3,0,-1));
				winningLines.add(oneWinCondition);
			}
		//xy plane-4
		for(int i = 0; i<4; i++)
			{
				List<positionTicTacToe> oneWinCondition = new ArrayList<positionTicTacToe>();
				oneWinCondition.add(new positionTicTacToe(0,3,i,-1));
				oneWinCondition.add(new positionTicTacToe(1,2,i,-1));
				oneWinCondition.add(new positionTicTacToe(2,1,i,-1));
				oneWinCondition.add(new positionTicTacToe(3,0,i,-1));
				winningLines.add(oneWinCondition);
			}
		
		//4 additional diagonal winning lines
		List<positionTicTacToe> oneWinCondition = new ArrayList<positionTicTacToe>();
		oneWinCondition.add(new positionTicTacToe(0,0,0,-1));
		oneWinCondition.add(new positionTicTacToe(1,1,1,-1));
		oneWinCondition.add(new positionTicTacToe(2,2,2,-1));
		oneWinCondition.add(new positionTicTacToe(3,3,3,-1));
		winningLines.add(oneWinCondition);
		
		oneWinCondition = new ArrayList<positionTicTacToe>();
		oneWinCondition.add(new positionTicTacToe(0,0,3,-1));
		oneWinCondition.add(new positionTicTacToe(1,1,2,-1));
		oneWinCondition.add(new positionTicTacToe(2,2,1,-1));
		oneWinCondition.add(new positionTicTacToe(3,3,0,-1));
		winningLines.add(oneWinCondition);
		
		oneWinCondition = new ArrayList<positionTicTacToe>();
		oneWinCondition.add(new positionTicTacToe(3,0,0,-1));
		oneWinCondition.add(new positionTicTacToe(2,1,1,-1));
		oneWinCondition.add(new positionTicTacToe(1,2,2,-1));
		oneWinCondition.add(new positionTicTacToe(0,3,3,-1));
		winningLines.add(oneWinCondition);
		
		oneWinCondition = new ArrayList<positionTicTacToe>();
		oneWinCondition.add(new positionTicTacToe(0,3,0,-1));
		oneWinCondition.add(new positionTicTacToe(1,2,1,-1));
		oneWinCondition.add(new positionTicTacToe(2,1,2,-1));
		oneWinCondition.add(new positionTicTacToe(3,0,3,-1));
		winningLines.add(oneWinCondition);	
		
		return winningLines;
		
	}
	randomTicTacToe(int setPlayer)
	{
		player = setPlayer;
	}
}
