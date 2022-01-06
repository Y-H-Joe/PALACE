package tools;

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;

import structure.AromaticResonanceApplier;

public class MyMath {
	
	public static void main(String[] args){
		Double[][] a = {{0.0,1.0,8.0,4.0,9.0},
		{1.0,0.0,5.0,1.0,2.0},
		{8.0,5.0,0.0,3.0,1.0},
		{4.0,1.0,3.0,0.0,2.0},
		{9.0,2.0,1.0,2.0,0.0}};
		
		List<List<Double>> m	=	new ArrayList<List<Double>>();
		for(int i=0;i<5;i++){
			List<Double> mm = new ArrayList<Double>();
		
			for(int j=0;j<5;j++){
				
				mm.add(a[i][j]);
			}
			m.add(mm);
				
		}
		List<List<Integer>> m2	=	minimalSpanningTree(m);
		printI(m2);
		
	}
	
	public static void print(List<Double> a){
		for(double b:a)
			System.out.print(b+ " ");
		
		System.out.print("\n");
	}
	
	public static void printD(List<List<Double>> a){
		for(List<Double> l:a){
			for(double d:l)
				System.out.print(d+ " ");
			System.out.print("\n");
		}
	}
	public static void printI(List<List<Integer>> a){
		for(List<Integer> l:a){
			for(double d:l)
				System.out.print(d+ " ");
			System.out.print("\n");
		}
	}
	/**Determine the greatest common divisor of an array of numbers
	 * 
	 * @param numbers
	 * @return gcd
	 */
	public static int greatestCommonDivisor(int[] numbers){
		if(numbers.length > 2){
			int [] oneShorter	=	new int[numbers.length-1];
			for(int i = 0; i < oneShorter.length-1;	i++){
				oneShorter[i]	=	numbers[i];
			}
			oneShorter[oneShorter.length-1]	=	greatestCommonDivisor(numbers[numbers.length-1],numbers[numbers.length-2]);
			return greatestCommonDivisor(oneShorter);
		}
		else{
			return greatestCommonDivisor(numbers[0],numbers[1]);
		}
	}
	
	/**Determine the greatest common divisor of two numbers
	 * 
	 * @param a
	 * @param b
	 * @return gcd
	 */
	public static int greatestCommonDivisor(int a, int b){
		if(b == 0)	return a;
		
		return greatestCommonDivisor(b,a%b);
	}
	
	/**Determine the greatest common divisor of a list of numbers
	 * 
	 * @param numbers
	 * @return gcd
	 */
	public static int greatestCommonDivisor(List<Integer> numbers){
		
		List<Integer>	copy	=	new ArrayList<Integer>(numbers);
		//Specifically for the component coefficients list structure, with a -1 for break between reactants and products.
		if(copy.contains(-1)){
			copy.remove(copy.indexOf(-1));
		}
		
		if(copy.size() == 1){
			return copy.get(0);
		}
		
		if(copy.size() == 0){
			return 1;
		}
		
		if(copy.size() > 2){
			List<Integer> oneShorter	=	new ArrayList<Integer>();
			
			for(int i = 0; i < copy.size()-2;	i++){
				oneShorter.add(copy.get(i));
			}
			
			oneShorter.add(greatestCommonDivisor(copy.get(copy.size()-1),copy.get(copy.size()-2)));
			return greatestCommonDivisor(oneShorter);
		}
		else{
			return greatestCommonDivisor(copy.get(0),copy.get(1));
		}
	}
	
	/**Method to get the next set of indices for the next permutation of a set of sets.<br>
	 * For a set of sets of the following structure: R1 - {M11,M12,..,M1n}, R2 - {M21,M22,..,M2m}, ... Rx - {...}<br>
	 * an independent combination of x of the M components (one from each R) wants to be made.
	 * The currentPermutation array comprises x elements and each element at index i is an index in Ri (so
	 * it points to M[i][currentPermutation[i]. This allows all permutations to be constructed in createAllPermutations()
	 * </p>
	 * When using this method, "pos" should always be set to the last index of the current permutation array.
	 * </p>
	 * This method increases the indices to go to the next permutation as follows.<br> 
	 * The value of currentPermutation[pos] is increased by one. If the value becomes equal to the maximum specified
	 * value (ie the number of M elements in R), the value is reset to 0 at that pos and the value at pos-1 is
	 * increases using the same method.
	 * </p>
	 * Example:<br>
	 * R1={1,2}<br>
	 * R2={2}<br>
	 * R3={3,4}<br>
	 * initial currentPermutation is {0,0,0}. After the first call: {0,0,1}, after the second call: {1,0,0} 
	 * as by increasing the previous to {0,0,2}, 2=R1.length, so it is reset and becomes {0,1,0}, however, 1=R2.length, 
	 * so finally {1,0,0} is returned.
	 * 
	 * @param currentPermutation - is altered by method
	 * @param maxCount - size of each array of elements
	 * @param pos - should always be currentPermutation.length-1
	 * @return true if not all permutations have been encountered, false if output is {maxCount[0]-1,maxCount[1]-1,...}:
	 * indicates that indices for the final permutation have been generated.
	 */
	public static boolean nextPermutation(int[] currentPermutation, int[] maxCount, int pos){
		
		currentPermutation[pos]++;
		
		if(currentPermutation[pos] == maxCount[pos]){
			if(pos == 0){return false;}
			
			currentPermutation[pos]	=	0;
			return nextPermutation(currentPermutation, maxCount, pos-1);
		}
		
		return true;
	}
	
	/**Increase the bit at position pos by one and adapt others if necessary. 
	 * If the bit is larger than 2^bit.length - 1 (overfow) a row of -1 is returned.
	 * 
	 * @param bit
	 * @param pos
	 */
	public static void addOne(int[] bit, int pos){
		if(pos == 0 && bit[pos] == 1){
			Logger.getLogger(AromaticResonanceApplier.class).error("Out of bounds");
			for(int i=0;i<bit.length;i++){
				bit[i]=-1;
			}
		}
		else{
			if(bit[pos] == 0)
				bit[pos]++;
			else if(bit[pos] == 1){
				bit[pos]--;
				addOne(bit,pos - 1);
			}
		}
	}
	
	/**Determine the minimal spanning tree for a matrix represented weighted graph.
	 * Returns the index pairs of the nodes of the edges that have been added. The weights should all be positive.
	 * @param matrix
	 * @return
	 */
	public static List<List<Integer>> minimalSpanningTree(List<List<Double>> matrix){
		
		List<Double> distances			=	new ArrayList<Double>();
		List<Integer> addedNodes		=	new ArrayList<Integer>();
		List<List<Integer>> addedEdges	=	new ArrayList<List<Integer>>();
		List<Integer> connectedPos		=	new ArrayList<Integer>();
		for(int i = 0;	i < matrix.size();	i++)
			connectedPos.add(0);
		int currentPos					=	0;
		
		while(addedNodes.size() < matrix.size()){
			distances	=	updateDistances(matrix, distances, currentPos, connectedPos);
			findNextMinimum(matrix, currentPos, connectedPos, addedNodes, addedEdges, distances);
			currentPos	=	addedNodes.get(addedNodes.size()-1);
		}
		
		return addedEdges;		
	}
	
	private static List<Double> updateDistances(List<List<Double>> matrix,
										List<Double> distances,
										int pos,
										List<Integer> connectedPos){

		if(distances.isEmpty()){
			if(pos == 0)//Initiate distances.
				for(int i = 0;	i < matrix.size();	i++)
					distances.add(matrix.get(i).get(0));
			
		}
		else{			
			for(int i = 0;	i < matrix.size();	i++){
				double increment	=	matrix.get(i).get(pos);
				double currentDist	=	distances.get(i);
				double posDist		=	distances.get(pos);
				if(increment + posDist < currentDist){
					distances.set(i, increment + posDist);
					connectedPos.set(i,	pos);
				}
			}
		}

		return distances;
	}
	
	private static void findNextMinimum(List<List<Double>> matrix, 
										int pos,
										List<Integer> connectingNodes,
										List<Integer> addedNodes, 
										List<List<Integer>> addedEdges,
										List<Double> distances){
		
		double minDistance	=	-1;
		int minPos			=	0;
		for(int i = 0;	i < distances.size();	i++){
			double dist	=	distances.get(i);
			if(i != pos && !addedNodes.contains(i)){
				if(minDistance == -1 || dist < minDistance){
					minDistance	=	dist;
					minPos	=	i;
				}
			}
		}
		
		List<Integer> addEdge	=	new ArrayList<Integer>();
		addEdge.add(connectingNodes.get(minPos));
		addEdge.add(minPos);
		addedEdges.add(addEdge);
		if(!addedNodes.contains(connectingNodes.get(minPos)))
			addedNodes.add(connectingNodes.get(minPos));
		//In for loop already checked that min pos is not present
		addedNodes.add(minPos);
	}
	
	private static boolean allZeroRow(List<Double> row){
		
		for(double d:row){
			if(d != 0.0)
				return false;
		}
		return true;
	}
	
	private static boolean allZeroColumn(List<List<Double>> matrix, int column){
		
		for(int i = 0;	i < matrix.size();	i++)
			if(matrix.get(i).get(column) != 0.0)
				return false;
		
		return true;
	}
	
	public static List<List<Double>> removeZeroRowAndColumn(List<List<Double>> matrix){
		//If matrix is empty, return empty matrix.		
		if(matrix.size() == 0){return matrix;}
		
		for(int i = 0;	i < matrix.size();	i++){
			List<Double> row	=	matrix.get(i);
			if(allZeroRow(row)){
				matrix.remove(row);
				i--;
			}
		}
		//If matrix is empty, return empty matrix.
		if(matrix.size() == 0){return matrix;}
		
		for(int i = 0;	i < matrix.get(0).size();	i++)
			if(allZeroColumn(matrix, i))
				for(int j = 0;	j < matrix.size();	j++)
					matrix.get(j).remove(i);
		
		return matrix;
	}
}