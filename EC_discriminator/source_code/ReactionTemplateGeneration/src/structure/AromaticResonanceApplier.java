package structure;

import java.util.ArrayList;
import java.util.List;

import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IRingSet;

import reactionMapping.MappedReaction;
import tools.MyMath;

public class AromaticResonanceApplier {
	
	/**Resonates the aromatic structure: double and single bonds are changed.
	 * Only applicable to single and double ring aromatic structures and a limited number of cases with three rings
	 * in a single molecule.
	 * 
	 * @param mol
	 */
	public static List<MappedReaction> applyResonance(MappedReaction mappedReaction, List<IAtomContainer> aromaticMols){
		//All possible combinations of aromatic resonance structures (per molecule)
		//Each line contains flags whether or not the corresponding molecule should be resonated. 
		//Does not yet account for different resonance structures of the molecule. 
		//Eg. if 2 aromatic molecules combinations will look like:
		//[0,0] -> neither of the two will be resonated (ie original)
		//[0,1] -> the second is resonated, the first is original
		//[1,0] -> ...
		//[1,1] -> ...
		boolean[][] combinations	=	new boolean[(int)Math.pow(2, aromaticMols.size())][aromaticMols.size()];
		fillChangeArray(combinations);		
		
		List<MappedReaction> allPossibleResonances	=	new ArrayList<MappedReaction>();

		for(int i = 1;	i < combinations.length;	i++){
			List<MappedReaction> possibilitiesForOneCombination	=	new ArrayList<MappedReaction>();
			MappedReaction clonedReaction	=	new MappedReaction();
			try {
				clonedReaction	=	mappedReaction.clone();
				possibilitiesForOneCombination.add(clonedReaction);
			} catch (CloneNotSupportedException e) {}
			
			for(int j = 0;	j < aromaticMols.size();	j++){
				if(combinations[i][j]){
					IAtomContainer aromaticMol	=	aromaticMols.get(j);
					int[] idReact				=	mappedReaction.identifyReactant(aromaticMol);
					boolean isReactant			=	idReact[0]==1?true:false;
					int molIndex				=	idReact[1];
					IAtomContainer clonedMol	=	isReactant?
													/*Yes*/clonedReaction.getReactant(molIndex):
													/*No */clonedReaction.getProduct(molIndex);
					IRingSet changedRings		=	new RingAnalyser(clonedMol).getRings(); 
					
					AromaticResonator.resonate(possibilitiesForOneCombination, 
									   isReactant, 
									   molIndex, 
									   changedRings.getAtomContainerCount());
				}
			}
			allPossibleResonances.addAll(possibilitiesForOneCombination);
			
		}
		return allPossibleResonances;
	}
		
	/**Create an array of arrays to store which molecules should be resonated in case of multiple aromatic molecules
	 * participating in the reaction.
	 * 
	 * @param change
	 */
	private static void fillChangeArray(boolean[][] change){
		int[] bit	=	new int[change[0].length];
		
		for(int i = 0;	i < change.length;	i++){
			for(int j = 0;	j < bit.length;	j++){
				if(bit[j] == 0){
					change[i][j]	=	false;
				}
				else{
					change[i][j]	=	true;
				}
			}
			if(i != change.length - 1)
				advanceOne(bit);
		}
	}
	
	/**Increase the bit array by one
	 * 
	 * @param bit
	 */
	private static void advanceOne(int[] bit){
		MyMath.addOne(bit, bit.length-1);
	}
}