package structure;

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IAtomContainerSet;
import org.openscience.cdk.tools.StructureResonanceGenerator;

import reactionMapping.MappedReaction;
import tools.MyMath;

public class Resonance {

	private static Logger logger	=	Logger.getLogger(Resonance.class);
	
	private MappedReaction mappedrxn;
	private List<IAtomContainerSet> reactantResonance;
	private List<IAtomContainerSet> productResonance;
	private List<MappedReaction> reactions;
	private int totalCombinations;
	
	public Resonance(MappedReaction reaction){
		
		this.mappedrxn		=	reaction;
		reactions			=	new ArrayList<MappedReaction>();
		reactantResonance	=	new ArrayList<IAtomContainerSet>();
		productResonance	=	new ArrayList<IAtomContainerSet>();
		totalCombinations	=	1;
	}
	
	/**Get the number of generated resonance structure combinations.
	 * 
	 * @return
	 */
	public int count(){
		return totalCombinations;
	}
	
	/**Get the possible reactions, taking into account resonance.
	 * 
	 * @return list of reactions
	 */
	public List<MappedReaction> getPossibleReactions(){
		
		findStructures();
		try{
		findAllCombinations();
		}
		catch(CloneNotSupportedException e){
			logger.warn("Failed to clone molecules for resonance structure generation, not taking resonance into account.");
			reactions.add(mappedrxn);
		}
		return reactions;
	}
	
	/**Find all possible combinations of resonance structures for the reaction and return a set of reactions
	 * corresponding to these changes.
	 * @throws CloneNotSupportedException 
	 */
	private void findAllCombinations() throws CloneNotSupportedException{
		
		int reactCount	=	mappedrxn.getReactantCount();
		int prodCount	=	mappedrxn.getProductCount();
		int count		=	reactCount + prodCount;
		int[] indices	=	new int[count];
		int[] max		=	new int[count];
		
		//The number of possible resonance structures per reactant/product: this is necessary for the permutations!
		for(int i = 0;	i < reactCount;	i++){
			max[i]	=	reactantResonance.get(i).getAtomContainerCount();
		}
		for(int i = reactCount;	i < count; i++){
			max[i]	=	productResonance.get(i-reactCount).getAtomContainerCount();
		}
		
		do{
			MappedReaction reaction	=	new MappedReaction();
			int i = 0;
			
			while(i < reactCount){
				reaction.addReactant(reactantResonance.get(i).getAtomContainer(indices[i]).clone());
				i++;
			}
			while(i >= reactCount && i < count){
				reaction.addProduct(productResonance.get(i - reactCount).getAtomContainer(indices[i]).clone());
				i++;
			}
			
			reaction.makeNewMapping(mappedrxn);
			reaction.reassignMappings();
		
			reactions.add(reaction);
		}
		while(MyMath.nextPermutation(indices, max, count - 1));
	}
	
	/**Find all possible resonance structures.
	 * 
	 */
	private void findStructures(){

		//Do not take symmetry into account! Atom mappings are sensitive to exact correspondence of atoms, so
		//resonance structures of symmetrical molecules are NOT equivalent here.
		StructureResonanceGenerator resonanceGen	=	new StructureResonanceGenerator(false);

		for(IAtomContainer reactant:this.mappedrxn.getReactants().atomContainers()){
			IAtomContainerSet structures	=	resonanceGen.getStructures(reactant);
			reactantResonance.add(structures);
			totalCombinations*=	structures.getAtomContainerCount();
		}

		for(IAtomContainer product:this.mappedrxn.getProducts().atomContainers()){
			IAtomContainerSet structures	=	resonanceGen.getStructures(product);
			productResonance.add(structures);
			totalCombinations*=	structures.getAtomContainerCount();
		}
	}
}
