package fileGeneration.generators;

import java.util.ArrayList;
import java.util.List;

import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IBond.Order;
import org.openscience.cdk.ringsearch.RingSearch;

import changes.Change;
import constants.StringConstants;
import io.readers.xmlClasses.ConstraintType;
import io.readers.xmlClasses.InpConstraint;
import io.readers.xmlClasses.InpReactant;
import io.readers.xmlClasses.InpReactiveCenter;
import io.readers.xmlClasses.InpRecipe;
import io.readers.xmlClasses.InpTransformation;
import io.readers.xmlClasses.LimitType;
import io.readers.xmlClasses.TransformationType;
import manipulators.SMARTSManipulator;
import reactionCenterDetection.ReactionCenter;

public class RecipeAndReactantsGenerator {
		
	private ReactionCenter RC;
	
	private RecipeAndReactantsGenerator(ReactionCenter RC){
		
		this.RC					=	RC;
	}
	
	/**Generate the recipe block for the reaction family corresponding to the given ReactionCenter RC
	 * 
	 * @param RC
	 * @return recipe block
	 */
	private InpRecipe generateRecipe(){
		
		InpRecipe recipe	=	new InpRecipe();
		
		//Each transformation corresponds to one change of RC: algorithm asserts that no double changes are
		//present in the RC
		for(int i = 0;	i < RC.getChanges().size();	i++){
			Change change			=	RC.getChanges().get(i);
			InpTransformation trans	=	new InpTransformation();
			
			trans.setType(TransformationType.fromValue(change.getChangeType().name()));
			//Each atom in RC has been assigned an letter, which belongs to the transformation
			String centers	=	"";
			for(int j = 0;	j < change.getAtoms().length;	j++){
				if(j == change.getAtoms().length-1){
					centers	+=	change.getAtoms()[j].getProperty(StringConstants.RECIPECENTER);
				}
				
				else{
					centers	+=	change.getAtoms()[j].getProperty(StringConstants.RECIPECENTER)+",";
				}
			}
			
			trans.setCenters(centers);
			recipe.addInpTransformation(trans);
		}
		
		return recipe;
	}
	
	/**Generate the reactant block(s) for the reaction family corresponding the the given ReactionCenter RC
	 * 
	 * @param RC
	 * @return reactant blocks
	 */
	private List<InpReactant> generateReactants(){
		
		List<InpReactant> reactants	=	new ArrayList<InpReactant>();
		
		for(int i = 0;	i < RC.getReactantCount();	i++){
			reactants.add(generateReactant(i));
		}
		
		return reactants;
	}
	
	/**Generates the block for one reactant[index] of the ReactionCenter RC
	 * 
	 * @param RC
	 * @param index
	 * @return reactant[index] block
	 */
	private InpReactant generateReactant(int index){
		
		InpReactant reactant	=	new InpReactant();
		
		reactant.setSmarts(RC.getSmartsI(index));
		reactant.setValue(index+1);
		
		IAtomContainer React	=	RC.getReactant(index);
		
		for(int i = 0;	i < RC.getCenters().size();	i++){
			IAtom atom	=	RC.getCenters().get(i);
			//Link the correct SMARTS representation of the center to the correct letter.
			if(React.contains(atom)){
				InpReactiveCenter reactiveCenter	=	new InpReactiveCenter();
				String neigh						=	getANeighbour(atom);
				
				if(neigh.equals(StringConstants.EMPTY)){/*If no neighbour found, don't set a neighbour*/}
				else	reactiveCenter.setNeighbour(neigh);
				
				reactiveCenter.setSmarts("[$("+SMARTSManipulator.setAsFirst(React,React.getAtomNumber(atom))+")]");
				reactiveCenter.setSymbol(atom.getProperty(StringConstants.RECIPECENTER));
				reactant.addInpReactiveCenter(reactiveCenter);
			}
		}
		
		List<InpConstraint> constraints	=	generateConstraints(React, index);
		
		for(InpConstraint constraint:constraints)
			reactant.addInpMoleculeConstraint(constraint);
		
		return reactant;
	}
	
	/**Set two default molecular constraints: <br>
	 * max number of carbon atoms is Max(4 , 4 + Number of Carbon atoms in the ReactionCenter<br>
	 * max number of radicals present is the number of radicals in the reaction reactant<br>
	 * min number of radicals present is the number of radicals in the reaction center reactant<br>
	 * TODO: Find a way to make this better
	 * @return constraint line
	 */
	private List<InpConstraint> generateConstraints(IAtomContainer reactant, int index){
		
		InpConstraint constraint1	=	new InpConstraint();
		InpConstraint constraint2	=	new InpConstraint();
		
		constraint1.setType(ConstraintType.SINGLEELECTRONCOUNT);
		constraint2.setType(ConstraintType.SINGLEELECTRONCOUNT);
		
		int centerSE	=	reactant.getSingleElectronCount();
		int reactSE		=	RC.getReaction().getReactant(index).getSingleElectronCount();
		//At least the number of SE in the reaction center
		constraint1.setLimit(LimitType.MIN);
		constraint1.setValue(centerSE);
		//At most the number of SE in the reactant in the reaction
		constraint2.setLimit(LimitType.MAX);
		constraint2.setValue(reactSE);
		
		List<InpConstraint> constraints	=	new ArrayList<InpConstraint>();
		
		constraints.add(constraint1);
		constraints.add(constraint2);
		
		return constraints;
	}
	
	/**Get the center symbol (letter) of one of the neighbouring atoms of the specified atom (which must belong
	 * to the RC)
	 * 
	 * @param RC
	 * @param atomOfCenter
	 * @return letter identifying the atom in the reactive center
	 */
	private String getANeighbour(IAtom atomOfCenter){
		
		IAtomContainer react		=	RC.getReactant(atomOfCenter);
		List<IAtom> neighbours	=	react.getConnectedAtomsList(atomOfCenter);
		
		String neigh	=	StringConstants.EMPTY;
		String out		=	StringConstants.EMPTY;
		
		//Check for neighbours that are unlikely to be symmetrical. (Genesys is very sensitive to symmetry)
		for(IAtom neighbour:neighbours){
			List<IAtom> reacts	=	new ArrayList<IAtom>();
			reacts.add(atomOfCenter);
			reacts.add(neighbour);
			
			//Search for a neighbour that is singly bonded: reduces the chance of symmetry.
			//Search for a neighbour that is not in a ring: reduces the chance of symmetry.
			boolean notSymmetrical	=	RC.getReactant(atomOfCenter).getBond(atomOfCenter, neighbour).getOrder() == Order.SINGLE;
			boolean notInRing		=	!new RingSearch(react).cyclic(neighbour);
			boolean notHydrogen		=	!(neighbour.getSymbol().equals("H") || neighbour.getAtomicNumber()==1);
			
			if(RC.isInCenter(neighbour) && notSymmetrical && notInRing && notHydrogen){
				neigh	=	neighbour.getProperty(StringConstants.RECIPECENTER);
			}
		}
		//If no non-symmetrical neighbours are found, take the first (non-hydrogen) one that is in the center.
		if(neigh.equals(StringConstants.EMPTY)){
			for(IAtom neighbour:neighbours){
				boolean notHydrogen		=	!(neighbour.getSymbol().equals("H") || neighbour.getAtomicNumber()==1);
				if(RC.isInCenter(neighbour) && notHydrogen){
					neigh	=	neighbour.getProperty(StringConstants.RECIPECENTER);
					break;
				}
			}
		}
		//if still not found, take first
		
		if(neigh.equals(StringConstants.EMPTY)){
			for(IAtom neighbour:neighbours){
				if(RC.isInCenter(neighbour)){
					neigh	=	neighbour.getProperty(StringConstants.RECIPECENTER);
					break;
				}
			}
		}
		
		if(!neigh.equals("")){
			out	=	neigh;
		}
		
		return out;
	}
	
	protected static List<Object> generate(ReactionCenter RC){
		
		RecipeAndReactantsGenerator rRC	=	new RecipeAndReactantsGenerator(RC);
		List<Object> output	=	new ArrayList<Object>();
		output.add(rRC.generateRecipe());
		output.add(rRC.generateReactants());
		
		return output;
	}
}