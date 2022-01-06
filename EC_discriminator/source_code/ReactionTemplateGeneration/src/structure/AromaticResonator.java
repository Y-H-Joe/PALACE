package structure;

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.openscience.cdk.CDKConstants;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IBond;
import org.openscience.cdk.interfaces.IRingSet;

import constants.StringConstants;
import reactionMapping.MappedReaction;

public class AromaticResonator {
	
	private static Logger logger		=	Logger.getLogger(AromaticResonator.class);
	
	private List<MappedReaction> possibilitiesForOneCombination;
	private boolean isReactant;
	private int molIndex;
	

	/**Generate a list of reactions in which the reactants have been resonated in all possible combinations.
	 * All possibilities will be generated for molecules containing up to two aromatic rings. In case there are three 
	 * rings present, resonance structures will be generated for some cases, more specifically: </br>
	 * - The rings do not share any atoms, in any way </br>
	 * - Two rings share two atoms, the third ring shares none with either of the two others.</br>
	 * 
	 * @param possibilitiesForOneCombination
	 * @param isReactant
	 * @param molIndex
	 * @param ringCount
	 */
	public static void resonate(List<MappedReaction> possibilitiesForOneCombination, 
								boolean isReactant, 
								int molIndex, 
								int ringCount){
		
		AromaticResonator resonator	=	new AromaticResonator(possibilitiesForOneCombination, isReactant, molIndex);
		
		switch(ringCount){
		
		case 0: break;//Do nothing, it is possible that neither of the rings detected was aromatic
		case 1:	resonator.resonateForOneRing();break;
		case 2: resonator.resonateForTwoRings();break;
		case 3: resonator.resonateForThreeRings();break;
		
		default: logger.error("Resonance not implemented for more than three rings in a molecule");

		}
	}

	/**Construct a new resonator
	 * 
	 * @param possibilitiesForOneCombination
	 * @param isReactant
	 * @param molIndex
	 * @category constructor
	 */
	private AromaticResonator(List<MappedReaction> possibilitiesForOneCombination, boolean isReactant, int molIndex){
	
		this.possibilitiesForOneCombination	=	possibilitiesForOneCombination;
		this.isReactant						=	isReactant;
		this.molIndex						=	molIndex;
	}
	
	/**Fill the possibilitiesForOneCombination with all resonance possibilities in case there is only one
	 * ring to take into account.
	 */
	private void resonateForOneRing(){
		List<MappedReaction> originals	=	new ArrayList<MappedReaction>();
		//Apply the resonance to each reaction currently in the list!
		for(MappedReaction reaction:possibilitiesForOneCombination){
			//keep originals too!
			originals.add(reaction);
			RingAnalyser ringAnalyser	=	null;
			if(isReactant)
				ringAnalyser	=	new RingAnalyser(reaction.getReactant(molIndex));
			else
				ringAnalyser	=	new RingAnalyser(reaction.getProduct(molIndex));

			IRingSet changedRings	=	ringAnalyser.getRings();
			
			resonate(changedRings.getAtomContainer(0),Names.O);	
		}
		possibilitiesForOneCombination.addAll(originals);
	}
	
	/**Fill the possibilitiesForOneCombination with all resonance possibilities in case there are two
	 * rings to take into account.
	 */
	private void resonateForTwoRings(){

		List<MappedReaction> allNewReactions	=	new ArrayList<MappedReaction>();

		for(MappedReaction reaction:possibilitiesForOneCombination){
			List<MappedReaction> newReactions	=	new ArrayList<MappedReaction>();
			RingAnalyser ringAnalyser	=	null;
			if(isReactant)
				ringAnalyser	=	new RingAnalyser(reaction.getReactant(molIndex));
			else
				ringAnalyser	=	new RingAnalyser(reaction.getProduct(molIndex));

			IRingSet changedRings	=	ringAnalyser.getRings();
			//Step one: check whether connected rings.
			int shared	=	ringAnalyser.countShared(0,1);
			//if two shared: connected rings, otherwise not. If the rings are connected: does not matter if one
			//of them has not changed: resonance will always affect both.
			if(shared == 2){

				resonateTwoFusedRings(changedRings,reaction,newReactions,isReactant,molIndex,0,1);
			}
			//if not connected: 
			//if both changed: adds three possibilities: rotate a (0), rotate b (1), rotate both (2).
			//if only one changed: only resonate that one.
			else{
				boolean changed1	=	changedRings.getAtomContainer(0).getProperty(Names.Changed);
				boolean changed2	=	changedRings.getAtomContainer(1).getProperty(Names.Changed);
				if(changed1&&changed2){

					resonateTwoSeparateRings(reaction, newReactions, isReactant, molIndex, Names.A, Names.B);
				}
				else if(changed1){

					resonateSingleRing(reaction, newReactions, isReactant, molIndex, Names.A);
				}
				else if(changed2){

					resonateSingleRing(reaction, newReactions, isReactant, molIndex, Names.B);
				}
			}
			allNewReactions.addAll(newReactions);
		}
		possibilitiesForOneCombination.addAll(allNewReactions);
	}
	
	/**Fill the possibilitiesForOneCombination with resonance possibilities in case there are three
	 * rings to take into account. Only those possibilities will be added when the rings either do not share any
	 * atoms at all, or at most two rings share atoms.
	 */
	private void resonateForThreeRings(){

		List<MappedReaction> allNewReactions	=	new ArrayList<MappedReaction>();
		
		for(MappedReaction reaction:possibilitiesForOneCombination){
			List<MappedReaction> newReactions	=	new ArrayList<MappedReaction>();
			RingAnalyser ringAnalyser	=	null;
			if(isReactant)
				ringAnalyser	=	new RingAnalyser(reaction.getReactant(molIndex));
			else
				ringAnalyser	=	new RingAnalyser(reaction.getProduct(molIndex));

			IRingSet changedRings	=	ringAnalyser.getRings();
			
			int sharedAB	=	ringAnalyser.countShared(0,1);
			int sharedAC	=	ringAnalyser.countShared(0,2);
			int sharedBC	=	ringAnalyser.countShared(1,2);

			if((sharedAB != 0 && sharedAC != 0) || (sharedAB !=0 && sharedBC != 0)){
				logger.error("Fused system with more than 2 rings. Not applying resonance");
			}
			else{
				boolean changed1	=	changedRings.getAtomContainer(0).getProperty(Names.Changed);
				boolean changed2	=	changedRings.getAtomContainer(1).getProperty(Names.Changed);
				boolean changed3	=	changedRings.getAtomContainer(2).getProperty(Names.Changed);

				if(sharedAB == 0 && sharedAC == 0 && sharedBC == 0){
					if(changed1&&changed2&&changed3){
						for(int k = 0; k < 7; k++)
							try {
								newReactions.add(reaction.clone());
							} catch (CloneNotSupportedException e) {}

						if(isReactant){
							resonate(newReactions.get(0).getReactant(molIndex),Names.A);
							resonate(newReactions.get(1).getReactant(molIndex),Names.B);
							resonate(newReactions.get(2).getReactant(molIndex),Names.C);
							resonate(newReactions.get(3).getReactant(molIndex),Names.AB);
							resonate(newReactions.get(4).getReactant(molIndex),Names.AC);
							resonate(newReactions.get(5).getReactant(molIndex),Names.BC);
							resonate(newReactions.get(6).getReactant(molIndex),Names.ABC);

						}
						else{
							resonate(newReactions.get(0).getProduct(molIndex),Names.A);
							resonate(newReactions.get(1).getProduct(molIndex),Names.B);
							resonate(newReactions.get(2).getProduct(molIndex),Names.C);
							resonate(newReactions.get(3).getProduct(molIndex),Names.AB);
							resonate(newReactions.get(4).getProduct(molIndex),Names.AC);
							resonate(newReactions.get(5).getProduct(molIndex),Names.BC);
							resonate(newReactions.get(6).getProduct(molIndex),Names.ABC);
						}
					}
					else if(changed1 && changed2){

						resonateTwoSeparateRings(reaction, newReactions, isReactant, molIndex, Names.A, Names.B);
					}
					else if(changed1 && changed3){

						resonateTwoSeparateRings(reaction, newReactions, isReactant, molIndex, Names.A, Names.C);
					}
					else if(changed2 && changed3){

						resonateTwoSeparateRings(reaction, newReactions, isReactant, molIndex, Names.B, Names.C);
					}
					else if(changed1){

						resonateSingleRing(reaction, newReactions, isReactant, molIndex, Names.A);
					}
					else if(changed2){

						resonateSingleRing(reaction, newReactions, isReactant, molIndex, Names.B);
					}
					else if(changed3){

						resonateSingleRing(reaction, newReactions, isReactant, molIndex, Names.C);
					}
				}
				//If one fused ring and a separate: apply resonance to the fused ring. Afterwards, apply resonance
				//to the separate one, then apply to both
				else if(sharedAB != 0){
					if(changed3){
						resonateSingleRing(reaction, newReactions, isReactant,molIndex,Names.C);				
					}
					//resonate ring system: once for the original reaction, once for each of the new reactions			//ring has already been resonated.
					int size	=	newReactions.size();
					for(int i = 0;	i < size;	i++)
						resonateTwoFusedRings(changedRings, newReactions.get(i), newReactions, isReactant, molIndex, 1, 2);
					resonateTwoFusedRings(changedRings, reaction, newReactions, isReactant, molIndex, 0, 1);
					
				}
				else if(sharedAC != 0){
					if(changed2){
						resonateSingleRing(reaction, newReactions, isReactant,molIndex,Names.B);				
					}
					//resonate ring system: once for the original reaction, once for each of the new reactions			//ring has already been resonated.
					int size	=	newReactions.size();
					for(int i = 0;	i < size;	i++)
						resonateTwoFusedRings(changedRings, newReactions.get(i), newReactions, isReactant, molIndex, 1, 2);
					resonateTwoFusedRings(changedRings, reaction, newReactions, isReactant, molIndex, 0, 2);
				}
				else if(sharedBC != 0){
					
					if(changed1){
						resonateSingleRing(reaction, newReactions, isReactant,molIndex,Names.A);				
					}
					//resonate ring system: once for the original reaction, once for each of the new reactions			//ring has already been resonated.
					int size	=	newReactions.size();
					for(int i = 0;	i < size;	i++){
						resonateTwoFusedRings(changedRings, newReactions.get(i), newReactions, isReactant, molIndex, 1, 2);
					}
					resonateTwoFusedRings(changedRings, reaction, newReactions, isReactant, molIndex, 1, 2);
				}
			}
			allNewReactions.addAll(newReactions);
		}
		possibilitiesForOneCombination.addAll(allNewReactions);
		
	}

	/**Add the possibilities to the newReactions list that arise from a two-ring fused system.
	 * 
	 * @param changedRings
	 * @param reaction
	 * @param newReactions
	 * @param isReactant
	 * @param molIndex
	 * @param ring1
	 * @param ring2
	 */
	private static void resonateTwoFusedRings(IRingSet changedRings, 
			MappedReaction reaction, 
			List<MappedReaction> newReactions, 
			boolean isReactant, 
			int molIndex,
			int ring1,
			int ring2){

		String prop1	=	StringConstants.EMPTY;
		String prop2	=	StringConstants.EMPTY;

		switch(ring1){
		case 0: prop1 = Names.A;break;
		case 1: prop1 = Names.B;break;
		case 2: prop1 = Names.C;break;
		default: break;
		}
		switch(ring2){
		case 0: prop2 = Names.A;break;
		case 1: prop2 = Names.B;break;
		case 2: prop2 = Names.C;break;
		default: break;
		}

		//Check how many double bonds each of the two rings has.
		int double1	=	changedRings.getAtomContainer(ring1).getDoubleBondCount(true);
		int double2	=	changedRings.getAtomContainer(ring2).getDoubleBondCount(true);
		//If both have three: two more resonance structures:
		if(double1 == 3 && double2 == 3){
			//Create an additional reaction: this combination implies two resonance structures.
			//The first can be performed on the "reaction", the second will be performed on a duplicate.
			for(int i = 0;	i < 2;	i++)
				try{
					newReactions.add(reaction.clone());
				}catch(CloneNotSupportedException e){}
			
			int l	=	newReactions.size();
			if(isReactant){
				resonate(newReactions.get(l-2).getReactant(molIndex),prop1);
				resonate(newReactions.get(l-1).getReactant(molIndex),prop2);
			}
			else{
				resonate(newReactions.get(l-2).getProduct(molIndex),prop1);
				resonate(newReactions.get(l-1).getProduct(molIndex),prop2);
			}
		}
		else if(double1 == 3){
			//Same comment as above
			for(int i = 0;	i < 2;	i++)
				try{
					newReactions.add(reaction.clone());
				}catch(CloneNotSupportedException e){}
			
			int l	=	newReactions.size();
			if(isReactant){
				resonate(newReactions.get(l-2).getReactant(molIndex),prop1);
				resonate(newReactions.get(l-1).getReactant(molIndex),prop1);
				resonate(newReactions.get(l-1).getReactant(molIndex),prop2);
			}
			else{
				resonate(newReactions.get(l-2).getProduct(molIndex),prop1);
				resonate(newReactions.get(l-1).getProduct(molIndex),prop1);
				resonate(newReactions.get(l-1).getProduct(molIndex),prop2);
			}
		}
		else if(double2 == 3){
			
			for(int i = 0;	i < 2;	i++)
				try{
					newReactions.add(reaction.clone());
				}catch(CloneNotSupportedException e){}

			int l	=	newReactions.size();
			if(isReactant){
				resonate(newReactions.get(l-2).getReactant(molIndex),prop2);
				resonate(newReactions.get(l-1).getReactant(molIndex),prop2);
				resonate(newReactions.get(l-1).getReactant(molIndex),prop1);
			}
			else{
				resonate(newReactions.get(l-2).getProduct(molIndex),prop2);
				resonate(newReactions.get(l-1).getProduct(molIndex),prop2);
				resonate(newReactions.get(l-1).getProduct(molIndex),prop1);
			}
		}
	}

	/**Add the possibilities for a molecule for which only the rings with the specified properties
	 * have to be resonated
	 * 
	 * @param reaction
	 * @param newReactions
	 * @param isReactant
	 * @param molIndex
	 * @param prop1
	 * @param prop2
	 */
	private static void resonateTwoSeparateRings(MappedReaction reaction, 
												 List<MappedReaction> newReactions, 
												 boolean isReactant, 
												 int molIndex,
												 String prop1,
												 String prop2){
		
		for(int k = 0; k < 3; k++)
			try {
				newReactions.add(reaction.clone());
			} catch (CloneNotSupportedException e) {}
		
		int l	=	newReactions.size();
		
		if(isReactant){
			resonate(newReactions.get(l-3).getReactant(molIndex),prop1);
			resonate(newReactions.get(l-2).getReactant(molIndex),prop2);
			resonate(newReactions.get(l-1).getReactant(molIndex),prop1+prop2);
		}
		else{
			resonate(newReactions.get(l-3).getProduct(molIndex),prop1);
			resonate(newReactions.get(l-2).getProduct(molIndex),prop2);
			resonate(newReactions.get(l-1).getProduct(molIndex),prop1+prop2);
		}
	}

	/**Add the possibilities for a molecule in which only the ring with the specified property must be resonated.
	 * 
	 * @param reaction
	 * @param newReactions
	 * @param isReactant
	 * @param molIndex
	 * @param prop
	 */
	private static void resonateSingleRing(MappedReaction reaction,
										   List<MappedReaction> newReactions,
										   boolean isReactant, 
										   int molIndex, 
										   String prop){
		
		try {
			newReactions.add(reaction.clone());
		} catch (CloneNotSupportedException e) {}
		
		int l	=	newReactions.size();
		
		if(isReactant){
			resonate(newReactions.get(l-1).getReactant(molIndex),prop);
		}
		else{
			resonate(newReactions.get(l-1).getProduct(molIndex),prop);
		}
		
	}
	
	/**Changes the order of all bonds, marked with the correct property.
	 *  
	 * @param mol
	 * @param property
	 */	
	private static void resonate(IAtomContainer mol, String property) {
		
		for (IBond b : mol.bonds()){
			IAtom atom1	=	b.getAtom(0);
			IAtom atom2	=	b.getAtom(1);			
			//Check that the first atom belongs to the correct ring;
			boolean check1 = false;
			switch(property){
			case Names.A: check1 = atom1.getProperty(Names.A)==null?false:true;break;
			case Names.B: check1 = atom1.getProperty(Names.B)==null?false:true;break;
			case Names.C: check1 = atom1.getProperty(Names.C)==null?false:true;break;
			case Names.O: check1 = true;break;
			case Names.AB:check1 = (atom1.getProperty(Names.A)==null?false:true) || (atom1.getProperty(Names.B)==null?false:true); break;
			case Names.AC:check1 = (atom1.getProperty(Names.A)==null?false:true) || (atom1.getProperty(Names.C)==null?false:true); break;
			case Names.BC:check1 = (atom1.getProperty(Names.B)==null?false:true) || (atom1.getProperty(Names.C)==null?false:true); break;
			case Names.ABC:check1= (atom1.getProperty(Names.A)==null?false:true) || (atom1.getProperty(Names.B)==null?false:true) || (atom1.getProperty(Names.C)==null?false:true); break;
			}
			//Check the second one
			boolean check2 = false;
			switch(property){
			case Names.A: check2 = atom2.getProperty(Names.A)==null?false:true;break;
			case Names.B: check2 = atom2.getProperty(Names.B)==null?false:true;break;
			case Names.C: check2 = atom2.getProperty(Names.C)==null?false:true;break;
			case Names.O: check2 = true;break;
			case Names.AB:check2 = (atom2.getProperty(Names.A)==null?false:true) || (atom2.getProperty(Names.B)==null?false:true); break;
			case Names.AC:check2 = (atom2.getProperty(Names.A)==null?false:true) || (atom2.getProperty(Names.C)==null?false:true); break;
			case Names.BC:check2 = (atom2.getProperty(Names.B)==null?false:true) || (atom2.getProperty(Names.C)==null?false:true); break;
			case Names.ABC:check2= (atom2.getProperty(Names.A)==null?false:true) || (atom2.getProperty(Names.B)==null?false:true) || (atom2.getProperty(Names.C)==null?false:true); break;
			}
			
			if(check1&&check2){	
				changeBondOrder(b);
			}
		}
	}
	
	/**Changes the bond orders of all aromatic bonds in a molecule from single to double and vice versa.
	 * 
	 * @param mol
	 */
	private static void changeBondOrder(IBond b) {
		//Only change if aromatic (whether in correct ring has been checked in aromaticResonanceOneRing)
		if(b.getFlag(CDKConstants.ISAROMATIC)){
			if(b.getOrder() == IBond.Order.SINGLE){
				b.setOrder(IBond.Order.DOUBLE);
			}
			else if(b.getOrder() == IBond.Order.DOUBLE){
				b.setOrder(IBond.Order.SINGLE);
			}
		}
	}
}
