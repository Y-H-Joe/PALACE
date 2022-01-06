package reactionCenterDetection;

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.openscience.cdk.AtomContainer;
import org.openscience.cdk.CDKConstants;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IAtomContainerSet;
import org.openscience.cdk.interfaces.IBond;
import org.openscience.cdk.interfaces.IBond.Order;
import org.openscience.cdk.smiles.SmilesGenerator;

import changes.ChangeDetector;
import constants.StringConstants;
import manipulators.HydrogenAdder;
import manipulators.SMARTSManipulator;
import reactionMapping.MappedReaction;
import structure.AromaticResonanceApplier;
import structure.identifiers.InChIGeneratorWrapper;
import tools.SMARTSGenerator;

public class ReactionCenterDetector {
	
	private static Logger logger	=	Logger.getLogger(ReactionCenterDetector.class);
	private MappedReaction mappedrxn;
	private boolean aromatic	=	false;
	private boolean specific	=	false;
	
	public ReactionCenterDetector(MappedReaction mappedrxn){
		
		this.mappedrxn	=	mappedrxn;
	}
	
	public ReactionCenterDetector(MappedReaction mappedrxn, boolean specific){
		
		this.mappedrxn	=	mappedrxn;
		this.specific	=	specific;
	}
	
	public void setSpecific(boolean specific){
		this.specific	=	specific;
	}
	
	public MappedReaction configureReaction(){
		
		for(IAtomContainer ac:mappedrxn.getReactants().atomContainers()){
			for(IAtom at:ac.atoms()){
				hasChanged(at);
			}	
		}
		
		for(IAtomContainer reactant:this.mappedrxn.getReactants().atomContainers()){
			this.makeHydrogenExplicit(reactant, reactant);
			this.reviseHydrogenCounts(reactant);
		}
		
		return this.mappedrxn;
	}
	/**Returns a Reaction center containing the Genesys-compatible SMARTS-representations of the reactive centers
	 * and the actual changes that result from the reaction. 
	 * 
	 * @return
	 */
	public ReactionCenter detectReactionCenters(){
		
		SmilesGenerator sg	=	SmilesGenerator.generic().aromatic();
		String reaction		=	"Failed to create reaction SMILES";
				
		try {
			reaction	=	sg.createReactionSMILES(mappedrxn.getReaction());
		} catch (CDKException e) {
			e.printStackTrace();
		}
		
		logger.info("Starting reaction center detection for: "+reaction);
		long start_time 	=	System.nanoTime();
		
		this.setChangedAtoms();
		
		String[] SMARTS		=	new String[mappedrxn.getReactantCount()];
		
		IAtomContainer[] ReactCent	=	new IAtomContainer[mappedrxn.getReactantCount()];
		IAtomContainerSet Reactants	=	mappedrxn.getReactants();
		
		String info	=	"";
		
		//assess for each atom if and what has changed (and set as property)
		
		ReactionCenter RC	=	new ReactionCenter(mappedrxn);
		
		for(int i = 0;	i < mappedrxn.getReactantCount();	i++){
			
			//Start detection of changed atoms and removal of unchanged atoms in current reactant
			IAtomContainer reactant		=	Reactants.getAtomContainer(i);
			IAtomContainer clone		=	new AtomContainer(reactant);
	
			this.setAtomsToBeRemoved(reactant);

			//For intra-molecular reactions, it is important that the atoms between the two ends are connected to obtain
			//a connected (ie single) molecular fragment.
			this.ensureConnectedCenter(reactant);

			this.removeAtoms(clone);
			//end atom removal
			
			this.generaliseConnection(clone);
			//Hydrogens are not mapped by rdt, however, they can be important in the mechanism. Therefore
			//here explicit hydrogens are added where necessary.
			
			this.makeHydrogenExplicit(reactant, clone);
			//Due to removal of non-changing atoms, for some atoms a discrepancy may have arisen between the
			//actual number of connected atoms (connected atoms+implicit hydrogens) and the original number of
			//connected atoms (formal neighbour count). This will be remedied by adding implicit hydrogens
			//where discrepancies are found
			
			this.reviseHydrogenCounts(clone);
			
			//If all atoms have been removed, add just one in order to satisfy the 'bimolecular' trait!
			if(clone.getAtomCount() == 0){	
				clone	=	new AtomContainer(reactant);
				for(int a = 1;	a < reactant.getAtomCount();	a++){
					clone.removeAtomAndConnectedElectronContainers(reactant.getAtom(a));
				}
				clone.getAtom(0).setProperty(StringConstants.AVOIDEMPTY, true);
			}
						
			//Generate identifier for reaction center.
			InChIGeneratorWrapper.generate(clone);
			this.resetAromaticity(clone);
			
			ReactCent[i]				=	clone;
			int[] order					=	new int[clone.getAtomCount()];
			SMARTSGenerator smg			=	new SMARTSGenerator(clone, order);
			SMARTS[i]					=	smg.getSMARTSConnection();
			
			RC.setOrdersI(order,i);
			info	+=	"\t" + SMARTS[i];
		
		}
		
		RC.setReactants(ReactCent);
		RC.determineChanges();
		RC.setSmarts(SMARTS);		
		RC.setCenters();
		
		long end_time	=	System.nanoTime();
		double deltat	=	((long)((end_time - start_time) / 1000000)) / 1000.0;
		
		logger.info("Reaction center detected: "+info+
				"\nTime spent: " + deltat + " seconds");
		return RC;
	}
	
	/**Check which atoms should be removed and flag them.
	 * 
	 * @param reactant
	 */
	private void setAtomsToBeRemoved(IAtomContainer reactant){
		for(int j = 0;	j < reactant.getAtomCount();	j++){
			IAtom atom	=	reactant.getAtom(j);
			//If not changed...
			if(!(boolean) atom.getProperty(StringConstants.CHANGED)){
				//.. do not remove atom if neighbouring atom's stereo has changed <-> ignore for the moment
				if((boolean) atom.getProperty(StringConstants.NEIGHBOURSTEREO)){
					//atom.setProperty(StringConstants.SHOULDKEEP, true);
				}
				//If specific is true: make center more specific by adding peripherial atoms.
				else if(specific){
					//TODO: Include more atoms to reaction center if necessary		
					List<IBond> bonds	=	reactant.getConnectedBondsList(atom);
					
					for(IBond bond:bonds){
						//..do not remove atom if neighbouring atom has changed and 
						//- the connecting bond is not single
						//- the atom is neither hydrogen or carbon
						if((boolean)bond.getConnectedAtom(atom).getProperty(StringConstants.CHANGED)
							&&(!bond.getOrder().equals(Order.SINGLE)
							   ||
							   (!atom.getSymbol().equals("H")&&!atom.getSymbol().equals("C"))
							   )
						  ){	
							atom.setProperty(StringConstants.SHOULDKEEP,true);
						}
					}
					//if not yet set: remove
					if(atom.getProperty(StringConstants.SHOULDKEEP) == null){
						atom.setProperty(StringConstants.SHOULDKEEP, false);
					}
				}
				//if still not set: remove
				if(atom.getProperty(StringConstants.SHOULDKEEP) == null){
					atom.setProperty(StringConstants.SHOULDKEEP, false);
				}// Only necessary if not adding any extra: atom.setProperty(StringConstants.SHOULDKEEP, false);
			}
			//If changed: keep!
			else{
				atom.setProperty(StringConstants.SHOULDKEEP, true);
			}
		}
	}
	
	/**Remove all atoms for which the "SHOULDKEEP" property has been set to false
	 * 
	 * @param reactant
	 */
	private void removeAtoms(IAtomContainer reactant){
		
		int atomCount	=	reactant.getAtomCount();
		for(int atom = 0;	atom < atomCount;	atom++){
			if(!(boolean) reactant.getAtom(atom).getProperty(StringConstants.SHOULDKEEP)){
				reactant.removeAtomAndConnectedElectronContainers(reactant.getAtom(atom));
				atom--;
				atomCount--;
			}
		}
	}
	
	/**Check if the atom has undergone any form of chemical change during the reaction
	 * <br>
	 * @param mappedAtom
	 * @return
	 */
	private boolean hasChanged(IAtom mappedAtom){
		
		ChangeDetector cd	=	new ChangeDetector(mappedrxn, mappedAtom);
		
		return cd.detectChanged();
	}
	
	/**For each reactant atom, assess whether or not it has changed and what type of change it is.
	 * The changes are stored with the atom as property.
	 */
	private void setChangedAtoms(){
		
		int numbChanged		=	0;

		List<IAtomContainer> flags	=	new ArrayList<IAtomContainer>();
		for(IAtomContainer ac:mappedrxn.getReactants().atomContainers()){
			for(IBond bo:ac.bonds())
				bo.setProperty(SMARTSManipulator.Aromatic, bo.getFlag(CDKConstants.ISAROMATIC));
			
			
			for(IAtom at:ac.atoms()){
				hasChanged(at);
				//Set aromaticity property for smarts generation (remember aromaticity even after removing non reacting atoms).
				at.setProperty(SMARTSManipulator.Aromatic, at.getFlag(CDKConstants.ISAROMATIC));
				
				boolean aromaticR	=	false;
				boolean aromaticP	=	false;
				
				//If changed increase the changes count
				if((boolean)at.getProperty(StringConstants.CHANGED))
					numbChanged++;
				
				//If bonds have changed, consider aromaticity.
				if((boolean)at.getProperty(StringConstants.BONDCHANGE)){
					if(at.getFlag(CDKConstants.ISAROMATIC)){
						
						aromaticR	=	true;
						if(!flags.contains(ac))
							flags.add(ac);
					}
					//If the reactant is aromatic AND the product is aromatic, it suffices to just ignore the bond
					//order. This is done in hasChanged(IAtom) and finally in ChangeDetector.bondsChanged().
					else if(mappedrxn.getMappedProductAtom(at).getFlag(CDKConstants.ISAROMATIC)){
						aromaticP	=	true;
						IAtomContainer ac2	=	mappedrxn.getProduct(mappedrxn.getContainerIndexMappedAtom(at));
						if(!flags.contains(ac2))
							flags.add(ac2);	
					}
					//Exclusive or: only test different aromatic resonances if aromaticity is gained or lost.
					//Only change if aromatic is false, if true, aromatic change detected and leave at true
					if(!aromatic) aromatic	=	aromaticR||aromaticP&&!(aromaticR&&aromaticP);
				}
			}
		}
		//This part is now done during RDT mapping: map resonance structures to avoid starting from wrong mapping.
		//Aromatic resonance is still done here due to the assumption that aromatic compounds will most likely be 
		//correctly mapped.
		/*
		//Use the combination of resonance structures that results in the lowest number of changes.
		Resonance resonance	=	new Resonance(mappedrxn);
		List<MappedReaction> reactionResonance		=	resonance.getPossibleReactions();
		
		//If only one possiblility, don't bother
		if(reactionResonance.size() != 1){
			logger.info("Multiple resonance structures detected. Number of possible combinations: "+resonance.count());
			
			this.mappedrxn					=	getBestResonance(reactionResonance, numbChanged);
		}*/
		
		if(aromatic){
			logger.info("Aromaticity detected. Number of molecules involved: "+flags.size());
			List<MappedReaction> possibilities	=	AromaticResonanceApplier.applyResonance(mappedrxn,flags);
			logger.info("Total number of resonance possibilities: "+possibilities.size());

			this.mappedrxn					=	getBestResonance(possibilities, numbChanged);
		}
	}
	
	/**Set the changes for each of the options in the array. Returns an array of integers indicating of how many
	 * atoms bonds were changed.
	 * Resonance type can be aromatic or radical, ....
	 * 
	 * @param options
	 * @return changes per resonance structure
	 */
	private int[] setResonanceChanges(List<MappedReaction> options){
		int [] changed					=	new int[options.size()];
		int possibilityCounter			=	0;
		for(MappedReaction option:options){
			
			for(IAtomContainer ac:option.getReactants().atomContainers()){
				for(IAtom at:ac.atoms()){
					ChangeDetector cd	=	new ChangeDetector(option, at);
					cd.detectChanged();
					if((boolean)at.getProperty(StringConstants.CHANGED)){
						changed[possibilityCounter]++;
					}
				}
			}
			possibilityCounter++;
		}
		
		return changed;
	}
	
	/**Mark additional atoms as "SHOULDKEEP" to ensure that the resulting reaction center is connected
	 * 
	 * @param container
	 */
	private void ensureConnectedCenter(IAtomContainer container){
		
		Connector connector	=	new Connector(container);
		
		try{
			connector.connectAndSetProperties();
		}
		catch(CDKException e){
			logger.error("Failed to keep enough atoms to make the reaction center connected!");
		}
	}
	
	/**Makes a number of implicit hydrogens explicit, depending on by how much the hydrogen count changes during
	 * the reaction. The explicit hydrogens are not added to the reactant, but to the clone.
	 * 
	 * @param reactant
	 * @param clone
	 */
	private void makeHydrogenExplicit(IAtomContainer reactant, IAtomContainer clone){
		
		int atc	=	reactant.getAtomCount();
		
		for(int j = 0;	j < atc;	j++){
			int hydrogenChange	=	ChangeDetector.hydrogenHasDecreased(mappedrxn, reactant.getAtom(j), true);
			if(hydrogenChange > 0 && reactant.getAtom(j).getImplicitHydrogenCount() != 0){
				HydrogenAdder hydrogenAdder	=	new HydrogenAdder(Math.min(reactant.getAtom(j).getImplicitHydrogenCount(),hydrogenChange));
				hydrogenAdder.addHydrogens(clone, reactant.getAtom(j));
				hydrogenAdder.setChanges();
			}
		}
	}
	
	private void reviseHydrogenCounts(IAtomContainer container){
		
		for(int j = 0;	j < container.getAtomCount();	j++){
			int remaining	=	container.getAtom(j).getImplicitHydrogenCount()
								+(int)container.getBondOrderSum(container.getAtom(j));
			
			IAtom atom		=	container.getAtom(j);
			int original	=	atom.getValency();
					
			if(remaining < original){
				int newHC	=	atom.getImplicitHydrogenCount()+(original-remaining);
				atom.setProperty(StringConstants.OLDHydrogenCount, atom.getImplicitHydrogenCount());
				atom.setImplicitHydrogenCount(newHC);
			}
		}
	}
	
	/**Select the reaction with minimal changes from a list. Also provide the number of changes in the original
	 * reaction
	 * 
	 * @param possibilities
	 * @param changedOriginal
	 * @return reaction with least changes form list.
	 */
	private MappedReaction getBestResonance(List<MappedReaction> possibilities, int changedOriginal){
		
		int[] changed		=	setResonanceChanges(possibilities);
		int best			=	changedOriginal;
		int bestLocation	=	-1;
		
		//Find resonance structure resulting in minimal number of bond changes.
		for(int i = 0;	i < changed.length;	i++){
			logger.info("Resonance combination "+(i+1)+" has "+changed[i]+" changed atoms");
			if(changed[i] < best){
				best = changed[i];
				bestLocation = i;
			}
		}
		
		return	bestLocation == -1? mappedrxn: possibilities.get(bestLocation);

	}
	
	private void generaliseConnection(IAtomContainer mol){
		final String prop	=	"Not changed yet";
		for(IBond bond:mol.bonds()){
			if(bond.getProperty(prop) == null)
				bond.setProperty(prop, true);
			for(IAtom atom:bond.atoms())
				if(atom.getProperty(StringConstants.LINK) == null){}
				else if((boolean) atom.getProperty(StringConstants.LINK) && (boolean)bond.getProperty(prop)){
					bond.setOrder(Order.SINGLE);
					bond.setProperty(prop, false);
				}
		}
		// Do not do additional perception step. Preception should be fine at this point,
		// and valences and atom types should not change after this step.
		/*
		try{
			AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(mol);
		}
		catch(CDKException e){}*/
	}
	
	private void resetAromaticity(IAtomContainer mol){
		
		for(IAtom at:mol.atoms())
			at.setFlag(CDKConstants.ISAROMATIC, at.getProperty(SMARTSManipulator.Aromatic)==null?false:at.getProperty(SMARTSManipulator.Aromatic));
		
		for(IBond bo:mol.bonds())
			bo.setFlag(CDKConstants.ISAROMATIC, bo.getProperty(SMARTSManipulator.Aromatic)==null?false:bo.getProperty(SMARTSManipulator.Aromatic));
	}
}