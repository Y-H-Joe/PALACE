package reactionCenterDetection;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.log4j.Logger;
import org.openscience.cdk.AtomContainer;
import org.openscience.cdk.CDKConstants;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.graph.ConnectivityChecker;
import org.openscience.cdk.inchi.InChIGeneratorFactory;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IAtomContainerSet;
import org.openscience.cdk.interfaces.IBond;
import org.openscience.cdk.interfaces.IBond.Order;
import org.openscience.cdk.smiles.SmilesGenerator;
import org.openscience.cdk.smsd.Isomorphism;
import org.openscience.cdk.smsd.interfaces.Algorithm;

import changes.Change;
import changes.Change.ChangeType;
import constants.Constants;
import constants.StringConstants;
import reactionMapping.ManualMapper;
import reactionMapping.MappedReaction;
import structure.identifiers.InChIGeneratorWrapper;
import tools.Tools;
import tools.MyMath;

/**This class defines a reaction center, as one or two reactants, belonging to a reaction in which each
 * of the reactants contains only atoms that have changed or are of importance in the reaction.<br>
 * The latter implies that eg hetero atoms bonded to a changed carbon atom or atoms whose stereochemistry changes
 * are part of the reaction center.
 * 
 * @author pplehier
 *
 */
public class ReactionCenter {
	
	private static Logger logger	=	Logger.getLogger(ReactionCenter.class);
	private static boolean peroxy	=	false;
	private MappedReaction mappedReaction;
	private String[] smarts;
	private int[][] smartsOrders;
	private IAtomContainer[] reactants;
	private String[] inChIs;
	private List<Change> changes;
	private List<IAtom> centers;
	private int numberOfReactiveCenters;
	private int numberEncountered;
	private boolean cdkFailureNotified;
	private boolean inChIsSet;
	private boolean reversKinetics;
	private String name;
	private ReactionCenter reverse;
	//Limits the number of centers to 17 per reaction center. It is un-thinkable though that 18 or more atoms
	//play a role in a single reaction. 
	private static final String[] centerIDs={"A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"};
	
	protected ReactionCenter(MappedReaction mappedReaction){
		
		//Set hydrogenused property to false for all atoms.
		
		for(int i = 0;	i < mappedReaction.getReactantCount();	i++){
			for(int j = 0;	j < mappedReaction.getReactants()
											  .getAtomContainer(i)
											  .getAtomCount();	j++){
				mappedReaction.getReactants()
							  .getAtomContainer(i)
							  .getAtom(j)
							  .setProperty(StringConstants.HYDROGENUSED, false);
			}
		}
		this.inChIsSet			=	false;
		this.cdkFailureNotified	=	false;
		this.reversKinetics		=	false;
		this.numberEncountered	=	0;
		this.mappedReaction		=	mappedReaction;
		this.changes			=	new ArrayList<Change>();
		this.smartsOrders		=	new int[mappedReaction.getReactantCount()][];
		this.inChIs				=	new String[mappedReaction.getReactantCount()];
		
		//inChIs are initialised to an empty string
		for(int i = 0;	i < inChIs.length;	i++){
			inChIs[i]	=	"";
		}
	}
	
	/**Set the order of the smarts string for reactant[i]. 
	 * The input order works as follows:<br>
	 * order[i] contains the position in the smarts string of an atom with index [i] in the AtomContainer
	 * to which the smarts string corresponds.
	 * 
	 * @param order
	 * @param i
	 */
	protected void setOrdersI(int[] order , int i){
		
		this.smartsOrders[i]	=	order;
	}
	
	/**Set the smarts for the different reactants.
	 * 
	 * @param Smarts
	 */
	protected void setSmarts(String[] Smarts){
	
		this.smarts	=	Smarts;
	}
	
	/**Set the reactants of the reaction center
	 * 
	 * @param Reactants
	 */
	protected void setReactants(IAtomContainer[] reactants){
		
		this.reactants	=	reactants;
	}
	
	/**Set the InChIs of the different reactants of the reaction center. If an InChI cannot be generated for
	 * a reactant (which is possible for aromatic ring sections that do not contain the full ring), the InChI is
	 * left at an empty string.
	 * 
	 * @throws CDKException
	 */
	private void setInChIs() throws CDKException{
		
		if(this.inChIs[0].equals("")){
			InChIGeneratorFactory inchiFact	=	InChIGeneratorFactory.getInstance();
			
			for(int i = 0;	i < this.reactants.length;	i++){
				String inChI	=	null;
				try {
					inChI = inchiFact.getInChIGenerator(this.reactants[i]).getInchi();
				} catch (CDKException e) {}
				
				if(inChI != null){
					this.inChIs[i]	=	inChI;
				}
				//If inchi fails for single carbon atom species, use smiles.
				else if(this.reactants[i].getCAtomCount() == 1){
					this.inChIs[i]	=	SmilesGenerator.generic().create(this.reactants[i]);
				}
			}
		}
		
		this.inChIsSet	=	true;
	}
	
	/**Set the centers of the reaction center. A center is defined as one atom that is effectively changed.
	 * The centers array is a list (across all reactants) of these changed atoms.
	 */
	protected void setCenters(){
		
		List<IAtom> centers	=	new ArrayList<IAtom>();
		
		for(Change ch:this.changes){
			IAtom[] atoms	=	ch.getAtoms();
			for(int i = 0;	i < atoms.length;	i++){
				if(!centers.contains(atoms[i])){
					centers.add(atoms[i]);
				}
			}
		}
		
		this.centers	=	centers;
		
		//Assign center labeling to each of the center atoms
		for(int i = 0;	i < this.centers.size();	i++){
			this.centers.get(i).setProperty(StringConstants.RECIPECENTER,centerIDs[i]);
		}
		
		checkAllAtomsInACenter();
		
		this.numberOfReactiveCenters	=	centers.size();
	}
	
	/**Indicate that this reaction center is the reverse of another existing reaction center and should therefore
	 * be assigned reverse kinetics.
	 */
	public void setReverse() {
		
		this.reversKinetics	=	true;	
	}
	
	/**Give the reactive center a name.
	 * 
	 * @param name
	 */
	public void setName(String name){
		
		this.name	=	name;
	}
	
	/**Get the name of the reactive center
	 * 
	 * @return name
	 */
	public String getName(){
		
		return this.name;
	}
	
	/**Checks whether all atoms of the specified reactants are also in a center. Those that aren't were added because
	 * a specified reactant does not change in the reaction. To comply with the specified bimolecular trait, the 
	 * atoms are added, with symbol Z.
	 */
	private void checkAllAtomsInACenter(){
		
		for(IAtomContainer reactant:reactants){
			for(IAtom atom:reactant.atoms()){
				if(!centers.contains(atom) && atom.getProperty(StringConstants.AVOIDEMPTY) != null){
					centers.add(atom);
					atom.setProperty(StringConstants.RECIPECENTER,"Z");
				}
			}
		}
	}
	/**Retrieve the smarts of all reactants
	 * 
	 * @return smarts
	 */
	public String[] getSmarts(){
		
		return this.smarts;
	}
	
	/**Retrieve ths smarts of one reactant (the one at index)
	 * 
	 * @param index
	 * @return smarts
	 */
	public String getSmartsI(int index){
		
		return this.smarts[index];
	}
	
	/**Retrieve all reactants
	 * 
	 * @return reactants
	 */
	public IAtomContainer[] getReactants(){
		
		return this.reactants;
	}
	
	/**Get the number of reactive centers, ie atoms that are effectively changed.
	 * 
	 * @return number of changed atoms
	 */
	public int getNumberOfReactiveCenters(){
		
		return this.numberOfReactiveCenters;
	}
	
	/**Retrieve all centers
	 * 
	 * @return centers
	 */
	public List<IAtom> getCenters(){
		
		return this.centers;
	}
	
	/**Retrieve all changes
	 * 
	 * @return changes
	 */
	public List<Change> getChanges(){
		
		return this.changes;
	}
	
	/**Get the number of reactants
	 * 
	 * @return number of reactants
	 */
	public int getReactantCount(){
		
		return this.reactants.length;
	}

	/**Retrieve reactant[index]
	 * 
	 * @param index
	 * @return reactant
	 */
	public IAtomContainer getReactant(int index){
		
		return this.reactants[index];
	}

	/**Retrieve the reactant containing the specified atom
	 * 
	 * @param atom
	 * @return reactant
	 */
	public IAtomContainer getReactant(IAtom atom){
		
		for(int i = 0;	i < this.getReactantCount();	i++){
			if(this.reactants[i].contains(atom)){return this.reactants[i];}
		}
		
		return null;
	}

	/**Retrieve the mapped reaction that belongs to this reactive center
	 * 
	 * @return mapped reaction
	 */
	public MappedReaction getReaction(){
		
		return this.mappedReaction;
	}
	
	/**Find whether the reaction should be assigned reverse kinetics.
	 * 
	 * @return reaction has been assigned reverse kinetics.
	 */
	public boolean isReverse(){
		
		return this.reversKinetics;
	}
	
	public ReactionCenter getReverse(){
		
		return this.reverse;
	}
	
	public void setReverseCenter(ReactionCenter reverse){
		
		this.reverse	=	reverse;
	}
	
	/**Get the number of times this reaction family has been encountered
	 * 
	 * @return
	 */
	public int getNumberEncountered(){
		
		return this.numberEncountered;
	}
	
	/**Add one to the number of times this reaction family has been encountered.
	 * 
	 */
	public void encountered(){
		
		this.numberEncountered++;
	}
	/**Returns the indices {reactant, atom} of an atom in the reactants of the reactive center
	 * 
	 * @param atomOfCenter
	 * @return {index of reactant, index of atom in reactant}
	 */
	private int[] getIndices(IAtom atomOfCenter){
		
		int[] indices	=	{-1,-1};
		
		for(int i = 0;	i < this.reactants.length;	i++){
			if(this.reactants[i].contains(atomOfCenter)){
				indices[0]	=	i;
				indices[1]	=	this.reactants[i].getAtomNumber(atomOfCenter);
			}
		}
		
		return indices;
	}

	/**Check whether the reaction is electronically balanced.<br>
	 * Each increase/formation of a bond requires two electrons<br>
	 * Each decrease/breaking of a bond produces two electrons<br>
	 * Each formation of a radical/charge requires one electron<br>
	 * Each removal of a radical/charge produces one electron<br>
	 * The total sum should be zero.
	 * 
	 * Counts as a check  whether all important reactants/products have been listed.
	 */
	private void checkElectronBalance(){
		
		int electrons	=	0;
		
		for(int i = 0;	i<this.changes.size();	i++){
			switch(this.changes.get(i).getChangeType()){
			case INCREASE_BOND: electrons -= 2;	break;
			case DECREASE_BOND: electrons += 2;	break;
			case BREAK_BOND:	electrons += 2;	break;
			case FORM_BOND:		electrons -= 2; break;
			case GAIN_RADICAL:	electrons -= 1; break;
			case LOSE_RADICAL:	electrons += 1; break;
			case GAIN_CHARGE:	break;
			case LOSE_CHARGE:	break;
			case UNSET:							break;
			default:							break;
			}
		}
		
		if(electrons != 0){
			logger.error("The detected reaction mechanism is not in electron balance."+
						 "\nImportant reactants or products may be missing, check input!");
		}
	}
	
	/**Determines the changes to the atoms that are present in this reaction center
	 * and checks the electron balance of the reaction.
	 */
	protected void determineChanges(){
		
		for(int k = 0;	k < this.reactants.length;	k++){
			for(int j = 0;	j < this.reactants[k].getAtomCount();	j++){
				this.determineChanges(this.reactants[k].getAtom(j));
			}
		}
		
		this.checkElectronBalance();
	}

	/**Method perceives WHAT has changed to a specified atom of the ReactionCenter.<br>
	 * Adds the different Changes to the ReactionCenter
	 * 
	 * @param atomOfCenter
	 */
	private void determineChanges(IAtom atomOfCenter){
		
		if(!this.contains(atomOfCenter)){
			logger.fatal("Specified atom not found in reaction center!");
			System.exit(-1);
		}
		
		this.addChanges(atomOfCenter);
		
	}
	
	/**Check whether the reaction centers contains the specified atom
	 * 
	 * @param atom
	 * @return this contains? atom
	 */
	public boolean contains(IAtom atom){
		
		for(int i = 0;	i < this.getReactants().length;	i++){
			for(int j = 0;	j < this.getReactants()[i].getAtomCount();	j++){
				if(this.getReactants()[i].getAtom(j) == atom){return true;}
			}
		}
		
	return false;
	}
	
	/**Checks whether this reaction center already contains a change. <br>
	 * Only for change based on same atoms!<br>
	 * Should not be used for changes generated from a different reaction!
	 * 
	 * @param change
	 * @return
	 */
	public boolean contains(Change change){
		
		for(Change c:this.changes){
			if(c.isEqual(change)){return true;}
		}
		
		return false;
	}
	
	/**Checks whether this reaction center contains the reverse of a specified change.
	 * 
	 * @param change
	 * @return is reverse
	 */
	public boolean containsReverse(Change change){
		
		for(Change c:this.changes){
			if(c.isReverseOf(change)){return true;}
		}
		
		return false;
	}
	
	/**Find an unused hydrogen. This method is required to detect formed bonds with hydrogen as RDT does not
	 * map hydrogen  
	 * 
	 * @return unused hydrogen
	 */
	private IAtom findHydrogen(IAtom atom){
		
		for(int i = 0;	i < this.reactants.length;	i++){
			for(int j = 0;	j < this.reactants[i].getAtomCount();	j++){
				if((boolean)this.reactants[i].getAtom(j)
											 .getProperty(StringConstants.HYDROGENUSED) == false
				   &&
				   this.reactants[i].getAtom(j)
				   					.getSymbol()
				   					.equals("H")
				   &&!this.reactants[i].getAtom(j).equals(atom)){
					this.reactants[i].getAtom(j).setProperty(StringConstants.HYDROGENUSED, true);
					return this.reactants[i].getAtom(j);
				}
			}
		}
		
		return null;
	}
	
	/**Checks whether the specified atom is a center of this reaction center.
	 * An atom can be part of the Reaction Center and not be in the centers list of the Reaction Center.
	 * In that case, the method will return false.
	 * 
	 * @param atom
	 * @return
	 */
	public boolean isInCenter(IAtom atom){
		
		for(IAtom at:this.getCenters()){
			if(at.equals(atom)){
				return true;
			}
		}
		
		return false;
	}
	
	/**Checks if two reaction centers represent the same reaction family. True if:<br>
	 * -Reactive centers are the same<br>
	 * -Changes are the same<br>
	 * 
	 * @param RC
	 * @return this ?is same reaction family as RC
	 */
	public boolean isSameReactionFamily(ReactionCenter RC){
		
		//It is possible that the RC that is checked is null: if RDT failed to produce a mapping, the RC will 
		//be set to null. In such a case, this reaction center should be added!
		if(RC == null || this == null){
			return false;
		}
		
		//If the query RC's InChI's are not set, try to set them. If this fails, return false (cannot determine
		//whether the two RC are equal, so assume they are not
		if(!RC.inChIsSet){
			try{
				RC.setInChIs();
			} catch (CDKException e){
				return false;
			}
		}
		
		try {
			this.setInChIs();
			
		} catch (CDKException e) {
			//If InChI generation fails, add the reaction family! (but only notify failure once per reaction center)
			if(!cdkFailureNotified){
				cdkFailureNotified	=	true;
				logger.warn("InChI generation failed for reaction center "+this.smarts[0]+
							"\nThe corresponding reaction family will be added.");
			}
			return false;
		}

		if(this.hasSameReactants(RC)){
			if(this.hasSameChanges(RC)){
				return true;
			}
		}
		
		return false;
	}
	
	/**Checks if two reaction centers are the reverse of each other. True if:<br>
	 * -Application of the changes of one reaction center results in the other reaction center.
	 * -Changes are inverse of each other
	 * 
	 * @param RC
	 * @return this ?is same reaction family as RC
	 */
	public boolean isReverseReactionFamily(ReactionCenter RC){
		
		//If an incorrect reaction center was detected, the reaction center will be null
		if(RC == null || this == null)
			return false;
		
		//If some other call already set the reverse kinetics, do not do the comparison again.
		if(this.reversKinetics)
			return true;
		
		//If for any change the reverse is not present in the other reaction center, the reactions centers do not
		//represent a pair of reversible reactions, first requirement is having equal number of changes!
		if(this.changes.size() != RC.changes.size())
			return false;
		for(Change change:this.changes)
			if(!RC.containsReverse(change))
				return false;
		
		IAtomContainer reactants=	new AtomContainer();
		for(IAtomContainer react:RC.getReactants()){
			reactants.add(react);
		}
		
		IAtomContainer product	=	new AtomContainer();
		try{
			product	=	reactants.clone();
		}catch(CloneNotSupportedException e){}
		
		for(Change change:RC.changes){
				change.setToApply(reactants);
				change.apply(product);
		}
		
		IAtomContainerSet products	=	ConnectivityChecker.partitionIntoMolecules(product);
		List<String> inChIs	=	new ArrayList<String>();
		
		for(IAtomContainer prod:products.atomContainers()){
			InChIGeneratorWrapper.generate(prod);
			String inChI	=	prod.getProperty(CDKConstants.INCHI);
			if(inChI == null && prod.getCAtomCount() == 1){
				try {
					inChI	=	SmilesGenerator.generic().create(prod);
				} catch (CDKException e) {}
			}
			inChIs.add(inChI);
		}
		
		if(!this.inChIsSet){
			try {
				this.setInChIs();
			} catch (CDKException e) {
				//If InChI generation fails, add the reaction family! (but only notify failure once per reaction center)
				if(!cdkFailureNotified){
					cdkFailureNotified	=	true;
					logger.warn("InChI generation failed for reaction center "+this.smarts[0]);
				}		
			}
		}
		//If one of the reactants of this center is not found, the reaction centers do not represent a pair of 
		//reversible reactions.
		for(IAtomContainer react:this.reactants){
			String inChI	=	react.getProperty(CDKConstants.INCHI);
			if(inChI == null && react.getCAtomCount() == 1){
				try{inChI	=	SmilesGenerator.generic().create(react);}
				catch(CDKException e){}
			}
			if(inChI == null || !Tools.contains(inChIs, inChI))
				return false;
		}
		
		setReverse();
		return true;
	}
	
	/**Check whether the reaction center describes an identity reaction, ie. one in which there are no (net) changes
	 * to any of the reactants.
	 * 
	 * @return changes is empty?
	 */
	public boolean noChanges(){
		
		return this.changes.size()	==	0;
	}
	
	/**If there are radicals present in the reaction , it is safe to assume that they will play some role in
	 * the reaction mechanism. Therefore, if radicals are present, but the reaction center does not contain a change
	 * in which radicals are changed (ie GAIN_RADICAL or LOSE_RADICAL; if the combination break bond and form bond is 
	 * present for the radical center, it is also considered OK), it will be assumed that the mapping failed
	 * to correctly interpret the reaction and should hence be denoted as failed!
	 * 
	 * @return radical changes in reaction
	 */
	public boolean radicalMechanismOK(){
		
		boolean radicalInCenter	=	false;
		boolean noRadicals		=	false;
		
		for(IAtomContainer react:this.reactants){
			IAtomContainer reactant	=	this.mappedReaction.getReactant(this.mappedReaction.getReactantContainerIndex(react.getAtom(0)));
			//If no single electrons present, ok.
			if(reactant.getSingleElectronCount() == 0)
				noRadicals	=	true;
			else
				noRadicals	=	false;
			//if both have at least one single electron, ok, go to activity check.
			if(reactant.getSingleElectronCount() != 0 && react.getSingleElectronCount() != 0)
				radicalInCenter	=	true;
		}
		
		if(noRadicals)
			return true;
		
		//If the radical is not present in the reactive center, the mapping is likely to be incorrect.
		if(!radicalInCenter && !peroxy){
			peroxy	=	true;
			boolean ans	=	ManualMapper.peroxyRadicals(this);
			peroxy	=	false;
			return ans;
		}
		else if(!radicalInCenter)
			return false;
		
		//If the radical is present in the center, check whether it has an active participation, ie. either 
		//the radical is lost, or a combination of breaking and forming a bond takes place.
		for(IAtom center:this.getCenters()){
			if(!center.getSymbol().equals("H")){
				int index	=	this.mappedReaction.getReactantContainerIndex(center);
				IAtomContainer reactant	=	this.mappedReaction.getReactant(index);
				//if its a radical:
				if(reactant.getConnectedSingleElectronsCount(center) != 0){
					
					boolean radicalChange	=	false;
					boolean formedBond		=	false;
					boolean brokenBond		=	false;
					for(Change change:this.getChanges()){
						//IF single electrons present, at least some change to the number of se.
						//In this case, the radical participates in the mechanism, so mapping will most likely be OK
						if(change.contains(center)
								&& 
								(change.getChangeType()	==	ChangeType.GAIN_RADICAL 
								|| 
								change.getChangeType() == ChangeType.LOSE_RADICAL)
								){
							radicalChange	=	true;
							break;
						}
						
						//if both a bond is formed and broken at the radical center, ok too.
						if(change.contains(center) && change.getChangeType() == ChangeType.FORM_BOND){
							formedBond	=	true;
						}
						if(change.contains(center) && change.getChangeType() == ChangeType.BREAK_BOND){
							brokenBond	=	true;
						}							
					}
					if(brokenBond && formedBond)
						radicalChange	=	true;
					
					//if wrong mechanism, check whether can be solved by switching the peroxide mapping
					if(!radicalChange && !peroxy){
						peroxy	=	true;
						boolean ans	=	ManualMapper.peroxyRadicals(this);
						peroxy	=	false;
						return ans;
					}
					else if(!radicalChange)
						return false;
				}
			}
		}
		
		return true;
	}
	
	/**Checks whether this and the specified reaction centers have the same reactants, based on a comparison of
	 * the reactant InChI's.<br>
	 * If the reactant is part of an aromatic ring (and the whole ring is not present as reactive center), InChI
	 * generation will fail and return null (the inChI setter then leaves the InChI at "")<br> 
	 * Currently, this implies that this method will return true if both reactants have an empty InChI (and all
	 * other reactantsare the same)<br>
	 * This is a reasonable approximation. If the reactants should not match, this will be detected later on by the 
	 * isomorphism test to identify the difference in the changes
	 * 
	 * @param RC
	 * @return this ?has same reactants as RC
	 */
	private boolean hasSameReactants(ReactionCenter RC){
		return Tools.haveSameElements(this.inChIs,RC.inChIs);
	}

	
	/**Generate all possible combinations of mappings given a list of list of mappings. The list contains possible 
	 * mappings for each set of reactants:<br>
	 * allMaps.get(0) are all possible mappings of RC.reactants[0] to this.reactants[0] <br>
	 * The returned list contains all combinations of mappings of each reactants:<br>
	 * possibleMappingCombinations.get(0) is {allMaps[0][0],allMaps[1][0],...,allMaps[n][0]}.<br>
	 * possibleMappingCombinations.get(1) is {allMaps[0][0],allMaps[1][0],...,allMaps[n][1]}.<br>
	 * and so on.
	 * @param allMaps
	 * @return all possible combinations of mappings of the reactants
	 */
	private static List<List<Map<IAtom,IAtom>>> createAllPermutations(List<List<Map<IAtom,IAtom>>> allMaps){
		
		int[] maxCount	=	new int[allMaps.size()];
		
		for(int i = 0;	i < allMaps.size();	i++){
			maxCount[i]	=	allMaps.get(i).size();	
		}
		
		int[] currentPermutation	=	new int[allMaps.size()];
		List<List<Map<IAtom,IAtom>>> possibleMappingCombinations	=	new ArrayList<List<Map<IAtom,IAtom>>>();
		boolean stop	=	false;
		
		while(!stop){
			List<Map<IAtom,IAtom>> onePossibleCombination	=	new ArrayList<Map<IAtom,IAtom>>();
			
			for(int i = 0;	i < allMaps.size();	i++){
				onePossibleCombination.add(allMaps.get(i).get(currentPermutation[i]));
			}
			
			possibleMappingCombinations.add(onePossibleCombination);
			
			stop	=	!MyMath.nextPermutation(currentPermutation, maxCount, allMaps.size()-1);
		}
		
		return possibleMappingCombinations;
	}
	
	/**Checks whether this and the specified reaction center have the same changes
	 * </p>
	 * Much more general search than this.contains(Change).<br>
	 * Uses isomorphism search on the reactants of the different ReactionCenters to map the atoms<br>
	 * Due to symmetry of the reactants, it is possible that several mappings are found. Due to the labeling of
	 * the reactants in the reaction center, only one mapping will result in the correct assessment of the
	 * similarity between the two reaction centers.
	 * 
	 * @param RC
	 * @return this ?has same changes RC
	 */
	private boolean hasSameChanges(ReactionCenter RC){
		
		List<List<Map<IAtom,IAtom>>> allMaps	=	new ArrayList<List<Map<IAtom,IAtom>>>();
		
		for(int i = 0;	i < this.reactants.length;	i++){
			for(int j = 0;	j < RC.reactants.length;	j++){
				if(this.inChIs[i].equals(RC.inChIs[j])){
					Isomorphism iso	=	new Isomorphism(Algorithm.CDKMCS,true);
					
					try {
						iso.init(this.reactants[i], RC.reactants[j],false,false);
					} catch (CDKException e) {
						logger.warn("Failed to initiate isomorphism, might incorrectly assume different reaction centers");
						e.printStackTrace();
					}
					
					allMaps.add(iso.getAllAtomMapping());
				}
			}
		}
		
		//Due to symmetry in the reactant fragments, several mappings might be possible. As the atoms are labeled,
		//only one mapping will result in a correct assessment of the equality of the changes.
		List<List<Map<IAtom,IAtom>>> possibleMappingCombinations	=	createAllPermutations(allMaps);
		boolean ansOneMap	=	false;
		
		for(List<Map<IAtom, IAtom>> oneCombination:possibleMappingCombinations){
			
			Map<IAtom,IAtom> map	=	oneCombination.get(0);
			
			for(int i = 1;	i < oneCombination.size();	i++){
				map.putAll(oneCombination.get(i));
			}
			
			ansOneMap	=	this.checkForOneMapping(map, RC);
			
			if(ansOneMap){return ansOneMap;}
			
		}
		
		return false;
	}
	
	/**Check whether the reaction centers are equal given the specified mapping of the atoms of each center <br>
	 * The two changes are considered equal if the mapped atoms of this change are found in a change of the 
	 * specified ReactionCenter and the changeTypes are identical.
	 * 
	 * @param mapping of this.reactants to RC.reactants
	 * @param RC
	 * @return is this equal to RC
	 */
	private boolean checkForOneMapping(Map<IAtom,IAtom> map, ReactionCenter RC){
		//search for corresponding changes in the specified RC, with the specified mapping
		boolean[]found	=	new boolean[this.changes.size()];
			
		for(int i = 0;	i < this.changes.size();	i++){
			if(this.changes.get(i).getAtomCount() == 1){
				IAtom atom	=	this.changes.get(i).getAtom(0);
				
				for(int j = 0;	j < RC.changes.size();	j++){	
					if(RC.changes.get(j).contains(map.get(atom))
					   &&
					   RC.changes.get(j).getChangeType() == this.changes.get(i).getChangeType()){
						found[i]	=	true;
					}
				}
			}
			
			else if(this.changes.get(i).getAtomCount() == 2){
				IAtom atom1	=	this.changes.get(i).getAtom(0);
				IAtom atom2	=	this.changes.get(i).getAtom(1);

				for(int j = 0;	j < RC.changes.size();	j++){
					if(RC.changes.get(j).contains(map.get(atom1))
					   &&
					   RC.changes.get(j).contains(map.get(atom2))
					   &&
					   RC.changes.get(j).getChangeType() == this.changes.get(i).getChangeType()){
						found[i]	=	true;
					}
				}
			}
		}
		//Same reaction center if a corresponding change was found for EACH change in this
		boolean ans	=	true;
		
		for(int i = 0;i < found.length;	i++){
			ans	&=	found[i];
		}
		
		return ans;
	}
	
	/**Add the change corresponding to the breaking of a bond with hydrogen.to the specified atom, which is
	 * assumed to be part of the reactive center
	 * </p>
	 * Note: a change is only added if it is not yet present in this ReactionCenter
	 * 
	 * @param atomOfCenter
	 */
	private void addBrokenHydrogenBond(IAtom atomOfCenter){
		
		int[]indices		=	this.getIndices(atomOfCenter);
		IAtom connectedAtom	=	this.reactants[indices[0]].getConnectedBondsList(atomOfCenter)
														  .get(0)
														  .getConnectedAtom(atomOfCenter);
		Change aChange		=	new Change(Constants.BREAK_BOND,atomOfCenter,connectedAtom);
				
		this.addChange(aChange);

	}
	
	/**Add a change corresponding to the formation of a bond with hydrogen to the specified atom, which is
	 * assumed to be part of the reactive center
	 * </p>
	 * Note: a change is only added if it is not yet present in this ReactionCenter
	 * 
	 * @param atomOfCenter
	 */
	private void addFormedHydrogenBond(IAtom atomOfCenter){
		
		int hydrChange	=	atomOfCenter.getProperty(StringConstants.HYDROGENDIFFERENCE);
		int hydrAdded	=	0;
		//The hydrogen bonds that are assessed here are those that form implicit hydrogens in the products, therefore, a check is made 
		//that the product atom indeed has at least as many implicit hydrogen atoms as the reactant atom takes up in the reaction
		while(mappedReaction.getMappedProductAtom(atomOfCenter).getImplicitHydrogenCount() >= -hydrChange && hydrAdded < -hydrChange){
			Change aChange	=	new Change(Constants.FORM_BOND,atomOfCenter,this.findHydrogen(atomOfCenter));
			this.addChange(aChange);
			hydrAdded++;
		}
		/*
		if(hydrChange < 0 && (mappedReaction.getMappedProductAtom(atomOfCenter).getImplicitHydrogenCount()) != 0){
			Change aChange	=	new Change(Constants.FORM_BOND,atomOfCenter,this.findHydrogen(atomOfCenter));
			this.addChange(aChange);
		}*/
		
	}
	
	/**Add all changes corresponding to breaking or changing the order of the reactant bonds to the specified 
	 * atom, which is assumed to be part of the reactive center.
	 * </p>
	 * Note: a change is only added if it is not yet present in this ReactionCenter
	 * 
	 * @param atomOfCenter
	 */
	private void addChangedReactantBonds(IAtom atomOfCenter){
		
		int containerIndex	=	mappedReaction.getReactantContainerIndex(atomOfCenter);
		//Start Checking bonds for changes: if both atoms of a bond are flagged as "Bond changed" 
		//the bond between them will have been changed (different order or broken)
		List<IBond> atomBonds	=	this.reactants[containerIndex].getConnectedBondsList(atomOfCenter);
		//Start checking for bonds that have been broken/changed in order (i/e bonds that are present 				
		//in reactants
		for(IBond bond:atomBonds){
			//If one of the two bond atoms' mapping property is null: it is a hydrogen that has been added and 
			//of which the bond change (broken) has been dealt with in the first if statement
			if((boolean)bond.getAtom(0).getProperty(StringConstants.BONDCHANGE)
			   &&
			   (boolean)bond.getAtom(1).getProperty(StringConstants.BONDCHANGE)
			   &&
			   bond.getAtom(0).getProperty(StringConstants.MAPPINGPROPERTY) != null
			   &&
			   bond.getAtom(1).getProperty(StringConstants.MAPPINGPROPERTY) != null){
				IAtom mapped0	=	mappedReaction.getMappedProductAtom(bond.getAtom(0));
				IAtom mapped1	=	mappedReaction.getMappedProductAtom(bond.getAtom(1));
				
				int cont_ind0	=	mappedReaction.getContainerIndexMappedAtom(bond.getAtom(0));
				int cont_ind1	=	mappedReaction.getContainerIndexMappedAtom(bond.getAtom(1));
				
				//if the mapped atoms are in different containers, the bond has been broken.
				if(cont_ind0 != cont_ind1){
					//if the initial order was triple, split into decrease + decrease + break.
					if(bond.getOrder() == Order.TRIPLE){
						Change bChange	=	new Change(Constants.DECREASE_BOND, bond.getAtom(0), bond.getAtom(1));
						boolean added	=	this.addChange(bChange);
						//only override when previous was added (ie change hasn't been passed yet. Otherwise,
						//will add the decrease bond when checking the atoms inversely.
						if(added){
							Change cChange	=	new Change(Constants.DECREASE_BOND, bond.getAtom(0), bond.getAtom(1));
							this.addChangeOverride(cChange);
						}
					}
					//if the initial bond was double, the transformation should be split into decrease order + break
					if(bond.getOrder() == Order.DOUBLE){
						Change bChange	=	new Change(Constants.DECREASE_BOND, bond.getAtom(0), bond.getAtom(1));
						this.addChange(bChange);
					}
					
					Change aChange	=	new Change(Constants.BREAK_BOND, bond.getAtom(0), bond.getAtom(1));
					this.addChange(aChange);
				}
					
				else{
					int bondP	=	this.mappedReaction.getProducts()
													   .getAtomContainer(cont_ind0)
													   .getBondNumber(mapped0, mapped1);
					
					//if there is no longer a bond between the mapped atoms, the bond has been broken.
					if(bondP == -1){
						//if the initial order was triple, split into decrease + decrease + break.
						if(bond.getOrder() == Order.TRIPLE){
							Change bChange	=	new Change(Constants.DECREASE_BOND, bond.getAtom(0), bond.getAtom(1));
							boolean added	=	this.addChange(bChange);
							//only override when previous was added (ie change hasn't been passed yet. Otherwise,
							//will add the decrease bond when checking the atoms inversely.
							if(added){
								Change cChange	=	new Change(Constants.DECREASE_BOND, bond.getAtom(0), bond.getAtom(1));
								this.addChangeOverride(cChange);
							}
						}
						//if the initial bond was double, the transformation should be split into decrease order + break
						if(bond.getOrder() == Order.DOUBLE){
							Change bChange	=	new Change(Constants.DECREASE_BOND, bond.getAtom(0), bond.getAtom(1));
							this.addChange(bChange);
						}	

						Change aChange	=	new Change(Constants.BREAK_BOND, bond.getAtom(0), bond.getAtom(1));
						this.addChange(aChange);
					}
					
					//if there is still a bond between the mapped atoms, the bond order has changed.
					else{
						int orderChange	=	bond.getOrder().compareTo(
							this.mappedReaction.getProducts()
											   .getAtomContainer(cont_ind0)
											   .getBond(bondP)
											   .getOrder());
						//if the bond order of the product is higher, the bond order has increased
						if(orderChange < 0){
							Change aChange	=	new Change(Constants.INCREASE_BOND, bond.getAtom(0), bond.getAtom(1));
							this.addChange(aChange);
						}
						//if the bond order of the product is lower, the bond order has decreased
						else if(orderChange > 0){
							Change aChange	=	new Change(Constants.DECREASE_BOND, bond.getAtom(0), bond.getAtom(1));
							this.addChange(aChange);
						}
					}
				}
			}
		}
	}
	
	/**Add all changes corresponding to the formation of new bonds to the specified atom, which is assumed to be 
	 * part of the reaction center. These bonds cannot be detected by the method
	 * addFormedReactantBonds(atomOfCenter) as formed bonds are not present in the reactants.
	 * </p>
	 * Note: a change is only added if it is not yet present in this ReactionCenter
	 * 
	 * @param atomOfCenter
	 */
	private void addFormedProductBonds(IAtom atomOfCenter){
		
		//start checking product bonds: necessary to determine which bonds have been FORMED
		List<IBond> atomProdBonds	=	this.mappedReaction.getProducts()
														   .getAtomContainer(mappedReaction.getContainerIndexMappedAtom(atomOfCenter))
														   .getConnectedBondsList(mappedReaction.getMappedProductAtom(atomOfCenter));
		
		for(IBond bond:atomProdBonds){
			IAtom mapped0	=	mappedReaction.getMappedReactantAtom(bond.getAtom(0));
			IAtom mapped1	=	mappedReaction.getMappedReactantAtom(bond.getAtom(1));
			int bondR		=	this.mappedReaction.getReactants()
												   .getAtomContainer(mappedReaction.getReactantContainerIndex(atomOfCenter))
												   .getBondNumber(mapped0, mapped1);
			//if bond is not found (bondR=-1) a bond has been formed
			if(bondR == -1){
				Change aChange	=	new Change(Constants.FORM_BOND,mapped0,mapped1);
				this.addChange(aChange);
				//if the bond order is double, the change can be split into formation + order increase
				if(bond.getOrder() == Order.DOUBLE){
					Change bChange	=	new Change(Constants.INCREASE_BOND,mapped0,mapped1);
					this.addChange(bChange);
				}
				//if the bond order is triple, split into formation + increase + increase (exceptional, but
				//required for [CH2] disproportionation
				if(bond.getOrder() == Order.TRIPLE){
					Change bChange	=	new Change(Constants.INCREASE_BOND,mapped0,mapped1);
					boolean added	=	this.addChange(bChange);
					//only override existence check if the previous increase was added (i.e the change not yet passed).
					//otherwise when checking the atoms in the inversed oder: will add unnecessary increase bond.
					if(added){
						Change cChange	=	new Change(Constants.INCREASE_BOND,mapped0,mapped1);
						this.addChangeOverride(cChange);
					}
				}
			}
		}
	}
	
	/**Add the change in single electrons of the specified atom, which is assumed to be part of the reactive
	 * center.
	 * </p>
	 * Note: a change is only added if it is not yet present in this ReactionCenter
	 * TODO: get radicals ok to work with multiple
	 * @param atomOfCenter
	 */
	private void addChangedSingleElectrons(IAtom atomOfCenter){
		
		int containerIndex	=	mappedReaction.getReactantContainerIndex(atomOfCenter);
		int SEreact			=	this.reactants[containerIndex].getConnectedSingleElectronsCount(atomOfCenter);
		int SEprod			=	this.mappedReaction.getProducts()
												   .getAtomContainer(mappedReaction.getContainerIndexMappedAtom(atomOfCenter))
												   .getConnectedSingleElectronsCount(mappedReaction.getMappedProductAtom(atomOfCenter));
		
		//If 1 more single electron in product: transformation is GAIN RADICAL
		if(SEprod - SEreact == 1){
			Change aChange	=	new Change(Constants.GAIN_RADICAL,atomOfCenter);
			this.addChange(aChange);
		//If 1 fewer single electron in product: transformation is LOSE RADICAL
		}else if(SEreact - SEprod == 1){
			Change aChange	=	new Change(Constants.LOSE_RADICAL,atomOfCenter);
			this.addChange(aChange);
		//if difference is two, lose one twice!
		}else if(SEreact - SEprod == 2){
			Change aChange	=	new Change(Constants.LOSE_RADICAL,atomOfCenter);
			Change bChange	=	new Change(Constants.LOSE_RADICAL,atomOfCenter);
			this.addChangeOverride(aChange);
			this.addChangeOverride(bChange);
		//if difference is - two, add one twice.
		}else if(SEreact - SEprod == -2){
			Change aChange	=	new Change(Constants.GAIN_RADICAL,atomOfCenter);
			Change bChange	=	new Change(Constants.GAIN_RADICAL,atomOfCenter);
			this.addChangeOverride(aChange);
			this.addChangeOverride(bChange);
			//if difference is three, lose one thrice!
		}else if(SEreact - SEprod == 3){
			Change aChange	=	new Change(Constants.LOSE_RADICAL,atomOfCenter);
			Change bChange	=	new Change(Constants.LOSE_RADICAL,atomOfCenter);
			Change cChange	=	new Change(Constants.LOSE_RADICAL,atomOfCenter);
			this.addChangeOverride(aChange);
			this.addChangeOverride(bChange);
			this.addChangeOverride(cChange);
			//if difference is - three, add one thrice.
		}else if(SEreact - SEprod == -3){
			Change aChange	=	new Change(Constants.GAIN_RADICAL,atomOfCenter);
			Change bChange	=	new Change(Constants.GAIN_RADICAL,atomOfCenter);
			Change cChange	=	new Change(Constants.LOSE_RADICAL,atomOfCenter);
			this.addChangeOverride(aChange);
			this.addChangeOverride(bChange);
			this.addChangeOverride(cChange);			
		}else if(SEreact - SEprod == 0){}
		
		else{
			logger.fatal("It is not possible to gain or loose more than two radicals in a single reaction");
			System.exit(-1);
		}
	}
	
	/**Add the change in charge on the specified atom, which is assumed to be part of the reactive
	 * center.
	 * </p>
	 * Note: a change is only added if it is not yet present in this ReactionCenter
	 * 
	 * @param atomOfCenter
	 */
	private void addChangedCharges(IAtom atomOfCenter){
		
		int Creact	=	atomOfCenter.getFormalCharge();
		int Cprod	=	mappedReaction.getMappedProductAtom(atomOfCenter).getFormalCharge();
		
		//If 1 more charge in product: transformation is GAIN CHARGE
		if(Cprod - Creact == 1){
			Change aChange	=	new Change(Constants.GAIN_CHARGE,atomOfCenter);
			this.addChange(aChange);
		//If 1 fewer charge in product: transformation is LOSE CHARGE
		}else if(Creact - Cprod == 1){
			Change aChange	=	new Change(Constants.LOSE_CHARGE,atomOfCenter);
			this.addChange(aChange);
		//If 2 fewer charges in the product: twice lose charge
		}else if(Creact - Cprod == 2){
			Change aChange	=	new Change(Constants.LOSE_CHARGE,atomOfCenter);
			Change bChange	=	new Change(Constants.LOSE_CHARGE,atomOfCenter);
			this.addChangeOverride(aChange);
			this.addChangeOverride(bChange);
		//If 2 more charges in the product: twice gain charge
		}else if(Creact - Cprod == -2){
			Change aChange	=	new Change(Constants.GAIN_CHARGE,atomOfCenter);
			Change bChange	=	new Change(Constants.GAIN_CHARGE,atomOfCenter);
			this.addChangeOverride(aChange);
			this.addChangeOverride(bChange);
			
		}else if(Creact - Cprod == 0){}
		
		else{
			logger.fatal("It is not possible to gain or lose more than two charges in a single reaction");
			System.exit(-1);
		}
	}
	
	/**Add all changes to the specified atom, which is assumed to be part of the reactive center
	 * 
	 * @param atomOfCenter
	 */
	private void addChanges(IAtom atomOfCenter){
		
		//Due to the method, if hydrogen is present without mapping, it indicates that the bond with
		//hydrogen is broken
		if(atomOfCenter.getSymbol().equals("H")
		   &&
		   atomOfCenter.getProperty(StringConstants.MAPPINGPROPERTY) == null){
			this.addBrokenHydrogenBond(atomOfCenter);
		}
		
		else{
				
			//Check whether hydrogen count has increased. This implies that a new bond with hydrogen has been
			//formed, with a hydrogen of which the bond has been broken
			this.addFormedHydrogenBond(atomOfCenter);
			
			//Start checking all bonds
			this.addChangedReactantBonds(atomOfCenter);
			this.addFormedProductBonds(atomOfCenter);
			//end checking all bonds: now all changes to bonds should have been added
			
			//start checking changes to radicals
			this.addChangedSingleElectrons(atomOfCenter);
			//end checking changes to radicals
			
			//start checking changes to charges
			this.addChangedCharges(atomOfCenter);
			//end checking changes to charges
		}
	}

	/**Adds a change to the reactive center if that change is not yet present.
	 * 
	 * @param change
	 */
	private boolean addChange(Change change){
		
		boolean exists	=	this.contains(change);
		if(!exists){
			this.changes.add(change);
		}
		
		return !exists;
	}
	
	/**Adds a change to the reactive center, no matter what!
	 * 
	 * @param change
	 */
	private void addChangeOverride(Change change){
		
		this.changes.add(change);
	}

	public void changeTo(ReactionCenter newCenter) {
		this.centers				=	newCenter.centers;
		this.cdkFailureNotified		=	newCenter.cdkFailureNotified;
		this.changes				=	newCenter.changes;
		this.inChIs					=	newCenter.inChIs;
		this.inChIsSet				=	newCenter.inChIsSet;
		this.reactants				=	newCenter.reactants;
		this.mappedReaction			=	newCenter.mappedReaction;
		this.numberOfReactiveCenters=	newCenter.numberOfReactiveCenters;
		this.reversKinetics			=	newCenter.reversKinetics;
		this.smarts					=	newCenter.smarts;
		this.smartsOrders			=	newCenter.smartsOrders;
		
	}
	
	public void setOldHydrogenCounts(){
		
		for(IAtom atom:this.mappedReaction.getReactantAtoms()){
			if(atom.getProperty(StringConstants.OLDHydrogenCount) != null)
				atom.setImplicitHydrogenCount(atom.getProperty(StringConstants.OLDHydrogenCount));
		}
		for(IAtom atom:this.mappedReaction.getProductAtoms()){
			if(atom.getProperty(StringConstants.OLDHydrogenCount) != null)
				atom.setImplicitHydrogenCount(atom.getProperty(StringConstants.OLDHydrogenCount));
		}
	}
}