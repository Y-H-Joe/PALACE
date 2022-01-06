package manipulators;

import java.util.ArrayList;
import java.util.List;

import org.openscience.cdk.Bond;
import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IBond;
import org.openscience.cdk.interfaces.IBond.Order;
import org.openscience.cdk.smiles.SmilesParser;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;

import constants.StringConstants;
import reactionCenterDetection.ReactionCenterDetector;
import reactionMapping.MappedReaction;

/**This class generates a randomly related reaction to a specified mapped reaction.
 * For each reactant in the reaction, the number of hydrogens that do not participate in the reaction are
 * determined. Each of these hydrogens is replaced by a random molecular fragment. The probability of replacing
 * the hydrogen decreases with each replacement.
 * 
 * @author pplehier
 *
 */
public class FragmentAdder {

	private static final int maxDoubleBond	=	2;
	private static final int maxTotalC		=	10;
	private static final int maxTotalO		=	3;
	private static final String carbon		=	"C";
	private static final String oxygen		=	"O";
	private static final String hydrogen	=	"H";
	private MappedReaction mappedReaction;
	private static List<IAtomContainer> namedFragments;
	private boolean noRepeats;
	
	public static int maxTotalC(){
		
		return maxTotalC;
	}
	/**Construct an adder and set the predefined molecular fragments.
	 * 
	 * @param mappedReaction
	 */
	public FragmentAdder(MappedReaction mappedReaction){
		
		ReactionCenterDetector rcd	=	new ReactionCenterDetector(mappedReaction);
		this.mappedReaction			=	rcd.configureReaction();
		
		createNamedFragments();
	}
	
	/**For each reactant atom, add random fragments.
	 * 
	 */
	public void addFragments(){

		boolean small	=	true;//as soon as one of the reactants is too large: not ok
		
		for(int i = 0;	i < this.mappedReaction.getReactantCount();	i++){
				small	=	small && (this.mappedReaction.getReactant(i).getCAtomCount() < maxTotalC
											 ||
											 this.mappedReaction.getReactant(i).getOAtomCount() < maxTotalO);
		}
		
		if(small){
			noRepeats	=	false;
			
			for(IAtom atom:this.mappedReaction.getReactantAtoms())
				addFragments(atom);
		}
		else{
			noRepeats	=	true;
		}
	}
	
	public boolean largeMolecule(){
		
		return noRepeats;
	}
	
	private void addFragments(IAtom atom){
		
		int availableHydrogens	=	availableImplicitHydrogens(atom);
		double addingProbability=	0.9;
		double decrement		=	3;
		
		while(availableHydrogens > 0){
			double add	=	Math.random();
			
			if(add < addingProbability)
				try {
					addAFragment(atom);
				} catch (CloneNotSupportedException e) {
					e.printStackTrace();
				}
			
			addingProbability	=	addingProbability / decrement;
			availableHydrogens--;
		}
	}
	
	/**Create a list containing some standard (non-linear) fragments.
	 * 
	 */
	private static void createNamedFragments(){
		
		namedFragments		=	new ArrayList<IAtomContainer>();		
		//Phenyl with explicit double and single for InChI generation
		String phenyl		=	"C1=CC=CC=C1";
		String vinyl		=	"C=C";
		String tbutyl		=	"C(C)(C)C";
		String isobutyl		=	"CC(C)C";
		String isobutenyl	=	"CC(=C)C";
		String allyl		=	"CC=C";
		String sallyl		=	"C(=C)C";
		String isopropyl	=	"C(C)C";
		String cyclopentyl	=	"C1CCCC1";
		String acetyl		=	"C(=O)C";
		String acetate		=	"CC(=O)O";
		String dme			=	"COC";
		
		String[] fragments	=	{phenyl, vinyl, tbutyl, isobutyl, isobutenyl, allyl, sallyl, isopropyl, cyclopentyl, acetyl, acetate, dme};
		SmilesParser sp		=	new SmilesParser(DefaultChemObjectBuilder.getInstance());
		
		for(String fragment:fragments){
			try {
				IAtomContainer frag	=	sp.parseSmiles(fragment);
				namedFragments.add(frag);
			} catch (CDKException e) {
			}
		}
	}
	
	/**Add a random-ish fragment to the specified atom
	 * 
	 * @param atom
	 * @throws CloneNotSupportedException 
	 */
	private void addAFragment(IAtom atom) throws CloneNotSupportedException{
		
		IAtomContainer fragment	=	pickAFragment();
		IAtomContainer reactant	=	this.mappedReaction.getReactant(this.mappedReaction.getReactantContainerIndex(atom));
		IAtomContainer product	=	this.mappedReaction.getProduct(this.mappedReaction.getContainerIndexMappedAtom(atom));
		int reactDoubleBond		=	reactant.getDoubleBondCount(false);
		
		//Choose a new fragment if the specified conditions are not met.
		while((atom.getSymbol().equals(oxygen) 
			  && 
			  fragment.getAtomCount() != 0 
			  && 
			  fragment.getAtom(0).getSymbol().equals(oxygen))
			  ||
			  (reactDoubleBond >= maxDoubleBond
			  &&
			  fragment.getDoubleBondCount(false) != 0)
			  ){
			fragment	=	pickAFragment();
		}

		boolean searching		=	true;
		int counter				=	0;
		IAtom mappedAtom		=	this.mappedReaction.getMappedProductAtom(atom);
		int totalC				=	fragment.getCAtomCount() + reactant.getCAtomCount();
		int totalO				=	fragment.getOAtomCount() + reactant.getOAtomCount();
		int doubleBonds			=	reactant.getDoubleBondCount(false) + fragment.getDoubleBondCount(false);
		
		if(totalC < maxTotalC 
		   && 
		   totalO < maxTotalO 
		   && 
		   (doubleBonds < maxDoubleBond || fragment.getDoubleBondCount(false) == 0)){
			if(fragment.getAtomCount() == 0){}
			else{
				while(searching){
					if(fragment.getAtom(counter).getImplicitHydrogenCount() != 0){
						searching	=	false;
						break;
					}
				}
				
				try {
					add(atom, reactant, mappedAtom, product, fragment, counter);

				} catch (CDKException e) {
					e.printStackTrace();
				}
				
				counter++;
			}
		}
	}
	
	private void add(IAtom atom, IAtomContainer reactant,
					 IAtom mappedAtom, IAtomContainer product, 
					 IAtomContainer fragment, int counter) throws CloneNotSupportedException, CDKException{

		IAtomContainer mapFrag	=	fragment.clone();
		IAtom connector			=	fragment.getAtom(counter);
		IBond connectingBond	=	new Bond(atom, connector);
		IAtom mappedConnector	=	mapFrag.getAtom(counter);
		IBond mappedBond		=	new Bond(mappedAtom,mappedConnector);

		connector.setImplicitHydrogenCount(connector.getImplicitHydrogenCount() - 1);
		mappedConnector.setImplicitHydrogenCount(mappedConnector.getImplicitHydrogenCount() - 1);
		atom.setImplicitHydrogenCount(atom.getImplicitHydrogenCount() - 1);
		mappedAtom.setImplicitHydrogenCount(mappedAtom.getImplicitHydrogenCount() - 1);


		reactant.add(fragment);
		reactant.addBond(connectingBond);

		product.add(mapFrag);
		product.addBond(mappedBond);

		AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(reactant);
		AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(product);
		
		(new RadicalHandler()).correctMultipleRadicals(reactant);
		(new RadicalHandler()).correctMultipleRadicals(product);
	}
	private int availableImplicitHydrogens(IAtom atom){
		
		if(atom.getSymbol().equals(hydrogen)){return 0;}
		else{
			int explicitHydrogenCount	=	0;
			int implicitHydrogenCount	=	atom.getImplicitHydrogenCount();
			int requiredInReaction		=	atom.getProperty(StringConstants.HYDROGENDIFFERENCE);
			List<IAtom> neighbours		=	this.mappedReaction.getReactant(this.mappedReaction.getReactantContainerIndex(atom))
															   .getConnectedAtomsList(atom);

			for(IAtom neighbour:neighbours){
				if(neighbour.getSymbol().equals(hydrogen))
					explicitHydrogenCount++;
			}
	
			return implicitHydrogenCount - ((requiredInReaction - explicitHydrogenCount) >= 0?
												(requiredInReaction - explicitHydrogenCount) :
													0);
		}
	}
	
	private static IAtomContainer pickAFragment() throws CloneNotSupportedException{
		
		double choice	=	Math.random();
		
		if(choice > 0.1){
			return makeRandomFragment();
		}
		else{
			return pickFromList();
		}
	}
	
	private static IAtomContainer pickFromList() throws CloneNotSupportedException{
		
		int choice	=	(int)Math.floor(namedFragments.size()*(Math.random()));
		
		return namedFragments.get(choice).clone();
	}
	
	private static IAtomContainer makeRandomFragment(){
		
		double xC			=	Math.random()*10;
		double xO			=	Math.random()*10;
		int doubleBond		=	0;
		int numberOfCarbon	=	(int)Math.floor((672.0/25.0)/(xC+16.0/5.0)-7.0/5.0);
		int numberOfOxygen	=	(int)Math.floor((1.875)/(xO+0.75)+0.5);
		List<String> symbol	=	new ArrayList<String>();
		orderRandom(symbol, numberOfCarbon, numberOfOxygen);
		List<IAtom> atoms	=	new ArrayList<IAtom>();
		List<IBond> bonds	=	new ArrayList<IBond>();
		
		for(String name:symbol){
			IAtom atom	=	DefaultChemObjectBuilder.getInstance().newInstance(IAtom.class, name);
			
			if(name.equals(carbon))
				atom.setImplicitHydrogenCount(4);
			else if(name.equalsIgnoreCase(oxygen))
				atom.setImplicitHydrogenCount(2);
			else if(name.equals(hydrogen))
				atom.setImplicitHydrogenCount(0);
			
			atoms.add(atom);
		}
		
		for(int i = 1;	i < atoms.size();	i++){
			if(symbol.get(i).equals(carbon) && symbol.get(i-1).equals(carbon)){
				double bond	=	Math.random();
				if(bond < 0.75	|| doubleBond >= maxDoubleBond){
					bonds.add(new Bond(atoms.get(i),atoms.get(i-1),Order.SINGLE));
					atoms.get(i).setImplicitHydrogenCount(atoms.get(i).getImplicitHydrogenCount() - 1);
					atoms.get(i-1).setImplicitHydrogenCount(atoms.get(i-1).getImplicitHydrogenCount() - 1);
				}
				else{
					bonds.add(new Bond(atoms.get(i),atoms.get(i-1),Order.DOUBLE));
					atoms.get(i).setImplicitHydrogenCount(atoms.get(i).getImplicitHydrogenCount() - 2);
					atoms.get(i-1).setImplicitHydrogenCount(atoms.get(i-1).getImplicitHydrogenCount() - 2);
					doubleBond++;
				}
			}
			else if(symbol.get(i).equals(oxygen) && i == atoms.size() - 1 && !symbol.get(i-1).equals(oxygen)){
				double bond	=	Math.random();
				if(bond < 0.75 || doubleBond >= maxDoubleBond){
					bonds.add(new Bond(atoms.get(i),atoms.get(i-1),Order.SINGLE));
					atoms.get(i).setImplicitHydrogenCount(atoms.get(i).getImplicitHydrogenCount() - 1);
					atoms.get(i-1).setImplicitHydrogenCount(atoms.get(i-1).getImplicitHydrogenCount() - 1);
				}
				else{
					bonds.add(new Bond(atoms.get(i),atoms.get(i-1),Order.DOUBLE));
					atoms.get(i).setImplicitHydrogenCount(atoms.get(i).getImplicitHydrogenCount() - 2);
					atoms.get(i-1).setImplicitHydrogenCount(atoms.get(i-1).getImplicitHydrogenCount() - 2);
					doubleBond++;
				}
			}
			else{
				bonds.add(new Bond(atoms.get(i),atoms.get(i-1),Order.SINGLE));
				atoms.get(i).setImplicitHydrogenCount(atoms.get(i).getImplicitHydrogenCount() - 1);
				atoms.get(i-1).setImplicitHydrogenCount(atoms.get(i-1).getImplicitHydrogenCount() - 1);
			}
		}
		IAtomContainer fragment	=	DefaultChemObjectBuilder.getInstance().newInstance(IAtomContainer.class);
		
		for(IAtom atom:atoms){
			fragment.addAtom(atom);
		}
		for(IBond bond:bonds){
			fragment.addBond(bond);
		}
		
		try {
			AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(fragment);
		} catch (CDKException e) {
			e.printStackTrace();
		}
		

		return fragment;
	}
	
	private static void orderRandom(List<String> symbols, double nC, double nO){
		
		double total	=	nC + nO;
		double choice	=	Math.random();
		
		if(total == 0){
			return;
		}
		else if(choice < nC / total){
			symbols.add(carbon);
			orderRandom(symbols, nC - 1, nO);
		}
		else{
			symbols.add(oxygen);
			orderRandom(symbols, nC, nO - 1);
		}
	}
}