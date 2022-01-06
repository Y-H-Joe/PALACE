package changes;

import org.apache.log4j.Logger;
import org.openscience.cdk.Atom;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IBond.Order;

import constants.Constants;

/**This class defines a change to a reaction center, using the keywords from Genesys
 * 
 * @author pplehier
 *
 */
public class Change {
	
	private static Logger logger	=	Logger.getLogger(Change.class);
	private ChangeType changeType;
	private IAtom[] atoms;
	private int[] applicationIndices;
	
	public enum ChangeType{
		
		DECREASE_BOND,
		INCREASE_BOND,
		BREAK_BOND,
		FORM_BOND,
		GAIN_RADICAL,
		LOSE_RADICAL,
		GAIN_CHARGE,
		LOSE_CHARGE,
		UNSET;
		
		private ChangeType(){		
		}
		
		private boolean isInverseOf(ChangeType type){
			switch(this){
			case DECREASE_BOND: return type == INCREASE_BOND;
			case INCREASE_BOND: return type == DECREASE_BOND;
			case FORM_BOND:		return type == BREAK_BOND;
			case BREAK_BOND:	return type == FORM_BOND;
			case GAIN_RADICAL:	return type == LOSE_RADICAL;
			case LOSE_RADICAL:	return type == GAIN_RADICAL;
			case GAIN_CHARGE:	return type == LOSE_CHARGE;
			case LOSE_CHARGE:	return type == GAIN_CHARGE;
			default:			return false;
			}
		}
	}
	
	/**Create new change that affects 2 atoms<br>
	 * Can only be: INCREASE_BOND, DECREASE_BOND, FORM_BOND, BREAK_BOND or UNSET
	 * 
	 * @param type
	 * @param atom1
	 * @param atom2
	 */
	public Change(int type,IAtom atom1, IAtom atom2){
		
		if(type >= 4 && type != Constants.UNSET){
			logger.fatal("Cannot create change type "+ChangeType.values()[type]+" with two atoms. Exiting ...");
			System.exit(-1);
		}
		
		this.changeType		=	ChangeType.values()[type];
		IAtom[] newAtoms	=	{atom1,atom2};
		this.atoms			=	newAtoms;
	}
	
	/**Create new change that affects 1 atom<br>
	 * Can only be: GAIN_RADICAL, LOSE_RADICAL, GAIN_CHARGE, LOSE_CHARGE or UNSET
	 * 
	 * @param type
	 * @param atom1
	 */
	public Change(int type, IAtom atom1){
		
		if(type < 4){
			logger.fatal("Cannot create chage type "+ChangeType.values()[type]+" with only one atom. Exiting ...");
			System.exit(-1);
		}
		
		this.changeType		=	ChangeType.values()[type];
		IAtom[] newAtoms	=	{atom1};
		this.atoms			=	newAtoms;
	}
	
	/**Set the change type
	 */
	public void setChangeType(int type){
		
		this.changeType	=	ChangeType.values()[type];
	}
	
	/**Retrieve the change type
	 * 
	 * @return change type
	 */
	public ChangeType getChangeType(){
		
		return this.changeType;
	}

	/**Set the atoms of the change.<br>
	 * The number of atoms should be one or two
	 * 
	 * @param atoms
	 */
	public void setAtoms(IAtom[] atoms){
		
		if(atoms.length != 1 && atoms.length != 2){
			logger.fatal("A change should have at least one atom and at most two. Exiting ...");
			System.exit(-1);
		}
		
		this.atoms	=	atoms;
	}
	
	/**Add an atom to the current change, but cannot add more than two atoms.
	 * 
	 * @param atom
	 */
	public void addAtom(IAtom atom){
		
		if(this.atoms.length >= 2){
			logger.fatal("A change cannot have more than two atoms. Exiting ...");
			System.exit(-1);
		}
		
		IAtom[] newAtoms	=	new Atom[this.atoms.length+1];
		
		for(int i = 0;	i < this.atoms.length;	i++){
			newAtoms[i]	=	this.atoms[i];
		}
		
		newAtoms[this.atoms.length]	=	atom;
		this.atoms	=	newAtoms;
	}
	
	/**Retrieve the changed atoms
	 * 
	 * @return change atoms
	 */
	public IAtom[] getAtoms(){
		
		return this.atoms;
	}
	
	/**Retrieve the atom at the specified index
	 * 
	 * @param index
	 * @return atom at index
	 */
	public IAtom getAtom(int index){
		
		if(index < this.atoms.length){
			return this.atoms[index];
		}
		
		else{
			logger.error("Index out of range for change's atom array. Returning null.");
			return null;
		}
	}
	
	/**Get the number of atoms in the change
	 * 
	 * @return number of atoms
	 */
	public int getAtomCount(){
	
		return this.atoms.length;
	}
	
	/**Compares two changes.<br>
	 * Changes are considered equal if the types match and the atomtypes match.
	 */
	public boolean isReverseOf(Change b){
		
		
		if(this.atoms.length != b.getAtoms().length){return false;}
		else{
			//Independent of change type, the atoms in each change should be the same.
			for(int i = 0;	i < this.atoms.length;	i++){
				if(!b.contains(this.atoms[i].getSymbol())){return false;}
			}
			
			
			if(!this.changeType.isInverseOf(b.getChangeType())){return false;}
		}
		return true;
	}
	
	/**Compares two changes<br>
	 * Will only give true if atoms are identical as well (ie only useful for comparisons in a molecule<br>
	 * TODO: compare two more general changes, ie, match same atom types as well.
	 * 
	 * @param b
	 * @return this equals b
	 */
	public boolean isEqual(Change b){
	
		if(this.atoms.length != b.getAtoms().length){return false;}
		else{
			for(int i = 0;	i < this.atoms.length;	i++){
				if(!b.contains(this.atoms[i])){return false;}
			}
			if(!this.changeType.equals(b.getChangeType())){return false;}
		}
		return true;
	}
	
	/**Check whether a change contains an atom<br>
	 * Atoms equality defined by the equals method: very strict!
	 *  
	 * @param atom
	 * @return change contains atom
	 */
	public boolean contains(IAtom atom){
		
		for(int i = 0;	i < this.atoms.length;	i++){
			if(this.atoms[i].equals(atom)){return true;}
		}
		return false;
	}
	
	/**Check whether a change contains an atom<br>
	 * Atom is identified by its symbol.
	 *  
	 * @param atom
	 * @return change contains atom
	 */
	public boolean contains(String atom){
		
		for(int i = 0;	i < this.atoms.length;	i++){
			if(this.atoms[i].getSymbol().equals(atom)){return true;}
		}
		return false;
	}
	
	/**Set the atom indices from the original reactant, such that the apply method can be used, even with an
	 * atom container that does not contain the original atoms.
	 * 
	 * @param originalReactant
	 */
	public void setToApply(IAtomContainer originalReactant){
		applicationIndices			=	new int[this.atoms.length];
		for(int i = 0;	i < this.atoms.length;	i++){
			applicationIndices[i]	=	originalReactant.getAtomNumber(this.atoms[i]);
		}
	}
	
	/**Apply the change to an atom container which consists of all reactants.
	 * 
	 * @param reactants
	 */
	public void apply(IAtomContainer reactants){
		
		switch(this.changeType){
		case DECREASE_BOND: this.decreaseBond(reactants);break;
		case INCREASE_BOND: this.increaseBond(reactants);break;
		case BREAK_BOND:	this.breakBond(reactants);break;
		case FORM_BOND:		this.formBond(reactants);break;
		case GAIN_RADICAL:	this.gainRadical(reactants);break;
		case LOSE_RADICAL:	this.loseRadical(reactants);break;
		case GAIN_CHARGE:	this.gainCharge(reactants);break;
		case LOSE_CHARGE:	this.loseCharge(reactants);break;
		default:			logger.error("Change type not defined");break;
		}
	}
	
	private void decreaseBond(IAtomContainer reactants){
		
		Order order	=	reactants.getBond(reactants.getAtom(applicationIndices[0]), 
										  reactants.getAtom(applicationIndices[1]))
								 .getOrder();
		
		switch(order){
		case DOUBLE: 	reactants.getBond(reactants.getAtom(applicationIndices[0]), reactants.getAtom(applicationIndices[1])).setOrder(Order.SINGLE);break;
		case TRIPLE: 	reactants.getBond(reactants.getAtom(applicationIndices[0]), reactants.getAtom(applicationIndices[1])).setOrder(Order.DOUBLE);break;
		case QUADRUPLE:	reactants.getBond(reactants.getAtom(applicationIndices[0]), reactants.getAtom(applicationIndices[1])).setOrder(Order.TRIPLE);break;
		default:		logger.error("Cannot decrease order: order is "+order);break;
		}
	}
	
	private void increaseBond(IAtomContainer reactants){
		
		Order order	=	reactants.getBond(reactants.getAtom(applicationIndices[0]), reactants.getAtom(applicationIndices[1])).getOrder();
		
		switch(order){
		case SINGLE: 	reactants.getBond(reactants.getAtom(applicationIndices[0]), reactants.getAtom(applicationIndices[1])).setOrder(Order.DOUBLE);break;
		case DOUBLE: 	reactants.getBond(reactants.getAtom(applicationIndices[0]), reactants.getAtom(applicationIndices[1])).setOrder(Order.TRIPLE);break;
		case TRIPLE:	reactants.getBond(reactants.getAtom(applicationIndices[0]), reactants.getAtom(applicationIndices[1])).setOrder(Order.QUADRUPLE);break;
		default:		logger.error("Cannot increase order: order is "+order);
		}
	}
	
	private void breakBond(IAtomContainer reactants){
		
		reactants.removeBond(reactants.getAtom(applicationIndices[0]), reactants.getAtom(applicationIndices[1]));
	}
	
	private void formBond(IAtomContainer reactants){
		
		reactants.addBond(applicationIndices[0], applicationIndices[1], Order.SINGLE);
	}
	
	private void gainRadical(IAtomContainer reactants){
		
		reactants.addSingleElectron(applicationIndices[0]);
	}
	
	private void loseRadical(IAtomContainer reactants){
		
		reactants.removeSingleElectron(reactants.getConnectedSingleElectronsList(reactants.getAtom(applicationIndices[0])).get(0));
	}
	
	private void gainCharge(IAtomContainer reactants){
		
		reactants.getAtom(applicationIndices[0]).setFormalCharge(reactants.getAtom(applicationIndices[0]).getFormalCharge() + 1);
	}
	
	private void loseCharge(IAtomContainer reactants){

		reactants.getAtom(applicationIndices[0]).setFormalCharge(reactants.getAtom(applicationIndices[0]).getFormalCharge() - 1);
	}
}