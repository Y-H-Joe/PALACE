package manipulators;

import java.util.ArrayList;
import java.util.List;

import org.openscience.cdk.CDKConstants;
import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IBond;
import org.openscience.cdk.interfaces.IChemObjectBuilder;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;

import constants.StringConstants;

public class HydrogenAdder {
	
	private static List<IAtom> hydrogen;
	private int counter;
	
	/**Create a new hydrogen for each hydrogen that must be added, such that they are not identical!
	 * 
	 * @param numberOfHydrogens
	 */
	public HydrogenAdder(int numberOfHydrogens){

		hydrogen	=	new ArrayList<IAtom>();
		
		for(int i = 0;	i < numberOfHydrogens;	i++){
			hydrogen.add(DefaultChemObjectBuilder.getInstance().newInstance(IAtom.class, "H"));
			
			hydrogen.get(i).setAtomTypeName("H");
			hydrogen.get(i).setImplicitHydrogenCount(0);
		}
		
		counter		=	0;
	}
	
	/**Replaces one implicit hydrogen by an explicit one on the specified atom and returns the index
	 * of the bond in the molecule.
	 * 
	 * @param molecule
	 * @param atom
	 * @return the index of the new bond with hydrogen in the atom container
	 */
	private int addOneHydrogen(IAtomContainer molecule,IAtom atom){
		
		int implicitHC				=	atom.getImplicitHydrogenCount();
		int newImplicitHC			=	implicitHC-1;
		IChemObjectBuilder builder	=	DefaultChemObjectBuilder.getInstance();
		IBond newHB					=	builder.newInstance(IBond.class,atom, hydrogen.get(counter), CDKConstants.BONDORDER_SINGLE);
		
		atom.setImplicitHydrogenCount(newImplicitHC);
		molecule.addAtom(hydrogen.get(counter));
		molecule.addBond(newHB);
		
		try {
			AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(molecule);
			(new RadicalHandler()).correctMultipleRadicals(molecule);
		} catch (CDKException e) {
			e.printStackTrace();
		}
		
		this.nextHydrogen();
		return molecule.getBondNumber(newHB);
	}
	
	/**Add the specified number of explicit hydrogens to an atom in a molecule
	 * 
	 * @param molecule
	 * @param atom
	 * @param numberH
	 * @return array containing the indices of the new bonds
	 */
	public int[] addHydrogens(IAtomContainer molecule, IAtom atom){
		
		int[] indices	=	new int[hydrogen.size()];
		
		for(int i = 0;	i < hydrogen.size();	i++){
			indices[i]	=	addOneHydrogen(molecule, atom);
		}
		
		return indices;
	}
	
	/**Set the changes properties for newly added hydrogens.<br>
	 * Explicit hydrogens should only be added if they are changed during the reaction!
	 */
	public void setChanges(){
		
		for(IAtom hydro:hydrogen){
			hydro.setProperty(StringConstants.BONDCHANGE, true);
			hydro.setProperty(StringConstants.CHANGED, true);
			hydro.setProperty(StringConstants.NEIGHBOURCHANGE, true);
			hydro.setProperty(StringConstants.HYDROGENUSED, false);
		}
	}
	
	private void nextHydrogen(){
		
		counter++;
	}
}