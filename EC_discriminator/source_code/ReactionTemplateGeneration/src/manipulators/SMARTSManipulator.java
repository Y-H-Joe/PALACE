package manipulators;

import org.openscience.cdk.AtomContainer;
import org.openscience.cdk.CDKConstants;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;

import tools.SMARTSGenerator;

public class SMARTSManipulator {

	public static final String Aromatic	=	"Aromatic in actual reactant";
	/**Generates a Smarts string for the given AtomContainer, but sets the Atom[index] as first atom of the 
	 * Smarts string.
	 * 
	 * @param mol
	 * @param index
	 * @return Smarts starting with atom[index].
	 */
	public static String setAsFirst(IAtomContainer mol,int index){
		
		return setAsFirst(mol, index, false);
	}
	
	/**Generates a Smarts string for the given AtomContainer, but sets the Atom[index] as first atom of the 
	 * Smarts string. The boolean indicates whether the generated smiles should be made unique (i.e consistent
	 * such that equal molecules will have equal smarts)
	 * 
	 * @param mol
	 * @param index
	 * @param unique
	 * @return Smarts starting with atom[index].
	 */
	public static String setAsFirst(IAtomContainer mol,int index,boolean unique){
		
		IAtomContainer copy	=	new AtomContainer(mol);
		//Set a new property so the atom will be recognized as aromatic, even after perceiving types 
		//As this is not the entire molecule anymore, aromaticity will not always be detected where it was previously
		//present.
		for(IAtom at:mol.atoms())
			if(at.getFlag(CDKConstants.ISAROMATIC))
				at.setProperty(Aromatic, true);

		
		try {
			AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(copy);
			(new RadicalHandler()).correctMultipleRadicals(copy);
		} catch (CDKException e) {
			e.printStackTrace();
		}
		
		int[] order	=	new int[copy.getAtomCount()];
		order[0]	=	index;
		
		for(int i = 1;	i < copy.getAtomCount();	i++){
			if(i-1 < index){
				order[i]	=	i-1;
			}
			
			if(i-1 >= index){
				order[i]	=	i;
			}
		}
		
		IAtom[] atoms	=	new IAtom[copy.getAtomCount()];
		
		for(int i = 0;	i < atoms.length;	i++){
			atoms[i]	=	copy.getAtom(order[i]);
		}
		
		copy.setAtoms(atoms);
		
		SMARTSGenerator smg	=	new SMARTSGenerator(copy,order,unique);
		
		return smg.getSMARTSConnection();
	}
	
	/**Generates a Smarts string for the given AtomContainer, but sets the Atom[index] as first atom of the 
	 * Smarts string. The boolean indicates whether the generated smiles should be made unique (i.e consistent
	 * such that equal molecules will have equal smarts)
	 * 
	 * @param mol
	 * @param atom in mol
	 * @param unique
	 * @return Smarts starting with atom[index].
	 */
	public static String setAsFirst(IAtomContainer mol,IAtom atom, boolean unique){
	
		int index			=	mol.getAtomNumber(atom);
		
		return setAsFirst(mol, index, unique);
	}
	
	/**Generates a Smarts string for the given AtomContainer, but sets the Atom[index] as first atom of the 
	 * Smarts string. 
	 * 
	 * @param mol
	 * @param atom in mol
	 * @return Smarts starting with atom[index].
	 */
	public static String setAsFirst(IAtomContainer mol,IAtom atom){
	
		int index			=	mol.getAtomNumber(atom);
		
		return setAsFirst(mol, index);
	}
}