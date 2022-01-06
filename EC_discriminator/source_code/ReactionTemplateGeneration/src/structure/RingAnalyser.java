package structure;

import org.openscience.cdk.CDKConstants;
import org.openscience.cdk.RingSet;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IBond;
import org.openscience.cdk.interfaces.IRingSet;
import org.openscience.cdk.ringsearch.AllRingsFinder;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;

import constants.StringConstants;

/**Class to detect rings in a molecule. Only molecules that have aromatic rings which change during the reaction
 * are detected here
 *  
 * @author pplehier
 *
 */
public class RingAnalyser {

	private IAtomContainer mol;
	private IRingSet rings;
	
	/**
	 * @category constructor
	 * @param mol
	 */
	public RingAnalyser(IAtomContainer mol){
		this.mol	=	mol;
		this.rings	=	this.getChangedAromaticRings();
	}
	
	/**Retrieve the detected aromatic, changed rings
	 * 
	 * @return rings
	 */
	public IRingSet getRings(){
		return rings;
	}
	
	/**Detect which rings are both aromatic and changed by the reaction.
	 * 
	 * @return rings
	 */
	private IRingSet getChangedAromaticRings(){
		try {
			AtomContainerManipulator.fixAromaticity(mol);
		} catch (CDKException e) {}
		
		IRingSet rings			=	new RingSet();
		AllRingsFinder finder	=	new AllRingsFinder();
		try {
			rings	=	finder.findAllRings(mol);

		} catch (CDKException e) {}
		
		int atomContainerCount	=	rings.getAtomContainerCount();
		
		for(int i = 0;	i < atomContainerCount;	i++){
			IAtomContainer ring	=	rings.getAtomContainer(i);
			boolean changed		=	false;
			boolean aromatic	=	true;
			for(IBond bond:ring.bonds()){
				IAtom atom1			=	bond.getAtom(0);
				IAtom atom2			=	bond.getAtom(1);
				//Ring is aromatic when all bonds in the ring are aromatic.
				aromatic			=	aromatic && bond.getFlag(CDKConstants.ISAROMATIC);
				//Bond is changed when both atoms have changed
				boolean bondChange	=	(boolean)(atom1.getProperty(StringConstants.BONDCHANGE)!=null?
												atom1.getProperty(StringConstants.BONDCHANGE):
												(boolean)((IAtom)atom1.getProperty(StringConstants.MAPPINGPROPERTYATOM)).getProperty(StringConstants.BONDCHANGE))
										&&
										(boolean)(atom2.getProperty(StringConstants.BONDCHANGE)!=null?
												atom2.getProperty(StringConstants.BONDCHANGE):
												(boolean)((IAtom)atom2.getProperty(StringConstants.MAPPINGPROPERTYATOM)).getProperty(StringConstants.BONDCHANGE));
				
				if(bondChange){
					changed	=	true;
				}
			}
			
			ring.setProperty(Names.Changed, changed);
			if(!aromatic || ring.getAtomCount() > 7){
				rings.removeAtomContainer(ring);
				i--;
				atomContainerCount--;
			}
		}
		
		return rings;
	}
	
	/**Count how many atoms are shared between two rings in the set. Only runs if the indices are both smaller
	 * than the number of rings in the set.
	 * 
	 * @param index1
	 * @param index2
	 * @return number of shared atoms
	 */
	public int countShared(int index1, int index2){
		
		if(index1 >= rings.getAtomContainerCount() || index2 >= rings.getAtomContainerCount())
			return -1;
		
		int shared	=	0;
		String prop1=	StringConstants.EMPTY;
		String prop2=	StringConstants.EMPTY;
		
		switch(index1){
		case 0: prop1 = Names.A;break;
		case 1: prop1 = Names.B;break;
		case 2: prop1 = Names.C;break;
		default: break;
		}
		switch(index2){
		case 0: prop2 = Names.A;break;
		case 1: prop2 = Names.B;break;
		case 2: prop2 = Names.C;break;
		default: break;
		}
		
		for(IAtom atom1:rings.getAtomContainer(index1).atoms()){
			atom1.setProperty(prop1, true);
			for(IAtom atom2:rings.getAtomContainer(index2).atoms()){
				atom2.setProperty(prop2, true);
				if(atom1==atom2)
					shared++;
			}
		}
		
		return shared;
	}
}
