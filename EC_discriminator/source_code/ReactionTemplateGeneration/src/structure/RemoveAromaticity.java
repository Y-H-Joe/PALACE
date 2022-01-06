package structure;

import org.openscience.cdk.CDKConstants;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IBond;


public class RemoveAromaticity {


	public static void RemoveArom(IAtomContainer mol){

		for (IBond bond : mol.bonds()) {
			if(bond.getFlag(CDKConstants.ISAROMATIC)){
				bond.setFlag(CDKConstants.ISAROMATIC,false);	
			}
		}
		for (IAtom atom : mol.atoms()) {
			if(atom.getFlag(CDKConstants.ISAROMATIC)){
				atom.setFlag(CDKConstants.ISAROMATIC,false);	
			}
		}

	}
}
