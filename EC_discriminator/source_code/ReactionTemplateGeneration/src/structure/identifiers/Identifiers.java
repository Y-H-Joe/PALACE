package structure.identifiers;

import org.openscience.cdk.CDKConstants;
import org.openscience.cdk.interfaces.IAtomContainer;

public class Identifiers {
	
	public static String getIdentifier(IAtomContainer mol){
		if(mol.getProperty(CDKConstants.SMILES) == null){
			if(mol.getProperty(CDKConstants.INCHI) == null){
				InChIGeneratorWrapper.generate(mol);
				return mol.getProperty(CDKConstants.INCHI);
			}else{
				return mol.getProperty(CDKConstants.INCHI);
			}
		}else{
			return mol.getProperty(CDKConstants.SMILES);
		}
	}
}
