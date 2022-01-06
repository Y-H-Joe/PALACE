package structure.identifiers;

import org.openscience.cdk.CDKConstants;
import org.openscience.cdk.interfaces.IAtomContainer;

public class IdenticalMolecules {

	public static boolean areIdentical(IAtomContainer mol1, IAtomContainer mol2){
		String inchi1 = mol1.getProperty(CDKConstants.INCHI);
		String inchi2 = mol2.getProperty(CDKConstants.INCHI);
		if(inchi1 == null){
			InChIGeneratorWrapper.generate(mol1);
			inchi1 = mol1.getProperty(CDKConstants.INCHI);
		}
		if(inchi2 == null){
			InChIGeneratorWrapper.generate(mol2);
			inchi2 = mol2.getProperty(CDKConstants.INCHI);
		}
		if(inchi1 != null && inchi2 != null){
			return (inchi1.equals(inchi2));
		}
		return false;
	}
}
