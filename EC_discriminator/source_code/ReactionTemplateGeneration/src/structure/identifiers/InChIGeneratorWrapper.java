package structure.identifiers;

import org.apache.log4j.Logger;
import org.openscience.cdk.CDKConstants;
import org.openscience.cdk.PseudoAtom;
import org.openscience.cdk.aromaticity.Aromaticity;
import org.openscience.cdk.aromaticity.ElectronDonation;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.graph.Cycles;
import org.openscience.cdk.inchi.InChIGenerator;
import org.openscience.cdk.inchi.InChIGeneratorFactory;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;

import structure.RemoveAromaticity;


public class InChIGeneratorWrapper {
	private static Logger logger = Logger.getLogger(InChIGeneratorWrapper.class);
	public static synchronized void generate(IAtomContainer mol){
		RemoveAromaticity.RemoveArom(mol);
		InChIGeneratorFactory factory;
		String inchi;
		try {
			factory = InChIGeneratorFactory.getInstance();
			InChIGenerator gen;
			gen = factory.getInChIGenerator(mol);
			inchi = gen.getInchi();
			if(inchi == null){
				int isMetal	=	0;
				//if atoms in the atom container are pseudo atoms: metal center=> count them and add line to inchi
				//in format:
				// /k#number of centra
				
				for(IAtom atom:mol.atoms()){
					if(atom instanceof PseudoAtom)
						isMetal++;
				}
				
				if(isMetal == 0)
					logger.error("InChI could not be generated for :"+mol.toString());
				else{
					try {
						IAtomContainer adsorbate	=	mol.clone();
						for(IAtom atom:mol.atoms()){
							if(atom instanceof PseudoAtom)
								mol.removeAtomAndConnectedElectronContainers(atom);
						}
						
						try{
							InChIGenerator gen2	=	factory.getInChIGenerator(adsorbate);
							String inchi2		=	gen2.getInchi();
							
							//If inchi can't be generated for remaining molecule:error. (unless remaining molecule
							//is empty: then adsorbate was free center
							if(inchi2 == null && adsorbate.getAtomCount() != 0){
								logger.error("InChI counld not be generated for: "+mol.toString());
							}
							else if(adsorbate.getAtomCount() == 0){
								inchi2	=	"InChI=1S/k"+isMetal;
								mol.setProperty(CDKConstants.INCHI, inchi2);
							}
							else{
								inchi2+="/k"+isMetal;
								mol.setProperty(CDKConstants.INCHI, inchi2);
							}
						}
						catch(CDKException e){}
						
					} catch (CloneNotSupportedException e) {}
				}
			}
			else mol.setProperty(CDKConstants.INCHI, inchi);
		} catch (CDKException e) {}
		try {
			Aromaticity arom = new Aromaticity(ElectronDonation.piBonds(),
	                Cycles.allOrVertexShort());
			arom.apply(mol);
		} catch (CDKException e) {
			e.printStackTrace();
		}
	}
}
