package manipulators;

import java.util.List;

import org.apache.log4j.Logger;
import org.openscience.cdk.Atom;
import org.openscience.cdk.Bond;
import org.openscience.cdk.SingleElectron;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IAtomType;
import org.openscience.cdk.interfaces.IBond;
import org.openscience.cdk.interfaces.IBond.Order;
import org.openscience.cdk.interfaces.IReaction;
import org.openscience.cdk.interfaces.ISingleElectron;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;

public class RadicalHandler {
	
	private static Logger logger=Logger.getLogger(RadicalHandler.class);
	
	public RadicalHandler(){
		
	}
	
	/**Convert the single electrons of the molecule to dummy atoms
	 * 
	 * @param mol
	 */
	public void changeRadical(IAtomContainer mol){
				
		int sec	=	mol.getSingleElectronCount();
		
		for(int i = 0;	i < sec;	i++){
			IAtom RadicalDummy	=	new Atom("X1000");
			RadicalDummy.setImplicitHydrogenCount(0);
			RadicalDummy.setHybridization(IAtomType.Hybridization.S);
			
			int atomNumber	=	mol.getAtomNumber(mol.getSingleElectron(0).getAtom());	
			mol.removeSingleElectron(0);
			IBond dummyBond	=	new Bond(RadicalDummy,mol.getAtom(atomNumber));
			mol.addAtom(RadicalDummy);
			mol.addBond(dummyBond);
		}
	}

	/**Convert any dummy atoms back to radicals<br>
	 * Warning: this method does not remove the mapping of the dummy atom! This should be done separately.
	 * 
	 * @param mol
	 */
	public void changeDummy(IAtomContainer mol){
		int atomCount	=	mol.getAtomCount();
		
		for(int i = 0;	i < atomCount;	i++){
			IAtom at	=	mol.getAtom(i);
			//In RDT the dummy atom is converted to an R chain!
			if(at.getSymbol() == "R"){
				List<IBond> dummyBond	=	mol.getConnectedBondsList(at);
				
				if(dummyBond.size() != 1){
					logger.error("ERROR: Additional bond formed to radical!");
				}
				
				IAtom radicalAtom	=	dummyBond.get(0).getConnectedAtom(at);
				mol.removeAtomAndConnectedElectronContainers(at);
				
				ISingleElectron singleElectron	=	new SingleElectron(radicalAtom);
				mol.addSingleElectron(singleElectron);
				i--;
				atomCount--;
				
				try {
					AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(mol);
					correctMultipleRadicals(mol);
				} catch (CDKException e) {
					logger.error("Failed to correctly interpret molecule after removing dummy atom");
					e.printStackTrace();
				}
			}
		}
	}
	
	/**Multiple radicals have been poorly implemented in CKD (TODO: fix in CDK).
	 * For the time being, the trouble valency and neighbour counts are set by this method.
	 * @param rxn
	 */
	public void correctMultipleRadicals(IReaction rxn){
		
		for(int i = 0;	i < rxn.getReactantCount();	i++){
			correctMultipleRadicals(rxn.getReactants().getAtomContainer(i));
		}
		
		for(int i = 0;	i < rxn.getProductCount();	i++){
			correctMultipleRadicals(rxn.getProducts().getAtomContainer(i));
		}
	}
	
	/**Multiple radicals have been poorly implemented in CKD (TODO: fix in CDK).
	 * For the time being, the trouble valency and neighbour counts are set by this method.
	 * @param rxn
	 */
	public void correctMultipleRadicals(IAtomContainer container){
		
		if(container.getSingleElectronCount() > 1){
			for(IAtom atom:container.atoms()){
				if(container.getConnectedSingleElectronsCount(atom) == 2 && atom.getSymbol().equals("O")){
					atom.setFormalNeighbourCount(0);
					atom.setValency(0);
				}
				
				if(container.getConnectedSingleElectronsCount(atom) == 3 && atom.getSymbol().equals("C")){
					atom.setFormalNeighbourCount(1);
					atom.setValency(1);
				}
				
				if(container.getConnectedSingleElectronsCount(atom) == 2 && atom.getSymbol().equals("C")){
					atom.setFormalNeighbourCount(2);
					if(container.getMaximumBondOrder(atom) == Order.DOUBLE){
						atom.setValency(1);
					}else{
						atom.setValency(2);
					}
				}
			}
		}
	}
}