package io.readers;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

import org.apache.log4j.Logger;
import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.Reaction;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IReaction;
import org.openscience.cdk.io.MDLRXNV2000Reader;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;

import manipulators.RadicalHandler;


public class RXNFileReader {

	private static Logger logger	=	Logger.getLogger(RXNFileReader.class);
	private String fileName;
	
	public RXNFileReader(String fileName){
		this.fileName	=	fileName;
	}
	/**Interpret a name.rxn file to a CDK reaction. The boolean parameter indicates whether the atoms should be 
	 * processed in depth or not (ie. hybridization, valence etc).
	 * 
	 * @param percieveMolecule
	 * @return reaction
	 */
	public IReaction toReaction(boolean percieveMolecule){
		
		InputStream inn;
		IReaction rxn	=	new Reaction();
		
		try {
			inn	=	new FileInputStream(fileName);
			MDLRXNV2000Reader reader1	=	new MDLRXNV2000Reader(inn);
		    
			rxn	=	reader1.read(rxn);
			reader1.close();
			//clear memory
			reader1	=	null;
			inn		=	null;
			replaceRGroups(rxn);
			
			if(percieveMolecule){
				for(int i = 0;	i < rxn.getReactantCount();	i++){
					IAtomContainer reactant	=	rxn.getReactants().getAtomContainer(i);
					AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(reactant);			
					AtomContainerManipulator.fixImplicitHydrogenCount(reactant);
					AtomContainerManipulator.percieveAtomTypesAndConfigureUnsetProperties(reactant);
					
					if(checkMetals(reactant))
					AtomContainerManipulator.fixAromaticity(reactant);
				}
			
				for(int i = 0;	i < rxn.getProductCount();	i++){
					IAtomContainer product	=	rxn.getProducts().getAtomContainer(i);
					
					AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(product);			
					AtomContainerManipulator.fixImplicitHydrogenCount(product);
					AtomContainerManipulator.percieveAtomTypesAndConfigureUnsetProperties(product);
					
					if(checkMetals(product))
					AtomContainerManipulator.fixAromaticity(product);
				}
			}
			
			(new RadicalHandler()).correctMultipleRadicals(rxn);
			
		} catch (FileNotFoundException e) {
			logger.error("ERROR: Reaction file not found!");
			//e.printStackTrace();
		} catch (CDKException e) {
			logger.error("ERROR: Failed transforming file to reaction!");
			//e.printStackTrace();
		} catch (IOException e) {
			logger.error("ERROR: Reader error!");
			e.printStackTrace();
		}	
		
		return rxn;
	}
	
	private static boolean checkMetals(IAtomContainer mol){
		
		for(IAtom atom:mol.atoms()){
			if(atom.getHybridization() == null){
				return false;
			}
		}
		
		return true;
	}
	
	private static void replaceRGroups(IReaction reaction){
		//If R groups are present without mapping: actual R group-> replace by H.
		if(reaction.getMappingCount() == 0){
			for(IAtomContainer reactant:reaction.getReactants().atomContainers()){
				for(int i = 0;	i < reactant.getAtomCount();	i++){
					IAtom atom	=	reactant.getAtom(i);
					
					if(atom.getSymbol().equals("R")){
						AtomContainerManipulator.replaceAtomByAtom(reactant, atom, DefaultChemObjectBuilder.getInstance().newInstance(IAtom.class, "C"));
					}
				}
			}
			
			for(IAtomContainer product:reaction.getProducts().atomContainers()){				
				for(int i = 0;	i < product.getAtomCount();	i++){
					IAtom atom	=	product.getAtom(i);
					
					if(atom.getSymbol().equals("R")){
						AtomContainerManipulator.replaceAtomByAtom(product, atom, DefaultChemObjectBuilder.getInstance().newInstance(IAtom.class, "C"));
					}
				}
			}
		}
	}
}