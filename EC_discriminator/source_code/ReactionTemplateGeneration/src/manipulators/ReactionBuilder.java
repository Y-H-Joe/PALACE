package manipulators;

import org.apache.log4j.Logger;
import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.Reaction;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.exception.InvalidSmilesException;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IChemObjectBuilder;
import org.openscience.cdk.interfaces.IReaction;
import org.openscience.cdk.smiles.SmilesGenerator;
import org.openscience.cdk.smiles.SmilesParser;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;

public class ReactionBuilder {
	
	private static Logger logger=Logger.getLogger(ReactionBuilder.class);
	
	/**Build a reaction that is compatible with RDT, species are generated and added to the reaction separately.
	 * Radicals are transformed into dummy atoms to allow mapping
	 * 
	 * @param rxnSMILES
	 * @return RDT compatible reaction
	 */
	public static IReaction build(String rxnSMILES){
	
		IReaction rxn	=	new Reaction();
		
		try {
			rxn			=	fillSpecies(rxnSMILES);
			//processRadicals(rxn);
			
		} catch (InvalidSmilesException e) {
			logger.error("ERROR: Failed to construct reaction from smiles");
			e.printStackTrace();
		}
		
		return rxn;
	}
	
	/**Method to build a good reaction starting from a reaction. A good reaction is one that has dummy atoms
	 * in stead of radicals so it can be mapped by RDT.
	 * @param rxn
	 * @return rxn with dummy atoms
	 */
	public static IReaction build(IReaction rxn){
		
		IReaction build	=	getAromaticBondsSetToDoubleAndSingle(rxn);
		//processRadicals(build); No longer necessary, updated in RDT
		
		return build;
	}
	
	/**Remove the dummy atoms and make them single electrons as well. To avoid conflicts later on the mapping
	 * assigned to these dummy atoms must be removed as well.
	 * 
	 * @param rxn
	 */
	/*private static void revertRadicals(IReaction rxn){
		
		removeDummyMapping(rxn);
		processDummies(rxn);
	}*/
	
	/**Processes reaction smiles. Reactants and products are manually generated and then added to the reaction.
	 * The reaction smiles parser fails at generating the correct stereochemistry on the molecules.
	 * 
	 * @param rxnSMILES
	 * @return Reaction
	 * @throws InvalidSmilesException
	 */
	private static IReaction fillSpecies(String rxnSMILES) throws InvalidSmilesException{
		
		IChemObjectBuilder builder	=	DefaultChemObjectBuilder.getInstance();
		SmilesParser sp				=	new SmilesParser(builder);
		
		String[] RAP		=	rxnSMILES.split(">");
		String [] Reactants	=	RAP[0].split("\\.");
		String [] Products	=	RAP[2].split("\\.");
		String [] Agents	=	RAP[1].split("\\.");
		IReaction rxn		=	new Reaction();
		
		for(int i = 0;	i < Reactants.length;	i++){
			IAtomContainer React	=	sp.parseSmiles(Reactants[i]);
			
			try {
				AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(React);
				(new RadicalHandler()).correctMultipleRadicals(React);
			} catch (CDKException e) {
				logger.error("Failed to correctly percieve reactant "+i);
				e.printStackTrace();
			}
			
			rxn.addReactant(React);
		}
		
		for(int i = 0;	i < Products.length;	i++){
			IAtomContainer Prod	=	sp.parseSmiles(Products[i]);
			
			try {
				AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(Prod);
				(new RadicalHandler()).correctMultipleRadicals(Prod);
			} catch (CDKException e) {
				logger.error("Failed to correctly percieve product "+i);
				e.printStackTrace();
			}
			
			rxn.addProduct(sp.parseSmiles(Products[i]));
		}
		
		for(int i = 0;	i < Agents.length;	i++){
			IAtomContainer Agent	=	sp.parseSmiles(Agents[i]);
			
			try {
				AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(Agent);
			} catch (CDKException e) {
				logger.error("Failed to correctly percieve agent "+i);
				e.printStackTrace();
			}
			
			rxn.addAgent(sp.parseSmiles(Agents[i]));
		}
		
		return rxn;
	}
	
	/**Transform the dummy atoms back to single electrons, both in reactants and products. 
	 * 
	 * @param rxn
	 */
	/*private static void processDummies(IReaction rxn){
		
		RadicalHandler RH	=	new RadicalHandler();
		
		for(int i = 0;	i < rxn.getReactantCount();	i++){
			RH.changeDummy(rxn.getReactants().getAtomContainer(i));
		}
		
		for(int i = 0;	i < rxn.getProductCount();	i++){
			RH.changeDummy(rxn.getProducts().getAtomContainer(i));
		}
	}*/
	
	/**Transforms radicals into dummy atoms for this reaction. The processing is only done when radicals are
	 * present in either the reactants or products
	 * 
	 * @param rxn
	 */
	/*private static void processRadicals(IReaction rxn){
		
		RadicalHandler RH	=	new RadicalHandler();
		for(int i = 0;	i < rxn.getReactantCount();	i++){
			//only execute if radicals are initially present
			if(rxn.getReactants().getAtomContainer(i).getSingleElectronCount() != 0){
				RH.changeRadical(rxn.getReactants().getAtomContainer(i));
			}
		}
		
		for(int i = 0;	i < rxn.getProductCount();	i++){
			if(rxn.getProducts().getAtomContainer(i).getSingleElectronCount() != 0){
				RH.changeRadical(rxn.getProducts().getAtomContainer(i));
			}
		}
	}*/
	
	/**Remove the mapping in which the dummy atom (named R after passing RDT) is mapped to a reactant/product atom.
	 * A dummy atom will always be mapped to an other dummy atom, so searching only one side of the mapping
	 * suffices.
	 * 
	 * @param rxn
	 */
	/*private static void removeDummyMapping(IReaction rxn){
		
		int mappingCount	=	rxn.getMappingCount();
		
		for(int i = 0;	i < mappingCount;	i++){
			if(((IAtom)rxn.getMapping(i).getChemObject(0)).getSymbol().equals("R")){
				rxn.removeMapping(i);
				i--;
				mappingCount--;
			}
		}
	}*/
	
	/**Interpret the conjugated double bonds in ring systems to detect aromaticity. RDT does not output 
	 * aromatic bonds, but explicit double and single bonds (linked to the fact that aromatic smiles result in 
	 * erroneous RDT rxn output files). Therefore, aromaticity must be post-interpreted.
	 * 
	 * @param rxn
	 */
	public static void fixAromaticity(IReaction rxn){
		
		for(int i = 0;	i < rxn.getReactantCount();	i++){
			try {
				AtomContainerManipulator.fixAromaticity(rxn.getReactants().getAtomContainer(i));
			} catch (CDKException e) {
				logger.error("Failed to interpret aromaticity!");
				e.printStackTrace();
			}
		}
		
		for(int i = 0;	i < rxn.getProductCount();	i++){
			try {
				AtomContainerManipulator.fixAromaticity(rxn.getProducts().getAtomContainer(i));
			} catch (CDKException e) {
				logger.error("Failed to interpret aromaticity!");
				e.printStackTrace();
			}
		}
	}

	/**Method to process the read reaction from a .rxn file. The rxn format allows aromatic bonds to be specified
	 * by '4'. This results in the CDK bond order being set to 'UNSET', though a flag for aromaticity is added.<br>
	 * This causes problems when generating the smiles as for this type of aromatic bonds, the
	 * SmilesGenerator.isomeric().aromatic() must be used. This on its turn however triggers an unsolved bug in
	 * RDT: the output rxn files loose all info on aromaticity. <br>
	 * To allow a good input for RDT, aromatic bonds must be explicit (double-single). The fact that the bond is 
	 * aromatic is then determined via the fixAromaticity(). <br>
	 * This allows .rxn files with specified aromatic bonds to be read and smiles to be generated for RDT,
	 * while not compromising the functionality of other parts of the program.
	 * 
	 * @param Reaction without aromaticity
	 * @return Reaction with aromaticity
	 */
	private static IReaction getAromaticBondsSetToDoubleAndSingle(IReaction rxn){
		
		IReaction newRxn			=	new Reaction();
		SmilesGenerator sg			=	SmilesGenerator.isomeric();
		IChemObjectBuilder builder	=	DefaultChemObjectBuilder.getInstance();
		SmilesParser sp				=	new SmilesParser(builder);
		RadicalHandler rh			=	new RadicalHandler();
		
		for(int i = 0;	i < rxn.getReactantCount();	i++){
			try {
				IAtomContainer mol	=	sp.parseSmiles(sg.create(rxn.getReactants().getAtomContainer(i)));
				
				AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(mol);
				AtomContainerManipulator.fixImplicitHydrogenCount(mol);
				AtomContainerManipulator.percieveAtomTypesAndConfigureUnsetProperties(mol);
				AtomContainerManipulator.fixAromaticity(mol);
				rh.correctMultipleRadicals(mol);
				newRxn.addReactant(mol);
				
			} catch (Exception e) {
				logger.error("Failed to make aromaticity explicit for reactant "+i+"."+
							 "\nResults may not be correct.");
				newRxn.addReactant(rxn.getReactants().getAtomContainer(i));
			}	
		}
		
		for(int i = 0;	i < rxn.getProductCount();	i++){
			try {
				IAtomContainer mol	=	sp.parseSmiles(sg.create(rxn.getProducts().getAtomContainer(i)));
				
				AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(mol);
				AtomContainerManipulator.fixImplicitHydrogenCount(mol);
				AtomContainerManipulator.percieveAtomTypesAndConfigureUnsetProperties(mol);
				AtomContainerManipulator.fixAromaticity(mol);
				rh.correctMultipleRadicals(mol);
				newRxn.addProduct(mol);
				
			} catch (Exception e) {
				logger.error("Failed to make aromaticity explicit for product "+i+"."+
							 "\nResults may not be correct.");
				newRxn.addProduct(rxn.getProducts().getAtomContainer(i));
			}
		}

		return	newRxn;
	}
	
	public static void explicitHydrogen(IReaction reaction){
		
		for(IAtomContainer reactant:reaction.getReactants().atomContainers()){
			AtomContainerManipulator.convertImplicitToExplicitHydrogens(reactant);
		}
		for(IAtomContainer product:reaction.getProducts().atomContainers()){
			AtomContainerManipulator.convertImplicitToExplicitHydrogens(product);
		}
	}
}