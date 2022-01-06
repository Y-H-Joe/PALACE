package io;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.log4j.Logger;
import org.openscience.cdk.AtomContainer;
import org.openscience.cdk.CDKConstants;
import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.Reaction;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.inchi.InChIGeneratorFactory;
import org.openscience.cdk.inchi.InChIToStructure;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IReaction;
import org.openscience.cdk.smiles.SmilesParser;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;

import interfaces.IInput;
import io.readers.ChemkinFileReader;
import manipulators.ReactionBuilder;
import structure.identifiers.InChIGeneratorWrapper;
import tools.Tools;
/**
 * TODO:   How to take third body/pressure dependence in Genesys?
 * @author pplehier
 *
 */
public class ChemkinInput implements IInput{

	private static final String[] kineticsDefault	=	{"Kinetics","GROUP_ADDITIVITY","USER.xml","0","0"};
	private static final String reversibleA			=	"<=>";
	private static final String reversibleB			=	"=";
	private static final String irreversible		=	"=>";
	private static Logger logger	=	Logger.getLogger(ChemkinInput.class);
	private String inputPath;
	private List<IReaction> rxnInput;
	private List<String> nameInput;
	private List<Boolean> bimolInput;
	private List<String[]> kinetics;
	private Map<String,String> namesToIdentifiers;
	private Map<String,IAtomContainer> identifiersToMolecules;
	private double temperature;
	private int inputCount;
		
	public ChemkinInput(String inputPath){
		
		this.inputPath			=	inputPath;
		rxnInput				=	new ArrayList<IReaction>();
		nameInput				=	new ArrayList<String>();
		bimolInput				=	new ArrayList<Boolean>();
		identifiersToMolecules	=	new HashMap<String, IAtomContainer>();
		
		processFile();
		
		this.kinetics	=	defaultKinetics();
		this.temperature=	defaultTemperature();
	}
	
	private void processFile(){
		
		ChemkinFileReader cfr	=	new ChemkinFileReader(this.inputPath);
		cfr.read();
		
		List<String> reactionNames	=	cfr.getReactionNames();
		namesToIdentifiers			=	cfr.getNamesToIdentifiers();
		
		for(String reaction:reactionNames){
			//Remove the kinetic parameters of the reaction.
			String reactionNoKin=	reaction.split("\t")[0].trim();
			if(reactionNoKin.contains(reversibleA)){
				rxnInput.addAll(parseReversible(reactionNoKin,reversibleA));
				String reactionNoKinLegal	=	Tools.removeIllegal(reactionNoKin);
				nameInput.add(reactionNoKinLegal+"_forward");
				nameInput.add(reactionNoKinLegal+"_reverse");
				inputCount+=2;
			}
			else if(reactionNoKin.contains(irreversible)){
				rxnInput.add(parseIrreversible(reactionNoKin));
				String reactionNoKinLegal	=	Tools.removeIllegal(reactionNoKin);
				nameInput.add(reactionNoKinLegal);
				inputCount+=1;
			}
			else if(reactionNoKin.contains(reversibleB)){
				rxnInput.addAll(parseReversible(reactionNoKin,reversibleB));
				String reactionNoKinLegal	=	Tools.removeIllegal(reactionNoKin);
				nameInput.add(reactionNoKinLegal+"_forward");
				nameInput.add(reactionNoKinLegal+"_reverse");
				inputCount+=2;
			}
			else{
				logger.error("Reaction line \""+reaction+"\" does not contain reaction arrow. Ignoring ...");
				System.exit(-1);
			}
		}
	}
	
	/**Parse the reaction name that was read from the chemkin file for irreversible reactions
	 * 
	 * @param reaction
	 * @param arrow
	 * @return forwards reaction
	 * @throws CDKException
	 */
	private IReaction parseIrreversible(String reaction){
		
		reaction			=	removeThirdBody(reaction);
		String[] split		=	reaction.split("=>");
		String reactants	=	split[0].trim();
		String products		=	split[1].trim();
		String[] reactant	=	Tools.split(reactants,'+');
		String[] product	=	Tools.split(products,'+');
		IReaction forward	=	new Reaction();

		for(String react:reactant){
			
			String[] s	=	Tools.startsWithNumber(react);
			int coef	=	Integer.parseInt(s[0]);
			react		=	s[1];
			
			if(react.equals("M") || react.equals("(M)")){/*Do not add third body to CDK reaction*/}
			else{
				try{
					String id	=	namesToIdentifiers.get(react);
					
					if(id==null){
						logger.fatal("Molcule named "+react+" is not properly defined in chemkin. Exiting...");
						System.exit(-1);
					}
					
					IAtomContainer molecule	=	parseMolecule(id);
					
					for(int i = 0; i < coef; i++)
						forward.addReactant(molecule.clone());
				}
				catch(CDKException e){
					logger.error("Incorrect species identifier found for "+react);
					System.exit(-1);
				} 
				catch (CloneNotSupportedException e) {
					e.printStackTrace();
				}
			}
		}

		for(String prod:product){
			
			String[] s	=	Tools.startsWithNumber(prod);
			int coef	=	Integer.parseInt(s[0]);
			prod		=	s[1];
			
			if(prod.equals("M") || prod.equals("(M)")){/*Do not add third body to CDK reaction*/}
			else{
				try{
					String id	=	namesToIdentifiers.get(prod);
				
					if(id==null){
						logger.fatal("Molcule named "+prod+" is not properly defined in chemkin. Exiting...");
						System.exit(-1);
					}
					
					IAtomContainer molecule	=	parseMolecule(id);
					
					for(int i = 0; i < coef; i++)
						forward.addProduct(molecule.clone());
				}
				catch(CDKException e){
					logger.error("Incorrect species identifier found for "+prod);
					System.exit(-1);
				} 
				catch (CloneNotSupportedException e) {
					e.printStackTrace();
				}
			}
		}
		
		bimolInput.add(forward.getReactantCount()==1?false:true);
		forward	=	ReactionBuilder.build(forward);
		
		return forward;
	}
	
	
	/**Parse the reaction name that was read from the chemkin file for reversible reactions
	 * 
	 * @param reaction
	 * @param arrow
	 * @return associated forwards and backwards reactions
	 * @throws CDKException
	 */
	private List<IReaction> parseReversible(String reaction, String arrow){
	
		reaction			=	removeThirdBody(reaction);
		String[] split		=	reaction.split(arrow);
		String reactants	=	split[0].trim();
		String products		=	split[1].trim();
		String[] reactant	=	Tools.split(reactants,'+');
		String[] product	=	Tools.split(products,'+');
		IReaction forward	=	new Reaction();
		IReaction reverse	=	new Reaction();
		
		for(String react:reactant){
			
			String[] s	=	Tools.startsWithNumber(react);
			int coef	=	Integer.parseInt(s[0]);
			react		=	s[1];
			
			if(react.equals("M") || react.equals("(M)")){/*Do not add third body to CDK reaction*/}
			else{
				try{
					String id	=	namesToIdentifiers.get(react);
					
					if(id==null){
						logger.fatal("Molcule named "+react+" is not properly defined in chemkin. Exiting...");
						System.exit(-1);
					}
					
					IAtomContainer molecule	=	parseMolecule(id);
					
					for(int i = 0; i < coef; i++){
						forward.addReactant(molecule.clone());
						reverse.addProduct(molecule.clone());
					}
				}
				catch(CDKException e){
					logger.error("Incorrect species identifier found for "+react);
					System.exit(-1);
				} 
				catch (CloneNotSupportedException e) {
					e.printStackTrace();
				}
			}
		}
		
		for(String prod:product){
			
			String[] s	=	Tools.startsWithNumber(prod);
			int coef	=	Integer.parseInt(s[0]);
			prod		=	s[1];
			
			if(prod.equals("M") || prod.equals("(M)")){/*Do not add third body to CDK reaction*/}
			else{
				try{
					String id	=	namesToIdentifiers.get(prod);
					
					if(id==null){
						logger.fatal("Molcule named "+prod+" is not properly defined in chemkin. Exiting...");
						System.exit(-1);
					}
					
					IAtomContainer molecule	=	parseMolecule(id);
					
					for(int i = 0; i < coef; i++){
						forward.addProduct(molecule.clone());
						reverse.addReactant(molecule.clone());
					}
				}
				catch(CDKException e){
					logger.error("Incorrect species identifier found for "+prod);
					System.exit(-1);
				} 
				catch (CloneNotSupportedException e) {
					e.printStackTrace();
				}
			}
		}
		
		List<IReaction> output	=	new ArrayList<IReaction>();
		//Process radicals!
		forward					=	ReactionBuilder.build(forward);
		reverse					=	ReactionBuilder.build(reverse);
		
		output.add(forward);
		output.add(reverse);
		bimolInput.add(forward.getReactantCount()==1?false:true);
		bimolInput.add(reverse.getReactantCount()==1?false:true);
		
		return output;
	}
	
	/**Construct a molecule from an identifier string
	 * 
	 * @param id
	 * @return molecule
	 * @throws CDKException
	 */
	private IAtomContainer parseMolecule(String id) throws CDKException{
		
		IAtomContainer molecule	=	new AtomContainer();
		
		//Parse InChI's
		if(id.startsWith("InChI=")){
			if(identifiersToMolecules.containsKey(id)){//If molecule has already been detected, look it up.
				molecule	=	identifiersToMolecules.get(id);
			}
			else{//If it hasn't, construct new molecule based on InChI, and add InChI to list.
				InChIToStructure itc	=	InChIGeneratorFactory.getInstance().getInChIToStructure(id, DefaultChemObjectBuilder.getInstance());
				molecule				=	itc.getAtomContainer();
				AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(molecule);
				identifiersToMolecules.put(id, molecule);
			}
		}
		//Otherwise, parse Smiles
		else{
			SmilesParser sp			=	new SmilesParser(DefaultChemObjectBuilder.getInstance());
			molecule				=	sp.parseSmiles(id);
			
			AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(molecule);
			InChIGeneratorWrapper.generate(molecule);
			
			String inChI	=	molecule.getProperty(CDKConstants.INCHI);
			
			if(!identifiersToMolecules.containsKey(inChI)){
				identifiersToMolecules.put(inChI, molecule);
			}
		}
		
		return molecule;
	}
	
	@Override
	public List<Boolean> getBimol() {

		return bimolInput;
	}

	@Override
	public List<String[]> getConstraints() {
		
		return null;
	}

	@Override
	public int getInputCount() {

		return inputCount;
	}

	@Override
	public List<String[]> getKinetics() {

		return this.kinetics;
	}

	@Override
	public List<String> getNames() {
		
		return this.nameInput;
	}

	@Override
	public List<IReaction> getReactions() {

		return this.rxnInput;
	}

	@Override
	public double getTemperature() {
		
		return this.temperature;
	}

	@Override
	public int tooManyReactantCount() {
		
		return 0;
	}

	@Override
	public String getInputPath() {
		
		return this.inputPath;
	}
	
	/**Sets the kinetics to default:<br>
	 *Group additivity block with file path set to USER = to be filled in by user after processing.
	 * 
	 * @return default kinetics
	 */
	private List<String[]> defaultKinetics(){
		
		List<String[]> kin	=	new ArrayList<String[]>();
		
		for(int i = 0;	i < inputCount;	i++){
			kin.add(kineticsDefault);
		}
			
		return kin;
	}
	
	/**Sets the temperature to the arbitrary default value of 298.15 K
	 * 
	 * @return 298.15
	 */
	private static double defaultTemperature(){
		
		return 298.15;
	}
	
	private static String removeThirdBody(String s){

		while(s.contains("(+")){
			int pos	=	s.indexOf("(+");
			int end	=	s.indexOf(")", pos);
			String front	=	s.substring(0, pos);
			String back		=	s.substring(end, s.length());
			if(back.startsWith(")")) back		=	back.substring(1, back.length());
			s	=	front + back;
		}
		return s;
	}
}
