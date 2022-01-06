package fileGeneration.generators;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IReaction;

import constants.StringConstants;
import interfaces.IInput;
import io.ChemkinInput;
import io.FileStructure;
import io.RXNInput;
import io.TextInput;
import io.readers.xmlClasses.Config;
import io.readers.xmlClasses.ConstraintType;
import io.readers.xmlClasses.InpConstraint;
import io.readers.xmlClasses.InpReactant;
import io.readers.xmlClasses.InpReactionFamily;
import io.readers.xmlClasses.InpRecipe;
import io.readers.xmlClasses.LimitType;
import reactionCenterDetection.ReactionCenter;
import reactionCenterDetection.ReactionCenterDetector;
import reactionMapping.MappedReaction;
import reactionMapping.RDTMapper;
import tools.Output;

public class ChemistryGenerator{
	
	private static Logger logger	=	Logger.getLogger(ChemistryGenerator.class);
	
	private Config config;
	private String outputDir;
	private final String uniqueDir	=	"UniqueReactions";
	@SuppressWarnings("unused")
	private List<String[]> constraints;
	private List<String[]> kinetics;
	private List<Boolean> bimolecular;
	private List<String> names;
	private int reactionCount;
	private List<ReactionCenter> uniqueCenters;
	private int fails;
	private int reverse;
	private int identical;
	private String failed;
	private StringBuffer summary;
	private boolean noFail;
	private ReactionCenter currentCenter;
	private List<IReaction> reactions;
	private double temperature;
	private IInput input;
	private int maxC;
	private int maxO;
	private int maxN;
	private int maxS;	
	
	/**Copy the input data from the input file to this generator.
	 * For the reaction centers, an empty array is made, which will be filled in by the algorithm later on <br>
	 * TODO: How to define constraints per input?<br>
	 * TODO: How to define kinetics per input, currently: default 0<br>
	 * 
	 * @param Input file name
	 */
	private void processInputChemistry(){
		
		names			=	input.getNames();
		reactions		=	input.getReactions();
		bimolecular		=	input.getBimol();
		reactionCount	=	input.getInputCount();
		kinetics		=	input.getKinetics();
		constraints		=	input.getConstraints();
		temperature		=	input.getTemperature();
		maxC			=	0;
		maxO			=	0;
		maxN			=	0;
		maxS			=	0;
		//filled in later
		currentCenter	=	null;	
	}
	/*
	/**Copy the input data from the input folder to this generator<br>
	 * 
	 * For the reaction centers, an empty array is made, which will be filled in by the algorithm later on
	 * 
	 * @param Input directory name
	 
	private void processRXNInputChemistry(){
		
			
		
		names			=	inputFolder.getNames();
		reactions		=	inputFolder.getReactions();
		bimolecular		=	inputFolder.getBimol();
		reactionCount	=	inputFolder.getInputCount();
		//set to default
		kinetics		=	defaultKinetics();	
		temperature		=	defaultTemperature();
		//not yet used
		constraints		=	new ArrayList<String[]>();	
		//filled in later
		reactionCenters	=	new ArrayList<ReactionCenter>();
	}
	
	/**Copy the input data from the input file to this generator.
	 * For the reaction centers, an empty array is made, which will be filled in by the algorithm later on
	 * 
	 * @param Input file name
	 
	private void processTextInputChemistry(){
		
		names			=	inputFile.getNames();
		reactions		=	inputFile.getReactions();
		bimolecular		=	inputFile.getBimol();
		reactionCount	=	inputFile.getInputCount();
		kinetics		=	inputFile.getKinetics();
		constraints		=	inputFile.getConstraints();
		temperature		=	inputFile.getTemperature();
		//filled in later
		reactionCenters	=	new ArrayList<ReactionCenter>();	
	}	*/
	
	/**Generate the lines for the family corresponding to reaction[index].
	 * 
	 * @param index of reaction
	 * @return family block
	 */
	@SuppressWarnings("unchecked")
	private InpReactionFamily generateFamily(int index, boolean zeroKinetics){
		
		InpReactionFamily reactionFamily	=	new InpReactionFamily();
		
		reactionFamily.setName(names.get(index));
		reactionFamily.setBimolecular(bimolecular.get(index));
		List<Object> recipeAndReactants	=	RecipeAndReactantsGenerator.generate(currentCenter);
		reactionFamily.setInpRecipe((InpRecipe) recipeAndReactants.get(0));
		reactionFamily.setInpReactant((List<InpReactant>) recipeAndReactants.get(1));
		reactionFamily.setInpKinetics(KineticsGenerator.generate(kinetics.get(index),currentCenter, zeroKinetics));
		
		uniqueCenters.add(currentCenter);
		
		return reactionFamily;
	}
	
	/**Detect and add the reaction center for the reaction[index] of the input, returns true if the mapping
	 * of the reaction was successful (ie not empty), returns false otherwise
	 * 
	 * @param index of reaction
	 */
	private boolean determineReactionCenter(int index, boolean specific){
	
		RDTMapper.setOutputFolder(outputDir);
		MappedReaction mappedReaction	=	RDTMapper.mapReaction(reactions.get(index),names.get(index));
		ReactionCenterDetector rcd		=	new ReactionCenterDetector(mappedReaction, specific);
		updateProductConstraints(mappedReaction);
		
		if(RDTMapper.lastMappingSucces()){
			ReactionCenter RC		=	rcd.detectReactionCenters();
			RC.setName(names.get(index));
			
			if(RC.noChanges()){
				logger.info("Reaction mapping indicates that "+names.get(index)+" is an identical reaction, not adding ...");
				identical++;
				noFail	=	true;
				return false;
			}
			else if(RC.radicalMechanismOK()){
				currentCenter	=	RC;
				return true;
			}
			else{
				logger.info("The reaction contains radicals that do not participate in the reaction. " +
							"Reaction mapping likely to be incorrect, not adding ...");
				return false;
			}
		}
		else{
			logger.info("Reaction mapping failed, no reaction family information for "+names.get(index)+", not adding ...");
			return false;
		}
	}
	
	/**Checks whether the reaction center with index is the same as any other reaction center in the system.<br>
	 * Returns -1 if no same families have been found<br>
	 * Returns the index of the same family if one is found<br>
	 * Counts how often a reaction family is encountered. Reversed reaction families are counted with the forwards reaction family
	 * @param index of reaction
	 * @return index of the first identical reaction center
	 */
	private int[] checkReactionCenter(ReactionCenter RC){
	
		ReactionCenter RC_A	=	RC;
		int[] ans	=	{-1,-1};
		
		if(RC_A == null)
			return ans;
		
		boolean set	=	false;
		for(int i = 0;	i < uniqueCenters.size();	i++){
			ReactionCenter RC_B	=	uniqueCenters.get(i); 
			if(RC_A.isSameReactionFamily(RC_B)){
				ans[0]	=	i;
				if(RC_B.isReverse()){	}
				else{
					RC_B.encountered();
				}
				break;
			}
			if(!set && RC_A.isReverseReactionFamily(RC_B)){
				RC_A.setReverseCenter(RC_B);
				RC_B.encountered();
				ans[1]	=	i;
				set		=	true;
			}	
		}
		
		if(ans[0] == -1 && ans[1] == -1){
			RC_A.encountered();
		}
		return ans;
	}
	
	/**Get the number of the selected resonance structure such that the correct .rxn file can be copied to the
	 * unique reactions folder
	 * 
	 * @param index
	 * @return resonance structure number
	 */
	private int getResonanceName(int index){
		
		return currentCenter.getReaction().getResonanceStructureNumber();
	}
	
	/**Constructs a new Chemistry generator
	 * 
	 * @param outputFolder
	 * @param input
	 */
	private ChemistryGenerator(Config config,String outputFolder, String input){
		
		this.config		=	config;
		outputDir		=	outputFolder;
		fails			=	0;
		reverse			=	0;
		uniqueCenters	=	new ArrayList<ReactionCenter>();
		reactionCount	=	0;
		identical		=	0;
		failed			=	StringConstants.EMPTY;
		noFail			=	false;
		summary			=	new StringBuffer();
		
		if(new File(input).isDirectory()){
			this.input	=	new RXNInput(input);
			processInputChemistry();
		}
		else if(input.endsWith(".txt")){
			this.input	=	new TextInput(input);
			processInputChemistry();
		}
		else if(input.endsWith(".rxn")){
			this.input	=	new RXNInput(input);
			processInputChemistry();
		}
		else if(input.endsWith(".inp")){
			this.input	=	new ChemkinInput(input);
			processInputChemistry();
		}
	}
	
	private void copyRXNFile(String name, int index, int resonance) throws IOException{
		
		File destDir		=	new File(outputDir+"\\"+uniqueDir);
		File dest			=	new File(outputDir+"\\"+uniqueDir+"\\"+name+".rxn");
		File[] dirs			=	new File(outputDir).listFiles();
		File currentMapping = 	null;
		
		destDir.mkdir();
		
		//Find the folder in which the data for this reaction has been stored.
		for(File file:dirs){
			if(file.getName().contains(name)){
				currentMapping	=	file;
				break;
			}
		}
		File rxnFile	=	null;
		
		//Find the reaction file corresponding to the analyzed mapping.
		if(currentMapping != null){
			rxnFile	=	currentMapping.listFiles(FileStructure.getRXNFileFilter())[resonance];
		}
		//if current mapping is null: reaction contained user defined mapping: copy the reaction from the input folder
		else{
			File inputDir	=	new File(this.input.getInputPath());
			rxnFile			=	inputDir.listFiles()[index];	
		}
		
		Files.copy(rxnFile.toPath(), dest.toPath(), StandardCopyOption.REPLACE_EXISTING);
	}
	
	private void updateProductConstraints(MappedReaction reaction){
		
		for(IAtomContainer product:reaction.getProducts().atomContainers()){
			
			int prodC	=	product.getCAtomCount();
			int prodO	=	product.getOAtomCount();
			int prodN	=	product.getNAtomCount();
			int prodS	=	product.getSAtomCount();
			
			if(prodC > maxC)	maxC	=	prodC;
			if(prodO > maxO)	maxO	=	prodO;
			if(prodN > maxN)	maxN	=	prodN;
			if(prodS > maxS)	maxS	=	prodS;
		}
		
	}
	
	private void generateSummary(){
		
		for(ReactionCenter center:uniqueCenters){
			this.summary.append(center.getName()+", "+center.getNumberEncountered()+"\n");
		}
	}

	/**Generate the chemistry part of the reaction family. Returns a list with several elements of information. </br>
	 * - List(0) - String: The xml element that defines the xml file </br>
	 * - List(1) - Integer: The number of reactions in the input </br>
	 * - List(2) - Integer: The number of unique families that were generated from the input </br>
	 * - List(3) - Integer: The number of reactions for which the reaction mapping failed (did not give a complete mapping). </br>
	 * - List(4) - String: The names of the reactions for which the reaction mapping failed.</br>
	 * - List(5) - Integer: The number of reactions in the input that contained too many reactants.</br>
	 * - List(6) - Integer: The number of reverse reactions that have been encountered.</br>
	 * - List(7) - Integer: The number of identical reactions that have been encountered.</br>
	 *  
	 * @param output
	 * @param input
	 * @return list
	 */
	protected static List<Object> generate(Config config, 
										   String output, 
										   String input, 
										   boolean zeroKinetics, 
										   boolean constraint, 
										   boolean specific){
		
		List<Object> info			=	new ArrayList<Object>();
		ChemistryGenerator chemGen	=	new ChemistryGenerator(config,output,input);
		
		chemGen.config.setInpTemperature(chemGen.temperature);

		//String out 					=	TemperatureGenerator.generate(chemGen.temperature);
		
		for(int i = 0;	i < chemGen.reactionCount;	i++){
			
			String name	=	chemGen.names.get(i);	
			logger.info("\nStarting generation for reaction: "+name+" at "+Output.getTime());
			//Generate reaction centers and check if the same mechanism has not already been found
			boolean mappingSuccess	=	chemGen.determineReactionCenter(i,specific);
			
			if(mappingSuccess){
				int[] checkResults	=	chemGen.checkReactionCenter(chemGen.currentCenter);
				int alreadyExistsAs	=	checkResults[0];
				int reverseOf		=	checkResults[1];
				//if isn't present yet, generate corresponding  family block
				if(alreadyExistsAs == -1 && mappingSuccess){
					config.addInpReactionFamily(chemGen.generateFamily(i, zeroKinetics));
					logger.info("Reaction center of reaction \""+name+"\" is new, adding ...");
					if(chemGen.currentCenter.isReverse()){
						logger.info("Reaction \""+name+"\" has been detected to be the reverse of reaction \""
									+chemGen.uniqueCenters.get(reverseOf).getName()+"\". \"REVERSE\" kinetics assigned.");
						chemGen.reverse++;
					}
					try{
						chemGen.copyRXNFile(name,i,chemGen.getResonanceName(i));
					}catch(IOException e){}
				}
				//if present, go to next reaction and inform.
				else{
					logger.info("Reaction center of reaction \""+name+"\" is the same as the"+
								" reaction center of reaction \""+chemGen.uniqueCenters.get(alreadyExistsAs).getName()+"\", not adding...");
				}
			}
			//noFail means identical reaction, so skip if at true
			else if(!chemGen.noFail){
				chemGen.failed	=	chemGen.failed + name+"\n";
				chemGen.fails++;
			}
			//if was identical reaction, reset!
			else{
				chemGen.noFail	=	false;
			}
			
			logger.info("Ended generation for reaction: "+name+" at "+Output.getTime());
			
			logger.info("\nProcessed "+(i+1)+"/"+chemGen.reactionCount+" reactions."+
						"\n\t - Added "+chemGen.uniqueCenters.size()+" new reaction famil"+((chemGen.uniqueCenters.size() == 1)?"y":"ies")+
						"\n\t - Failed mapping for "+chemGen.fails+" reaction"+((chemGen.fails == 1)?"":"s")+
						"\n\t - Detected "+chemGen.identical+" identical reaction"+((chemGen.identical == 1)?"":"s")+
						"\n\t - Detected "+chemGen.reverse+" reverse reaction"+((chemGen.reverse == 1)?"":"s"));
				
		
			System.gc();		
		}
		
		if(constraint){
			InpConstraint prodConstraint	=	new InpConstraint();
			prodConstraint.setType(ConstraintType.HEAVYATOMCOUNT);
			prodConstraint.setLimit(LimitType.MAX);
			prodConstraint.setValue(chemGen.maxC+chemGen.maxO+chemGen.maxN+chemGen.maxS);
			config.addInpProductConstraint(prodConstraint);
		}
		
		chemGen.generateSummary();
		
		info.add(config);
		info.add(chemGen.reactionCount);
		info.add(chemGen.uniqueCenters.size());
		info.add(chemGen.fails);
		info.add(chemGen.failed);
		info.add(chemGen.input.tooManyReactantCount());
		info.add(chemGen.reverse);
		info.add(chemGen.identical);
		info.add(chemGen.summary);
		
		return info;
	}
}