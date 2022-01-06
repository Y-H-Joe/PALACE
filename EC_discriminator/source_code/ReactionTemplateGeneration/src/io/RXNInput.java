package io;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.openscience.cdk.exception.NoSuchAtomTypeException;
import org.openscience.cdk.interfaces.IReaction;

import interfaces.IInput;
import io.readers.RXNFileReader;
import manipulators.ReactionBuilder;

/**This class defines the input type using a folder of .rxn files as input
 * 
 * @author pplehier
 *
 */
public class RXNInput implements IInput{
	
	private static final String[] kineticsDefault	=	{"Kinetics","GROUP_ADDITIVITY","USER","0","0"};
	
	private File folder;
	private File[] filesList;
	private static Logger logger=Logger.getLogger(RXNInput.class);
	private List<IReaction> rxnInput;
	private List<String> nameInput;
	private List<Boolean> bimolInput;
	private List<String[]> kinetics;
	private double temperature;
	private int inputCount;
	private static int uniOrBimolCounter	=	0;
	
	public RXNInput(String InputPath){
	
		this.folder		=	new File(InputPath);
		//if it is a directory, take all rxn files
		if(this.folder.isDirectory()){
			this.filesList	=	folder.listFiles();
		}
		//if not a directory, should be single .rxn file
		else{
			this.filesList		=	new File[1];
			this.filesList[0]	=	folder;
		}
		
		rxnInput		=	new ArrayList<IReaction>();
		nameInput		=	new ArrayList<String>();
		bimolInput		=	new ArrayList<Boolean>();
		
		this.processFiles();	
		
		this.kinetics	=	defaultKinetics();
		this.temperature=	defaultTemperature();
	}
	
	/**Process all files, read each file and generate the corresponding CDK entities and required info.
	 * Run a fix for aromatic components to ensure compatibility with RDT.
	 */
	private void processFiles(){
		
		for(File f:filesList){
			logger.info("Reading file: "+f.getName());
			inputCount++;
			
			RXNFileReader rfr	=	new RXNFileReader(f.getAbsolutePath());
			IReaction rxn		=	rfr.toReaction(true);
			boolean uniOrBimol	=	true;
			//If no mapping: process radicals and aromaticity.
			if(rxn.getMappingCount() == 0){
				rxn	=	ReactionBuilder.build(rxn);
			}
			//If mappings are present: files originate from mapping run: aromaticity is ok, won't contain
			//radicals, but potentially dummies.
			//No longer necessary, fixed in RDT.
			/*else{
				ReactionBuilder.revertRadicals(rxn);
			}*/
			try {
				
				if(!rxn.checkAtomBalance()){
					logger.fatal("Reaction no. "+inputCount+" is not in atom balance. Exiting ...");
					System.exit(-1);
				}
				
			} catch (NoSuchAtomTypeException e) {
				e.printStackTrace();
			}

			rxnInput.add(rxn);
			nameInput.add(f.getName().substring(0, f.getName().length()-4));

			switch(rxn.getReactantCount()){
			case 0:		logger.fatal("No reactants found for reaction "+f.getName());
			System.exit(-1);
			case 1:		bimolInput.add(false);
			break;
			case 2:		bimolInput.add(true);
			break;
			default:{	logger.fatal("Too many reactants found for reaction "+f.getName()+". Not processing ...");
				bimolInput.add(true);
				uniOrBimol	=	false;
				uniOrBimolCounter++;
			//System.exit(-1);
			}
			}	
			
			if(!uniOrBimol){
				inputCount--;
				bimolInput.remove(bimolInput.size()-1);
				nameInput.remove(nameInput.size()-1);
				rxnInput.remove(rxnInput.size()-1);
			}
		}
	}
	
	public List<String[]> getKinetics(){
		
		return this.kinetics;
	}
	
	/**Get the reactions in the input
	 * 
	 * @return reactions
	 */
	public List<IReaction> getReactions(){
		
		return rxnInput;
	}
	
	/**Get the names of the reactions (the names of the rxn files)
	 * 
	 * @return names
	 */
	public List<String> getNames(){
		
		return nameInput;
	}
	
	/**Returns the bimolecular property of all reactions
	 * 
	 * @return ?bimolecular
	 */
	public List<Boolean> getBimol(){
		
		return bimolInput;
	}
	
	/**Get the number of processed reactions
	 * 
	 * @return number of reactions
	 */
	public int getInputCount(){
		
		return inputCount;
	}
	
	/**Get the temperature
	 * 
	 * @return temperature
	 */
	public double getTemperature(){
		
		return this.temperature;
	}
	
	/**Get the constraints. For a .rxn type input, a constraint cannot be defined,
	 * so returns null
	 * 
	 * @return null
	 */
	public List<String[]> getConstraints(){
		
		return null;
	}
	
	public int tooManyReactantCount(){
		
		return uniOrBimolCounter;
	}
	
	public String getInputPath(){
		return folder.getPath();
	}
	/**Sets the kinetics to default:<br>
	 * Group additivity block with file path set to USER = to be filled in by user after processing.
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
}