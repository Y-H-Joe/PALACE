package io;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.openscience.cdk.interfaces.IReaction;

import interfaces.IInput;
import io.readers.TXTFileReader;
import manipulators.ReactionBuilder;
import tools.TxtReaction;

/**This class defines an input type using a single .txt file as input. The .txt file has following structure:<br>
 * Temperature \t temperature<br>
 * Name \t name of the first reaction<br>
 * Reactants \t smilesReactant1( \t smilesReactant2) [second reactant is optional]<br>
 * Products \t smilesProduct1 \t ... \t smilesProductN<br>
 * Constraints \t constraint 1 \t constraint2 \t ...<br>
 * Kinetics \t kinetics format<br>
 * 
 * @author pplehier
 *
 */
public class TextInput implements IInput{
	
	public String filepath;
	private static Logger logger=Logger.getLogger(TextInput.class);
	private List<String> nameInput;
	private List<Boolean> bimolInput;
	private List<String[]> reactInput;
	private List<String[]> prodInput;
	private List<String[]> kinetInput;
	private List<String[]> constrInput; 
	private List<IReaction> rxnInput;
	private int inputCount;
	private double temperature;
	private TXTFileReader tfr;
	
	public TextInput(String path){
		
		this.filepath	=	path;
		nameInput		=	new ArrayList<String>();
		bimolInput		=	new ArrayList<Boolean>();
		reactInput		=	new ArrayList<String[]>();
		prodInput		=	new ArrayList<String[]>();
		kinetInput		=	new ArrayList<String[]>();
		constrInput		=	new ArrayList<String[]>();
		rxnInput		=	new ArrayList<IReaction>();
		tfr				=	new TXTFileReader(this.filepath);
		this.processFile();	
	}
	
	/**Process the input file: read the file and generate the CDK entities and info for genesys
	 */
	private void processFile(){
		
		int linenumber	=	2;
		this.processTemperature();
		
		while(linenumber < tfr.countLines()){
			String nameInp	=	tfr.read(linenumber,"Name").split("\t")[1];	
			nameInput.add(nameInp);
			inputCount++;
			String[] initialReactInp	=	tfr.read(linenumber,"Reactants").split("\t");
			String[] reactInp			=	new String[initialReactInp.length-1];
		
			for (int i = 0;	i < reactInp.length;	i++){
				reactInp[i]	=	initialReactInp[i+1];
			}
		
			if(reactInp.length == 1){
				bimolInput.add(false);
				reactInput.add(reactInp);
				
			}else if(reactInp.length == 2){
				bimolInput.add(true);
				reactInput.add(reactInp);
				
			} else {
				bimolInput.add(false);
				reactInput.add(null);
				logger.fatal("No or too many reactants specified.");
				System.exit(-1);
			}
		
			String[] initialProdInp	=	tfr.read(linenumber,"Products").split("\t");
			String[] prodInp		=	new String[initialProdInp.length-1];
		
			for (int i = 0;	i < prodInp.length;	i++){
				prodInp[i]	=	initialProdInp[i+1];
			}
		
			prodInput.add(prodInp);
			String[] ConstrInp	=	tfr.read(linenumber,"Constraints").split("\t");
			constrInput.add(ConstrInp);
			String[] KinInp		=	tfr.read(linenumber,"Kinetics").split("\t");
			
			//Look at possible keywords in kinetics and check if the correct additional data are given
			this.check(KinInp);
			linenumber			=	tfr.getLineNumberNextKey(linenumber, "Name");	
			TxtReaction react	=	new TxtReaction(reactInp,prodInp);
			
			//Correctly build the reaction from the read smiles to maintain compatibility
			rxnInput.add(ReactionBuilder.build(react.getSmiles()));

			//Generation of block for a reaction not in atom balance will continue, may lead to incorrect/missing
			//transformations
			if(!react.checkAtomBalance()){
				logger.fatal("Reaction no. "+inputCount+" is not in atom balance. Exiting ...");
				System.exit(-1);
			}
		}
	}
	
	/**Read the temperature<br>
	 * Temperature should have a unit, and will be converted to K (warning is issued on conversion)<br>
	 * If no unit specified (or other than K, C, F, °C or °F error and exit)
	 */
	private void processTemperature(){
		
		String[] TempInp	=	tfr.read(1,"Temperature").split("\t");
		int count			=	TempInp.length;
		
		if(count > 2){
			logger.fatal("Too many arguments in input.");
			System.exit(-1);
		}
		
		else if(TempInp[count-1].endsWith("K")) {
			temperature	=	Double.parseDouble(TempInp[count-1].substring(0,TempInp[count-1].indexOf("K")));
		}
		
		else if(TempInp[count-1].endsWith("C")){
			temperature	=	Double.parseDouble(TempInp[count-1].substring(0,TempInp[count-1].indexOf("C")))+273.15;
			logger.warn("Input temperature thought to be in Celcius -> Converted to Kelvin.");
		}
		
		else if(TempInp[count-1].endsWith("°C")){
			temperature	=	Double.parseDouble(TempInp[count-1].substring(0,TempInp[count-1].indexOf("°")))+273.15;
			logger.warn("Input temperature thought to be in Celcius -> Converted to Kelvin.");
		}
		
		else if(TempInp[count-1].endsWith("°F")){
			temperature	=	5.0/9.0*(Double.parseDouble(TempInp[count-1].substring(0,TempInp[count-1].indexOf("°")))+459.67);
			logger.warn("Input temperature thought to be in Fahrenheit -> Converted to Kelvin.");
		}
		
		else if(TempInp[count-1].endsWith("F")){
			temperature	=	5.0/9.0*(Double.parseDouble(TempInp[count-1].substring(0,TempInp[count-1].indexOf("F")))+459.67);
			logger.warn("Input temperature thought to be in Fahrenheit -> Converted to Kelvin.");
		}
		
		else{
			logger.fatal("Wrong characters on line \"Temperature\"");
			System.exit(-1);
		}
	}
	
	/**Check the validity of the kinetics input line, as this is a required input in Genesys.<br>
	 * TODO: Check when changing kinetics input!
	 * 
	 * @param KinInp line 
	 */
	private void check(String [] KinInp){
		
		if(KinInp[1].equals("ARRHENIUS")||KinInp[1].equals("GROUP_ADDITIVITY")||KinInp[1].equals("EVANS_POLANYI")||KinInp[1].equals("BLOWERS_MASEL")||KinInp[1].equals("REVERSE")||KinInp[1].equals("AB_INITIO")||KinInp[1].equals("TEMP_INDEPENDENT_RATE")||KinInp[1].equals("LIBRARY")){
			kinetInput.add(KinInp);
		}else{
			logger.fatal("Wrong kinetics format.");
			System.exit(-1);
		}
	}	
	
	/**Retrieve the read Names form the input file
	 * 
	 * @return names
	 */
	public List<String> getNames(){
		return this.nameInput;
	}
	
	/**Retrieve the processed read reactions from the input file
	 * 
	 * @return processed reactions
	 */
	public List<IReaction> getReactions(){
		return this.rxnInput;
	}
	
	/**Retrieve the bimolecular property of the read reactions from the input file
	 * 
	 * @return bimolecular
	 */
	public List<Boolean> getBimol(){
		return this.bimolInput;
	}
	
	/**Retrieve the read kinetics from the input file
	 * 
	 * @return kinetics
	 */
	public List<String[]> getKinetics(){
		return this.kinetInput;
	}
	
	public String getInputPath(){
		return this.filepath;
	}
	
	/**Retrieve the read constraints from the input file
	 * 
	 * @return constraints
	 */
	public List<String[]> getConstraints(){
		return this.constrInput;
	}
	
	/**Get the number of read reaction from the input file
	 * 
	 * @return number of reactions
	 */
	public int getInputCount(){
		return inputCount;
	}
	
	/**Get the temperature specified in the input file
	 * 
	 *@return temperature
	 */
	public double getTemperature(){
		return temperature;
	}

	@Override
	public int tooManyReactantCount() {
		// TODO Auto-generated method stub
		return 0;
	}
}