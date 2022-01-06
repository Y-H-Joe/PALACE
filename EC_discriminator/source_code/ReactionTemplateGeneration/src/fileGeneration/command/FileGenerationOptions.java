package fileGeneration.command;

import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;

import interfaces.IOptions;

/**This class defines the possible options for the reaction family file generation command
 * 
 * @author pplehier
 *
 */
public class FileGenerationOptions implements IOptions{

	protected static final String output		=	"o";
	protected static final String input			=	"i";
	protected static final String help			=	"h";
	protected static final String outDir		=	"oD";
	protected static final String noOutputCheck	=	"oC";
	protected static final String zeroKinetics	=	"zK";
	protected static final String noConstraint	=	"nC";
	protected static final String specific		=	"s";
	
	private Options fileGenerationOptions;

	/**Constructor, create the options.
	 */
	public FileGenerationOptions(){
		
		fileGenerationOptions	=	this.createOptions();
	}
	
	/**Retrieve the generated options
	 * 
	 * @return Options
	 */
	public Options getOptions(){
		
		return this.fileGenerationOptions;
	}

	/**Create a new set of options
	 * 
	 * @return Options
	 */
	private Options createOptions(){
		
		Options fgOptions	=	new Options();
		Option input		=	new Option("i","input",true,"Name of the file or directory from which the input should be read."+
															 "\nA directory is expected to contain only .rxn files of the reactions that"+
															 "\nare to be processed.");
		fgOptions.addOption(input);
		
		Option output		=	new Option("o","input",true,"Name of the file to which the reaction families should be written."+
															"The file is automatically written to an output folder with the name \"outputFileGen\"");
		fgOptions.addOption(output);
		
		Option outputDir	=	new Option("oD","output directory",true,"Nameof the output directory, to which all reaction "+
																		"directories will be written.");
		fgOptions.addOption(outputDir);
		
		Option noOutputCheck=	new Option("oC","no output check", false, "If specified, the existence of the demanded output directory will not be "+
																		  "verified. Risk of losing data.");
		
		fgOptions.addOption(noOutputCheck);
		
		Option zeroKinetics	=	new Option("zK", "zero kinetics", false, "Will generate an arrhenius block for the kinetics with all parameters set to 0.");
	
		fgOptions.addOption(zeroKinetics);
		
		Option noConstraint	=	new Option("nC", "no constraints", false, "No product constraints on heavy atom count will be added.");
	
		fgOptions.addOption(noConstraint);
		
		Option specific		=	new Option("s", " specific", false, "Increase specificity by adding connected hetero atoms to the reaction center.");
		
		fgOptions.addOption(specific);
		
		Option help			=	new Option("h","help",false,"Help page for command line usage");
		
		fgOptions.addOption(help);
		
		return fgOptions;
	}
}