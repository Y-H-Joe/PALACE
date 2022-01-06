package fileGeneration.command;

import java.io.File;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.PosixParser;
import org.apache.commons.cli.ParseException;
import org.apache.log4j.Logger;

import constants.StringConstants;
import fileGeneration.ReactionFamilyGenerationHelper;
import fileGeneration.generators.FileGenerator;
import interfaces.ICommand;

/**This class interprets the command line for reaction family file generation
 * 
 * @author pplehier
 *
 */
public class FileGenerationCommand implements ICommand{

	private static Logger logger	=	Logger.getLogger(FileGenerationCommand.class);
	private CommandLine fileGenerationCommand;
	
	/**Constructor, parses the arguments of the command line
	 * 
	 * @param args
	 */
	public FileGenerationCommand(String[] args){
		
		FileGenerationOptions fGO	=	new FileGenerationOptions();
		CommandLineParser parser	=	new PosixParser();
		try{
			fileGenerationCommand	=	parser.parse(fGO.getOptions(),args,true);
		} catch(ParseException e){
			logger.fatal("Failed parsing the commands. Exiting ...");
			e.printStackTrace();
			System.exit(-1);
		}
	}
	
	/**Returns whether the command has a specified option.
	 * 
	 * @param option in char format
	 */
	public boolean hasOption(char opt){
		
		return this.fileGenerationCommand.hasOption(opt);
	}

	/**Returns whether the command has a specified option
	 * 
	 * @param option is String format
	 */
	public boolean hasOption(String opt){
		
		return this.fileGenerationCommand.hasOption(opt);
	}

	/**Returns the value of a specified option
	 * 
	 * @param option in char format
	 */
	public String getOption(char opt){
		
		return this.fileGenerationCommand.getOptionValue(opt);
	}

	/**Returns the value of a specified option
	 * 
	 * @param option in String format
	 */
	public String getOption(String opt){
		
		return this.fileGenerationCommand.getOptionValue(opt);
	}

	/**Checks whether required options are present in the command
	 * For file generation, the required options are either:<br>
	 * - help (-h)<br>
	 * - input (-i) and output (-o)<br>
	 * Checks whether specified inputs exist.
	 */
	private void checkOptions(){
		
		if(this.fileGenerationCommand.hasOption(FileGenerationOptions.help)){}
		else{
			if(!(this.fileGenerationCommand.hasOption(FileGenerationOptions.input)
			     &&
			     this.fileGenerationCommand.hasOption(FileGenerationOptions.output))){
				logger.fatal("Missing one of following options: -"+FileGenerationOptions.input+
															  "/-"+FileGenerationOptions.output);
				System.exit(-1);
			}
		}
		
		if(this.fileGenerationCommand.hasOption(FileGenerationOptions.input)){
			String input	=	this.fileGenerationCommand.getOptionValue(FileGenerationOptions.input);
			File inputF		=	new File(input);
			if(!inputF.exists()){
				logger.fatal("Specified input directory does not exist! Exiting ...");
				System.exit(-1);
			}
		}
	}

	/**Execute the command line
	 */
	public void execute(){
		
		checkOptions();
		
		if(this.hasOption(FileGenerationOptions.help)){
			ReactionFamilyGenerationHelper.printHelp();
		}
		
		FileGenerator generator	=	new FileGenerator(this.getOption(FileGenerationOptions.input),
				  									  this.getOption(FileGenerationOptions.output));
		if(this.hasOption(FileGenerationOptions.noOutputCheck)){
			generator.setOutputCheck(false);
		}
		else{
			generator.setOutputCheck(true);
		}
		
		if(this.hasOption(FileGenerationOptions.zeroKinetics)){
			generator.setZeroKinetics(true);
		}
		
		if(this.hasOption(FileGenerationOptions.noConstraint)){
			generator.setConstraint(false);
		}
		
		if(this.hasOption(FileGenerationOptions.outDir)){
			generator.setOutputDirectory(this.getOption(FileGenerationOptions.outDir));
		}
		else{
			generator.setOutputDirectory(StringConstants.OUTPUT);
		}
		
		if(this.hasOption(FileGenerationOptions.specific)){
			generator.setSpecific(true);
		}
		
		generator.run();
	}
}