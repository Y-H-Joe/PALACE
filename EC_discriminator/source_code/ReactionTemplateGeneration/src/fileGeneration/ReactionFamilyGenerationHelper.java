package fileGeneration;

import java.util.Collection;

import org.apache.commons.cli.Option;
import org.apache.log4j.Logger;

import constants.StringConstants;
import fileGeneration.command.FileGenerationOptions;
import interfaces.IHelper;
import tools.Output;

public class ReactionFamilyGenerationHelper implements IHelper{

private static Logger logger	=	Logger.getLogger(ReactionFamilyGenerationHelper.class);
	
	/**Print a help page for the file generation tool.
	 */
	@SuppressWarnings("unchecked")
	public static void printHelp(){
		
		FileGenerationOptions options	=	new FileGenerationOptions();
		Collection<Option> optionsList			=	options.getOptions().getOptions();
		
		logger.info("Displaying help for reaction family generation.\n"+Output.line("-", 25));
		
		for(Option opt:optionsList){
			logger.info(opt.getOpt()+" or --"+opt.getLongOpt()+": "+opt.getDescription());
		}
		
		example();
		
		logger.info("End of help for reaction family generation.\n"+Output.line("-", 25));
	}
	
	/**Print an example of the command line usage for the file generation tool.
	 */
	private static void example(){
		
		logger.info("The options are illustrated with the following examples.");
		
		logger.info("The command: \"-i Reactions.txt -o ReactionFamilies.xml\" will read the reactions from the .txt file and"+
				 	"(by default) write the reaction families to $WorkingDir"+StringConstants.OUTPUT+"ReactionFamilies.xml");
		logger.info("The command: \"-i Reactions -o families\" will read all reactions (.rxn) from the folder Reactions and"+
					"(by default) write the reaction families to $WorkingDir"+StringConstants.OUTPUT+"families.xml."+
					"\nNote that the xml extension is automatically added. If a different file extension than .xml is provided,"+
					"it will be overruled to .xml.");
		logger.info("The command: \"-i Reactions -o families -oD ReactionFamilies will read all reactions from the folder Reactions"+
					"and write the reaction families to ReactionFamilies\\families.");
	}
}