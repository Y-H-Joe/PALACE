package databaseExtraction;


import java.util.Collection;

import org.apache.commons.cli.Option;
import org.apache.log4j.Logger;

import databaseExtraction.command.DatabaseExtractionOptions;
import interfaces.IHelper;
import tools.Output;

public class DatabaseExtractionHelper implements IHelper{

	private static Logger logger	=	Logger.getLogger(DatabaseExtractionHelper.class);
	
	/**Print a help page for the database extraction tool.
	 */
	@SuppressWarnings("unchecked")
	public static void printHelp(){
		
		DatabaseExtractionOptions options	=	new DatabaseExtractionOptions();
		Collection<Option> optionsList			=	options.getOptions().getOptions();
		
		logger.info("Displaying help for database extraction.\n"+Output.line("-", 25));
		
		for(Option opt:optionsList){
			logger.info(opt.getOpt()+" or --"+opt.getLongOpt()+": "+opt.getDescription());
		}
		
		example();
		
		logger.info("End of help for database extraction.\n"+Output.line("-", 25));
	}
	
	/**Print an example of the command line usage for the database extraction tool.
	 */
	private static void example(){
		
		logger.info("The options are illustrated with the following examples.");
		
		logger.info("The command: \"-d KEGG -r ALL -i ALL\" will extract all available reactions from the KEGG"+
					"database and write the to .rxn files in the default folder \"/KEGG/\".");
		
		logger.info("The command: \"-d KEGG -r RANGE -i 4-25 -o keggdatabase\" will extract all available entries"+
					"between 4 and 25 and write them to .rxn file in the folder $WorkingDir/keggdatabase.");
	}
}