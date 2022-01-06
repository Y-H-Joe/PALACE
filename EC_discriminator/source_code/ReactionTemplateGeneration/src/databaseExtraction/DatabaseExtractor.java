package databaseExtraction;

import org.apache.log4j.Logger;

import databaseExtraction.command.DatabaseExtractionCommand;
import tools.Output;

/**This class executes the extraction of reaction files from an on-line chemical database
 * 
 * @author pplehier
 *
 */
public class DatabaseExtractor {
	
	private static Logger logger	=	Logger.getLogger(DatabaseExtractor.class);
	private static int symbolCount	=	50;
	
	public static void main(String[] args){
		
		logger.info("Extracting database information.");
		logger.info(Output.line("-", symbolCount));
		
		DatabaseExtractionCommand dEC	=	new DatabaseExtractionCommand(args);
		dEC.execute();
		
		logger.info(Output.line("-", symbolCount));
		logger.info("Ended database extraction.");
	}
}