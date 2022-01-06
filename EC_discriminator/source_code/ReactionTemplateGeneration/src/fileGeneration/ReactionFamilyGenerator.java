package fileGeneration;

import org.apache.log4j.Logger;

import fileGeneration.command.FileGenerationCommand;
import tools.Output;

/**This class executes the generation of a reaction family file for use with genesys
 * 
 * @author pplehier
 *
 */
public class ReactionFamilyGenerator {

	private static Logger logger	=	Logger.getLogger(ReactionFamilyGenerator.class);
	
	public static void main(String[] args){
		
		logger.info("Generating reaction families "+
				"\nStarted on: "+Output.getDate()+
				"\n"+Output.line("-",50));
		FileGenerationCommand fGC	=	new FileGenerationCommand(args);
		fGC.execute();
		logger.info(Output.line("-", 50)+
					"\nEnded generation of reaction families on "+Output.getDate());
	}
}