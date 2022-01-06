package databaseExtraction.command;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.log4j.Logger;

import constants.StringConstants;
import databaseExtraction.DatabaseExtractionHelper;
import databaseExtraction.databases.KEGG;
import databaseExtraction.databases.RMG;
import interfaces.ICommand;
import interfaces.IDatabase;
import io.readers.RMGReader;
import io.readers.TXTFileReader;
import tools.Output;

/**This class interprets the commands for the database extraction
 * 
 * @author pplehier
 *
 */
public class DatabaseExtractionCommand implements ICommand{

	private static Logger logger	=	Logger.getLogger(DatabaseExtractionCommand.class);
	private CommandLine databaseExtractionCommand;
	private IDatabase database;
	
	/**Constructor, parses the arguments of the command line
	 * 
	 * @param args
	 */
	public DatabaseExtractionCommand(String[] args){
		
		DatabaseExtractionOptions	dEO	=	new DatabaseExtractionOptions();
		CommandLineParser parser		=	new PosixParser();
		
		try {
			databaseExtractionCommand		=	parser.parse(dEO.getOptions(),args,true);
		} 
		catch (ParseException e) {
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
		
		return this.databaseExtractionCommand.hasOption(opt);
	}
	
	/**Returns whether the command has a specified option
	 * 
	 * @param option is String format
	 */
	public boolean hasOption(String opt){
		
		return this.databaseExtractionCommand.hasOption(opt);
	}
	
	/**Returns the value of a specified option
	 * 
	 * @param option in char format
	 */
	public String getOption(char opt){
		
		return this.databaseExtractionCommand.getOptionValue(opt);
	}
	
	/**Returns the value of a specified option
	 * 
	 * @param option in String format
	 */
	public String getOption(String opt){
		
		return this.databaseExtractionCommand.getOptionValue(opt);
	}
	
	/**Checks whether required options are present in the command
	 * For database extraction, the required options are either:<br>
	 * - help (-h)<br>
	 * - database (-d) and input (-i) and retrieval (-r)<br>
	 * - output (-o) is optional for KEGG (sub database is not required or used) <br>
	 * - output (-o) is optional for RMG, sub database (-s) is required unless retrieval type is ALL<br>
	 */
	private void checkOptions(){
		
		if(this.databaseExtractionCommand.hasOption(DatabaseExtractionOptions.help)){}
		else if(!(this.databaseExtractionCommand.hasOption(DatabaseExtractionOptions.database)
				 &&
				 this.databaseExtractionCommand.hasOption(DatabaseExtractionOptions.input)
				 &&
				 this.databaseExtractionCommand.hasOption(DatabaseExtractionOptions.retrieve))){
			logger.fatal("Missing one of following options: -"+DatabaseExtractionOptions.database+
															 "/-"+DatabaseExtractionOptions.retrieve+
															 "/-"+DatabaseExtractionOptions.input+
															 "\nTry -h for help.");
			System.exit(-1);
		}
		
		else if(this.databaseExtractionCommand.hasOption(DatabaseExtractionOptions.RMG) 
				&& 
				!this.databaseExtractionCommand.getOptionValue(DatabaseExtractionOptions.retrieve).equals(DatabaseExtractionOptions.retrieveALL)
				&&
				!this.databaseExtractionCommand.hasOption(DatabaseExtractionOptions.subDatabase)){
			logger.fatal("Missing necessary subdatabase specification for extraction from RMG.");
			System.exit(-1);
		}
	}
	
	/**Interprets the input option. As several retrieval methods are possible, the format of the input can 
	 * vary according to the retrieval type. <br>
	 * For "All": returns "ALL"<br>
	 * For "RANGE": returns an array containing min and max <br>
	 * For "SINGLE": returns an integer<br>
	 * For "FILE": returns an array with all entries of the file
	 *  
	 * @return Object
	 */
	private Object interpretRetrieve(){
		
		String retrieve	=	this.getOption(DatabaseExtractionOptions.retrieve);
		
		if(retrieve.equalsIgnoreCase(DatabaseExtractionOptions.retrieveALL)){
			return retrieve;
		}
		
		else if(retrieve.equalsIgnoreCase(DatabaseExtractionOptions.retrieveSINGLE)){
			try{
				return Integer.parseInt(this.getOption(DatabaseExtractionOptions.input));
			}
			
			catch(NumberFormatException e){
				logger.fatal("Non-numeric input specified for retrieval type \""+DatabaseExtractionOptions.retrieveSINGLE+"\". Exiting ..");
				System.exit(-1);
			}
		}
		
		else if(retrieve.equalsIgnoreCase(DatabaseExtractionOptions.retrieveRANGE)){
			String[] range	=	this.getOption(DatabaseExtractionOptions.input).split("-");
			
			if(range.length != 2){
				logger.fatal("Invalid input specified for retrieval type \""+DatabaseExtractionOptions.retrieveRANGE+"\". Exiting ...");
				System.exit(-1);
			}
			int[] limits	=	new int[2];
			
			try{
				limits[0]		=	Integer.parseInt(range[0]);
				limits[1]		=	Integer.parseInt(range[1]);
			}
			
			catch(NumberFormatException e){
				logger.fatal("Non-numeric input specified for retrieval type \""+DatabaseExtractionOptions.retrieveRANGE+"\". Exiting ...");
				System.exit(-1);
			}
			
			return limits;
		}
		
		else if(retrieve.equalsIgnoreCase(DatabaseExtractionOptions.retrieveFILE)){
			TXTFileReader reader	=	new TXTFileReader(this.getOption(DatabaseExtractionOptions.input));
			String[] entryString	=	reader.readArray();
			int[] entriesList		=	new int[entryString.length];
			
			for(int i = 0;	i < entriesList.length;	i++){
				try{
					entriesList[i]		=	Integer.parseInt(entryString[i]);
				}
				catch(NumberFormatException e){
					logger.fatal("Non-numeric input detected on line "+i+" of the input file. Exiting ...");
					System.exit(-1);
				}
			}
			
			return entriesList;
		}
		
		else{
			logger.fatal("Invalid retrieval type specified. Exiting ... ");
			System.exit(-1);
		}
		return null;
	}
	
	/**Set the type of database. Straight-forward for KEGG as no arguments are required. For RMG, if a sub database is
	 * specified, the database should be limited to this sub database. Otherwise, the full RMG database is constructed. 
	 */
	private void setDatabase(){
		
		String database	=	this.getOption(DatabaseExtractionOptions.database);
		
		if(database.equalsIgnoreCase(DatabaseExtractionOptions.KEGG)){
			this.database	=	new KEGG();
		}
		else if(database.equalsIgnoreCase(DatabaseExtractionOptions.RMG)){
			if(this.databaseExtractionCommand.hasOption(DatabaseExtractionOptions.subDatabase)){
				this.database	=	new RMG(this.databaseExtractionCommand.getOptionValue(DatabaseExtractionOptions.subDatabase));
			}
			else if(this.databaseExtractionCommand.getOptionValue(DatabaseExtractionOptions.retrieve)
												  .equals(DatabaseExtractionOptions.retrieveALL)){
				this.database	=	new RMG(StringConstants.ALL);
			}
			else{
				logger.fatal("The command must specify either a subdatabase, or ALL (or both). Exiting ...");
				System.exit(-1);
			}
	
		}
		else{
		 logger.fatal("Invalid database name given. Exiting ..."); 
		 System.exit(-1);
		}
	}
	
	/**Execute the command line, depending on the arguments. These determine which type of retrieve method must be 
	 * used.
	 */
	public void execute(){

		checkOptions();
		
		if(this.hasOption(DatabaseExtractionOptions.help)){
			DatabaseExtractionHelper.printHelp();
		}
		
		setDatabase();
		
		if(this.hasOption(DatabaseExtractionOptions.output)){
			this.database.setOutputDir(this.getOption(DatabaseExtractionOptions.output));
		}

		Object retrieve	=	this.interpretRetrieve();
		
		if((retrieve instanceof String)){
			if(this.getOption(DatabaseExtractionOptions.retrieve).equalsIgnoreCase(DatabaseExtractionOptions.retrieveALL)){
				//if retrieve resolves to a String, it should be "ALL". In that case, there are two possibilities:
				//Either a sub database was specified, in which case all entries of the sub database alone should be read
				//No sub database was specified, in which case all entries of all sub databases should be processed
				//This is taken into account for correct comment
				if(this.hasOption(DatabaseExtractionOptions.subDatabase)){
					logger.info("Retrieving entire available "+getLibName(this.getOption(DatabaseExtractionOptions.subDatabase))+" sub-database, writing to "+
								(this.hasOption(DatabaseExtractionOptions.output)?this.getOption(DatabaseExtractionOptions.output):this.database.defaultOutput()));
				}
				else{
					logger.info("Retrieving entire available "+this.database.getName()+" database, writing to "+
								(this.hasOption(DatabaseExtractionOptions.output)?this.getOption(DatabaseExtractionOptions.output):this.database.defaultOutput()));
				}
				
				logger.info("Start at "+Output.getDate());
				
				this.database.retrieve();
				
				logger.info("Ended retrieving entries");
				logger.info("Ended at "+Output.getDate());
			}
			else{
				logger.fatal("Incorrect combination of retrieval type and retrieval input. Try -h for usage help.");
				System.exit(-1);
			}
		}
		
		else if((retrieve instanceof Integer)){ 
			if(this.getOption(DatabaseExtractionOptions.retrieve).equalsIgnoreCase(DatabaseExtractionOptions.retrieveSINGLE)){
				
				if(this.hasOption(DatabaseExtractionOptions.subDatabase)){
					logger.info("Retrieving entry "+retrieve+" form the " +getLibName(this.getOption(DatabaseExtractionOptions.subDatabase))+" sub-database, writing to "+
								(this.hasOption(DatabaseExtractionOptions.output)?this.getOption(DatabaseExtractionOptions.output):this.database.defaultOutput()));
				}
				else{
					logger.info("Retrieving entry "+retrieve+" form the "+this.database.getName()+" database, writing to "+
								(this.hasOption(DatabaseExtractionOptions.output)?this.getOption(DatabaseExtractionOptions.output):this.database.defaultOutput()));
				}
				
				logger.info("Start at "+Output.getDate());
				
				this.database.retrieve((int) retrieve);
				
				logger.info("Ended retrieving entry");
				logger.info("Ended at "+Output.getDate());
			}
			else{
				logger.fatal("Incorrect combination of retrieval type and retrieval input. Try -h for usage help.");
				System.exit(-1);
			}
		}
		
		else if(retrieve instanceof int[]){
			int[] entries	=	(int[])retrieve;
			
			if(this.getOption(DatabaseExtractionOptions.retrieve).equalsIgnoreCase(DatabaseExtractionOptions.retrieveRANGE)){
				int lower	=	entries[0];
				int upper	=	entries[1];
				//switch if entered in wrong order.
				if(lower>upper){
					int temp	=	lower;
					lower		=	upper;
					upper		=	temp;
				}

				logger.info("Retrieving all entries between "+lower+" and "+upper+" form the "+this.database.getName()+" database, writing to "+
							(this.hasOption(DatabaseExtractionOptions.output)?this.getOption(DatabaseExtractionOptions.output):this.database.defaultOutput()));
				logger.info("Start at "+Output.getDate());
				
				this.database.retrieve(lower,upper);
				
				logger.info("Ended retrieving entries");
				logger.info("Ended at "+Output.getDate());
			}
			
			else if(this.getOption(DatabaseExtractionOptions.retrieve).equalsIgnoreCase(DatabaseExtractionOptions.retrieveFILE)){
				
				logger.info("Retrieving entries by file "+this.getOption(DatabaseExtractionOptions.input)+" form the "+this.database.getName()+" database, writing to "+
							(this.hasOption(DatabaseExtractionOptions.output)?this.getOption(DatabaseExtractionOptions.output):this.database.defaultOutput()));
				logger.info("Start at "+Output.getDate());
				
				this.database.retrieve(entries);
				
				logger.info("Ended retrieving entries");
				logger.info("Ended at "+Output.getDate());
			}
				
			else{
				logger.fatal("Incorrect combination of retrieval type and retrieval input. Try -h for usage help.");
				System.exit(-1);
			}
		}
		
		else{
			logger.fatal("Invalid type for input");
			System.exit(-1);
		}
	}
	
	private String getLibName(String arg){
		
		int indexLib;
		
		try{
			indexLib	=	Integer.parseInt(arg);
			return RMGReader.getLibraryNames()[indexLib];
		}
		catch(NumberFormatException notAnInteger){
			return arg;
		}
	}
}