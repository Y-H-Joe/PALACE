package databaseExtraction.command;

import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;

import constants.StringConstants;
import interfaces.IOptions;

/**This class defines the possible options for the database extraction command
 * 
 * @author pplehier
 *
 */
public class DatabaseExtractionOptions implements IOptions{

	private static final String implementedDatabases	=	"KEGG/RMG";
	protected static final String KEGG	=	"KEGG";
	protected static final String RMG	=	"RMG";
	
	protected static final String output		=	"o";
	protected static final String database		=	"d";
	protected static final String retrieve		=	"r";
	protected static final String input			=	"i";
	protected static final String help			=	"h";
	protected static final String subDatabase	=	"s";
	protected static final String retrieveALL	=	StringConstants.ALL;
	protected static final String retrieveSINGLE=	"SINGLE";
	protected static final String retrieveRANGE	=	"RANGE";
	protected static final String retrieveFILE	=	"FILE";
	
	private Options databaseExtractionOptions;
	
	/**Constructor, create the options.
	 */
	public DatabaseExtractionOptions(){
		
		databaseExtractionOptions	=	this.createOptions();
	}
	
	/**Retrieve the generated options
	 * 
	 * @return Options
	 */
	public Options getOptions(){
		return databaseExtractionOptions;
	}
	
	/**Create a new set of options
	 * 
	 * @return Options
	 */
	private Options createOptions(){
		
		Options dbExtractionOptions	=	new Options();
		
		Option database		=	new Option("d","database",true,("Database name - one of following: "+implementedDatabases));
		dbExtractionOptions.addOption(database);
		
		Option outputDir	=	new Option("o","output",true,"Name of the output directory. If not specified, defaults to [database name]");
		dbExtractionOptions.addOption(outputDir);
		
		Option retrieve		=	new Option("r","retrieve",true,"Which entries should be retrieved. Options are:"+
										   "\n\t"+DatabaseExtractionOptions.retrieveALL+": retrieve all entries"+
										   "\n\t"+DatabaseExtractionOptions.retrieveSINGLE+": retrieve a single entry"+
										   "\n\t"+DatabaseExtractionOptions.retrieveRANGE+": retrieve entries within a range"+
										   "\n\t"+DatabaseExtractionOptions.retrieveFILE+": column file containing the numbers of the entries to be retrieved");
		dbExtractionOptions.addOption(retrieve);
		
		Option input		=	new Option("i","input",true,"Input for retrieval. Argument type varies depending on -r:"+
										   "\n\t"+DatabaseExtractionOptions.retrieveALL+": ALL"+
										   "\n\t"+DatabaseExtractionOptions.retrieveSINGLE+": one number"+
										   "\n\t"+DatabaseExtractionOptions.retrieveRANGE+": two numbers, separated by \"-\""+
				   						 	"\n\t"+DatabaseExtractionOptions.retrieveFILE+": a file name");
		dbExtractionOptions.addOption(input);
		
		Option subDatabase	=	new Option("s","sub_database",true,"Specify which sub database should be retrieved. This option is "+
										   "not necessary for the KEGG database, nor when the retrieval type is ALL. If the retrieval "+
										   "type is ALL and a sub database is specified, all entries of this sub database will be processed"+
										   "\nThe argument is either the full name of the subdatabase, or the number in "+
										   "the list (as listed on the site.");
		dbExtractionOptions.addOption(subDatabase);
		
		Option help			=	new Option("h","help",false,"Help page for command line usage");
		dbExtractionOptions.addOption(help);
		
		return dbExtractionOptions;	
	}
}