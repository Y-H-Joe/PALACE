package io;

import org.apache.log4j.Logger;

import constants.StringConstants;
import interfaces.IDatabaseReader;
import interfaces.IEntry;
import io.readers.KEGGReader;
import tools.Output;

/**This class defines a single entry in the KEGG database and allows extraction of a .rxn file for that entry
 * 
 * @author pplehier
 *
 */
public class KEGGEntry implements IEntry{

	private static Logger logger		=	Logger.getLogger(KEGGEntry.class);
	private static final String RXN		=	"$RXN\n";
	private static final String MOL		=	"\n$MOL\n";
	private static final String header	=	"\n  databaseExtraction_rxn\t\t\t" + Output.getDate() + "\n";
	
	private IDatabaseReader KR;
	private int reactionNumber;
	private int[] componentCoefs;
	private String[] componentMolFiles;
	private String reactionFile;
	private boolean exists;
	private boolean unbalanced	=	false;
	
	/**Create new entry and read it from database
	 * 
	 * @param number
	 */
	public KEGGEntry(int number){
		
		this.reactionNumber	=	number;
		
		KR		=	new KEGGReader(this.reactionID());
		exists	=	KR.read();
		
		if(exists){
			componentCoefs		=	KR.getComponentCoefs();
			componentMolFiles	=	KR.getComponentMolFiles();
		
			this.createRXNFile();
			logger.info("Retrieved KEGG entry with ID "+getID());
		}
		else{
			this.reactionFile	=	StringConstants.EMPTY;
		}
	}
	
	/**Check whether the entry has data in the database
	 * 
	 * @return exists
	 */
	public boolean exists(){
		return exists;
	}
	
	/**Get the interpreted database information in .rxn format
	 * 
	 * @return String of rxn file
	 */
	public String getRXNFile(){
		
		return reactionFile;
	}
	
	/**Get the ID of the entry in the database
	 *
	 *@return ID
	 */
	public String getID(){
		
		return reactionID();
	}
	
	/**Create a .rxn file using the available .mol files in the database
	 * TODO: simplify coefficients.
	 */
	private void createRXNFile(){
		
		String rxnFileString	=	RXN+header;
		
		int reactantCount		=	KR.getReactantCount();
		int productCount		=	KR.getProductCount();
		rxnFileString			=	rxnFileString + "\n  " + reactantCount + "  " + productCount;
		
		for(int i = 0;	i < componentCoefs.length;	i++){
			for(int j = 0;	j < componentCoefs[i];	j++){
				if(!componentMolFiles[i].equals(StringConstants.EMPTY)){
					rxnFileString	=	rxnFileString + MOL + componentMolFiles[i];
				}
			}
		}
		
		this.reactionFile	=	rxnFileString;
	}
	
	/**Generate the ID based on the number
	 * 
	 * @return ID
	 */
	private String reactionID(){
		
		if (reactionNumber < 10){
			return "R0000"+reactionNumber;
			
		}else if(reactionNumber < 100){
			return "R000"+reactionNumber;
			
		}else if(reactionNumber < 1000){
			return "R00"+reactionNumber;
			
		}else if(reactionNumber < 10000){
			return "R0"+reactionNumber;
			
		}else if(reactionNumber < 100000){
			return "R"+reactionNumber;
			
		}else {
			logger.fatal("The KEGG identification is limited to R99999.");
			System.exit(-1);;
		}
		
		return "";
	}
	
	/**
	 * {@inheritDoc}
	 */
	public boolean unbalanced(){
		
		return this.unbalanced;
	}
	
	/**
	 * {@inheritDoc}
	 * @param balanced
	 */
	public void setUnbalanced(boolean unbalanced){
		
		this.unbalanced	=	unbalanced;
	}
	
	/**
	 * {@inheritDoc}
	 * 
	 * The KEGG database does not use sub-databases. Therefore this method will return and empty name.
	 */
	public String getSubDir(){
		
		return StringConstants.EMPTY;
	}
}