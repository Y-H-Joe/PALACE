package io;

import org.apache.log4j.Logger;

import constants.StringConstants;
import interfaces.IDatabaseReader;
import interfaces.IEntry;
import io.readers.RMGReader;
import tools.Output;

public class RMGEntry implements IEntry{
	
	private static Logger logger		=	Logger.getLogger(RMGEntry.class);
	private static final String RXN		=	"$RXN\n";
	private static final String MOL		=	"\n$MOL\n";
	private static final String header	=	"\n  databaseExtraction_rxn\t\t\t" + Output.getDate() + "\n";
	
	private String subDatabase;
	private int number;
	private IDatabaseReader RR;
	private int[] componentCoefs;
	private String[] componentMolFiles;
	private String reactionFile;
	private boolean exists;
	private boolean unbalanced	=	false;
	

	public RMGEntry(String subDatabase, int number){
		
		this.subDatabase	=	subDatabase;
		this.number			=	number;
		RR					=	new RMGReader(subDatabase,number);
		exists				=	RR.read();
		
		if(exists){
			componentCoefs		=	RR.getComponentCoefs();
			componentMolFiles	=	RR.getComponentMolFiles();
			
			this.createRXNFile();
			logger.info("Retrieved RMG entry with ID "+getID());
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
		
		return this.subDatabase+"_"+number;
	}
	
	/**Get the name of the sub-database to which the entry belongs
	 * 
	 * @return sub-database name
	 */
	public String getSubDir(){
		
		return this.subDatabase;
	}
	
	/**Create a .rxn file using the available .mol files in the database
	 * TODO: simplify coefficients.
	 */
	private void createRXNFile(){
		
		String rxnFileString	=	RXN+header;
		
		int reactantCount		=	RR.getReactantCount();
		int productCount		=	RR.getProductCount();
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
	
	/**
	 * {@inheritDoc}
	 */
	public boolean unbalanced(){
		
		return this.unbalanced;
	}
	
	/**
	 * {@inheritDoc}
	 */
	public void setUnbalanced(boolean unbalanced){
		
		this.unbalanced	=	unbalanced;
	}
}