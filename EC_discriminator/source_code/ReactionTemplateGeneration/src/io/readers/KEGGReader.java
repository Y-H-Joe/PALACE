package io.readers;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;

import constants.StringConstants;
import interfaces.IDatabaseReader;
import tools.Output;
import tools.MyMath;

/**This class reads data from the KEGG web-based database and translates it to .rxn files.
 * 
 * @author pplehier
 *
 */
public class KEGGReader implements IDatabaseReader{
	
	private static Logger logger				=	Logger.getLogger(KEGGReader.class);
	private static final String KEGG			=	"http://www.kegg.jp/";
	private static final String reactionEntry	=	"entry/";
	private static final String KEGG_Entry		=	KEGG+reactionEntry;
	private static final String moleculeEntry	=	"dbget-bin/www_bget?cpd:";
	private static final String componentMolFile=	"dbget-bin/www_bget?-f+m+compound+";
	//The link to a component is specified by the following line, followed by the component ID
	private static final String componentTag	=	"<a href=\"/dbget-bin/www_bget?cpd:";
	//This indicates a new cell in the table
	private static final String componentStart	=	"y:hidden\">";
	private static final String componentDelim	=	" \\+ ";
	private static final String comment			=	"Comment";
	//The reactant - product delimiter <=> is read as below
	private static final String reactionDelim	=	" &lt;=> ";
	private static final int reactProdBreakI	=	-1;
	private static final String reactProdBreakS	=	"Reactant-product break";
	private static final String header			=	"\n\tdatabaseExtraction_mol\t\t\t" + Output.getDate() + "\n";
	private static final String doesNotExist1	=	": No such data.";
	private static final String doesNotExist2	=	"No such data was found.";
	private static final String equationTag		=	"Equation";
	
	private String reactionID;
	private BufferedReader reactionReader;
	private List<String> componentIDs;
	private List<Integer> componentCoefs;
	private List<String> componentMolFiles;
	private int reactantCount;
	private int productCount;
	private int multiplier;
	
	/**Open a new reader
	 * 
	 * @param reactionID
	 */
	public KEGGReader(String reactionID){
		
		this.reactionID			= 	reactionID;
		this.componentIDs		=	new ArrayList<String>();
		this.componentCoefs		=	new ArrayList<Integer>();
		this.componentMolFiles	=	new ArrayList<String>();
		
		// Get the input stream through URL Connection   
		try {
			URL urlKEGG			=	new URL(KEGG_Entry+reactionID);
			URLConnection con 	=	urlKEGG.openConnection();
			InputStream is 		=	con.getInputStream();
			this.reactionReader	=	new BufferedReader(new InputStreamReader(is));
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/**Read database information for each molecule of the reaction
	 * 
	 * @return whether the operation was successful
	 */
	public boolean read(){
		
		boolean success	=	this.readComponents();
		
		if(success){
			//Order is important! 
			//Multiplier needs molfiles to have been set
			this.setMolFiles();
			this.setMultiplier();
			this.setReactantCount();
			this.setProductCount();
			return true;
		}else{
			return false;
		}
	}
	
	/**Get the number of reactants in the reaction (corrected for a multiplier,
	 * if the simplest coefficients were not given).
	 * 
	 * @return number of reactants
	 */
	public int getReactantCount(){
		return reactantCount / multiplier;
	}
	
	/**Get the number of products in the reaction (corrected for a multiplier, 
	 * if the simplest coefficients were not given).
	 * 
	 * @return the number of products
	 */
	public int getProductCount(){
		return productCount / multiplier;
	}
	
	/**Get the molecule ids in the KEGG database of the molecules in the reaction.
	 * A molID is similar to a reaction id but starts with M instead of R
	 * 
	 * @return Molecule IDs
	 */
	public String[] getComponentIDs(){
		
		String[] ids	=	new String[componentIDs.size()-1];
		int i			=	0;
		
		while(!componentIDs.get(i).equals(reactProdBreakS)){
			ids[i]		=	componentIDs.get(i);
			i++;
		}
		
		i++;
		
		while(i < componentIDs.size()){
			ids[i-1]	=	componentIDs.get(i);
			i++;
		}
		
		
		return ids;
	}
	
	/**Get the coefficients of all components in the reaction
	 * TODO:simplify if still multiple of each other
	 * @return coefficients.
	 */
	public int[] getComponentCoefs(){
		
		int[] coefs	=	new int[componentCoefs.size()-1];
		int i		=	0;
		
		while(!componentCoefs.get(i).equals(reactProdBreakI)){
			coefs[i]		=	componentCoefs.get(i);
			i++;
		}
		
		i++;
		
		while(i < componentCoefs.size()){
			coefs[i-1]	=	componentCoefs.get(i);
			i++;
		}
		
		simplifyCoefficients(coefs);
		return coefs;
	}
	
	/**Get the mol files for each component of the reaction from the database
	 * 
	 * @return mol files (in string format)
	 */
	public String[] getComponentMolFiles(){
		
		String[] mol	=	new String[componentMolFiles.size()-1];
		int i			=	0;
		
		while(!componentMolFiles.get(i).equals(reactProdBreakS)){
			mol[i]		=	componentMolFiles.get(i);
			i++;
		}
		
		i++;
		
		while(i < componentMolFiles.size()){
			mol[i-1]	=	componentMolFiles.get(i);
			i++;
		}
		
		return mol;
	}
	
	/**Read all components from a reaction to Strings
	 * Returns whether the operation was successful
	 * 
	 * @return success
	 */
	private boolean readComponents(){
		
        String line 		= null;
        String previousLine	=	StringConstants.EMPTY;
       
        try {
			while ((line = reactionReader.readLine()) != null) {
				//there are two possible outputs when an entry does not exist in the database.
				//if the line contains either, notify, return failure and to next reaction
				if(line.contains(doesNotExist1) || line.contains(doesNotExist2)){
					logger.info("The entry "+reactionID+" does not exist in the KEGG database");
					return false;
				}
				else{
					//The reaction equation is displayed in the cell next to "Equation" This cell is 
					//defined on the previous line
					if(previousLine.contains(equationTag)){
						String componentsLine		=	line.substring(line.indexOf(componentStart));
						componentsLine				=	(componentsLine.substring(componentStart.length()));
						//In the preliminary components, there is still one line whith 2 components, ie the one 
						//containing the reaction arrow.
						String[] componentsPrelim	=	componentsLine.split(componentDelim);
						String[] components			=	new String[componentsPrelim.length+1];
					
						//Insert the split reaction arrow line.
						int endOfReactants			=	0;
						for(int j = 0;	j < componentsPrelim.length;	j++){
							if(componentsPrelim[j].contains(reactionDelim)){
								for(int k = 0;	k < j;	k++){
									components[k]	=	componentsPrelim[k];
								}
							
								components[j]		=	componentsPrelim[j].split(reactionDelim)[0];
								components[j+1]		=	componentsPrelim[j].split(reactionDelim)[1];
								endOfReactants		=	j+1;
								for(int k = j + 1;	k < componentsPrelim.length;	k++){
									components[k+1]	=	componentsPrelim[k];
								}
							}
						}
						//interpret coefficients. TODO: find a fix when coefficient is undefined (eg letter)
						for(int j = 0;	j < endOfReactants;	j++){
							if(components[j].startsWith("<a")){
								componentCoefs.add(1);
							}
							else if(components[j].startsWith("n") || components[j].startsWith("(")){
								logger.info("The entry "+reactionID+" contains an unspecified number of reactants and is not retrieved");
								return false;
							}
							else{
								componentCoefs.add(Integer.parseInt(components[j].substring(0, 2).trim()));
							}
							int delimStart	=	components[j].indexOf(componentTag);
							int startIndex	=	delimStart+componentTag.length();
							int endIndex	=	startIndex+reactionID.length();
						
							componentIDs.add(components[j].substring(startIndex, endIndex));
						}
						//indicate the break between reactants and products.
						componentCoefs.add(reactProdBreakI);
						componentIDs.add(reactProdBreakS);
					
						for(int j = endOfReactants;	j < components.length; j++){
							if(components[j].startsWith("<a")){
								componentCoefs.add(1);
							}
							else if(components[j].startsWith("n") || components[j].startsWith("(")){
								logger.info("The entry "+reactionID+" contains an unspecified number of products and is not retrieved");
								return false;
							}
							else{
								componentCoefs.add(Integer.parseInt(components[j].substring(0, 2).trim()));
							}							
							int delimStart	=	components[j].indexOf(componentTag);
							int startIndex	=	delimStart+componentTag.length();
							int endIndex	=	startIndex+reactionID.length();
						
							componentIDs.add(components[j].substring(startIndex, endIndex));
						}
				
						//initialise all componentMolFiles as empty
						for(int i = 0;	i < componentIDs.size();	i++){
							if(componentIDs.get(i).equals(reactProdBreakS)){
								componentMolFiles.add(reactProdBreakS);
							}
							else{
								componentMolFiles.add(StringConstants.EMPTY);
							}
						}
						return true;
					}
				} 
				
				previousLine	=	line;
			}
        }
			catch (IOException e) {
			e.printStackTrace();
		}
        return false;
	}
	
	/**Get the molFile for one component and set in list. If the mol file for that component does not exist,
	 * the comment for that molecule is searched for an alternative name.
	 * 
	 * @param componentID
	 */
	private void setMolFileComponent(String componentID){
		
		String molFileString	=	getMolFile(componentID);
		
		if(molFileString.equals(StringConstants.EMPTY)){
			molFileString		=	getMolFile(findAlternativeID(componentID));
		}

		componentMolFiles.set(componentIDs.indexOf(componentID), molFileString);
	}
	
	/**Retrieve a mol file
	 * 
	 * @param componentID
	 * @return mol file
	 */
	private String getMolFile(String componentID){
		
		BufferedReader molReader= 	null;
		int lineNr				=	0;
		String molFileString	=	StringConstants.EMPTY;
		
		try {
			URL urlMol			=	new URL(KEGG+componentMolFile+componentID);
			URLConnection con 	=	urlMol.openConnection();
			InputStream is 		=	con.getInputStream();
			molReader			=	new BufferedReader(new InputStreamReader(is));
			String line			=	StringConstants.EMPTY;
			
			while ((line = molReader.readLine()) != null) {
				if(lineNr >2){
					molFileString	=	molFileString + "\n" + line;
				}
				
				lineNr++;
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		if(molFileString.equals(StringConstants.EMPTY)){
			return molFileString;
		}
		else{
			return header+molFileString;
		}
	}
	
	/**It is possible that an alternative ID exists for the molecule under the tab comments. 
	 * This method searches this id.
	 * @param currentID
	 * @return alternative ID
	 */
	private String findAlternativeID(String currentID){
		
		String alternativeID			=	StringConstants.EMPTY;
		
		try {
			URL urlKEGG						=	new URL(KEGG+moleculeEntry+currentID);
			URLConnection con 				=	urlKEGG.openConnection();
			InputStream is 					=	con.getInputStream();
			BufferedReader moleculeReader	=	new BufferedReader(new InputStreamReader(is));
			boolean found					=	false;
			String moleculeLine				=	null;
			String previousLine				=	StringConstants.EMPTY;
			String prePreviousLine			=	StringConstants.EMPTY;
			
			while((moleculeLine = moleculeReader.readLine()) != null && !found){
				if((previousLine.contains(comment) || prePreviousLine.contains(comment)) 
					&& 
					moleculeLine.contains(componentTag)){
					int delimStart	=	moleculeLine.indexOf(componentTag);
					int startIndex	=	delimStart+componentTag.length();
					int endIndex	=	startIndex+reactionID.length();
					alternativeID	=	moleculeLine.substring(startIndex,endIndex);
					found			=	true;
				}
				//The text of the cell can have one break line in it: the line containing the compound therefore comes
				//two lines after the line containing "Comment".
				prePreviousLine			=	previousLine;
				previousLine			=	moleculeLine;
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return alternativeID;
	}
	
	/**Set mol files for all components
	 */
	private void setMolFiles(){
		for(String ID:componentIDs){
			if(!ID.equals(reactProdBreakS)){
				this.setMolFileComponent(ID);
			}
		}
	}
	
	/**Determine the number of reactants. Does not count reactants for which the mol file is empty!
	 */
	private void setReactantCount(){
		
		int i		=	0;
		int count	=	0;	
		
		while(componentCoefs.get(i) != reactProdBreakI){
			if(!componentMolFiles.get(i).equals(StringConstants.EMPTY)){
				count	+=	componentCoefs.get(i);
			}
			
			i++;
		}
		
		reactantCount	=	count;
	}
	
	/**Determine the number of products. Does not count products for which the mol file is empty
	 */
	private void setProductCount(){
		
		int count	=	0;
		
		for(int i = componentCoefs.indexOf(reactProdBreakI)+1;	i < componentCoefs.size();	i++){
			if(!componentMolFiles.get(i).equals(StringConstants.EMPTY)){
				count	+=	componentCoefs.get(i);
			}
		}
		
		productCount	=	count;
	}
	
	/**Some reactions are stored with coefficients that are not in their simplest form. <br>
	 * EG: 4A+8B -> 6C+2D this method simplifies the coefficients to 2A+4B -> 3C+D
	 * @param coefficients
	 */
	private void simplifyCoefficients(int[] coefficients){

		for(int i = 0;	i < coefficients.length;	i++){
			coefficients[i]	=	coefficients[i] / multiplier;
		}
	}
	
	/**Set the multiplier for the component coefficients
	 */
	private void setMultiplier(){
		
		List<Integer> actualCoefs	=	new ArrayList<Integer>();
		
		for(int i = 0;	i < componentCoefs.size();	i++){
			if(componentMolFiles.get(i).equals(StringConstants.EMPTY)){}
			else{
				actualCoefs.add(componentCoefs.get(i));
			}
		}
		
		this.multiplier	= MyMath.greatestCommonDivisor(actualCoefs);
	}
}