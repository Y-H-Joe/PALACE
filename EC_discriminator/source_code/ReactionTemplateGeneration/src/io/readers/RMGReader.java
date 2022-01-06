package io.readers;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IChemObjectBuilder;
import org.openscience.cdk.io.MDLV2000Writer;
import org.openscience.cdk.smiles.SmilesParser;

import constants.StringConstants;
import interfaces.IDatabaseReader;
import tools.Tools;

public class RMGReader implements IDatabaseReader{
	
	private static Logger logger				=	Logger.getLogger(RMGReader.class);
	private static final String RMG				=	"http://rmg.mit.edu/";
	private static final String RMGLibrary		=	"database/kinetics/libraries/";
	private static final String RMGMolecule		=	"database/molecule/";
	private static final String htmlLink		=	"<a href=\"/";
	private static final String notFound		=	"RMG Page Not Found (404)";
	private static final String serverError		=	"RMG Server Error";
	private static final String reactionArrow	=	"class=\"reactionArrow\">";	
	@SuppressWarnings("unused")
	private static final String inChI			=	"InChI=";
	private static final String smiles			=	"SMILES";
	private static final String tableLineEnd	=	"</td>";
	private static final String temp			=	"temp.mol";
	private static List<String> libFile			=	null;
	private static String[] libraryNames		=	null;
	private static Integer[] entryCount			=	null;
	private static int[] restartFromLine		=	null;
	private static List<List<String>> subDataFiles;
	
	private String subData;
	private int libIndex;
	private int number;
	private int productCount;
	private int reactantCount;
	private List<String> componentIdentifiers;
	private List<String> reactantLinks;
	private List<String> productLinks;
	private List<String> componentLinks;
	private BufferedReader reactionReader;
	private String[] componentMolFiles;
	private int[] componentCoefs;
	
	
	public RMGReader(String subDatabase,int number){
	
		this.subData				=	subDatabase;
		this.number					= 	number;
		this.reactantLinks			=	new ArrayList<String>();
		this.productLinks			=	new ArrayList<String>();
		this.componentLinks			=	new ArrayList<String>();
		this.componentIdentifiers	=	new ArrayList<String>();
		this.libIndex				=	getLibraryIndex(this.subData);
		String line;
		
		if(restartFromLine == null){
			initialiseRestart();
		}
		if(subDataFiles == null){
			subDataFiles	=	new ArrayList<List<String>>();
			if(libraryNames == null) readLibraryNames();
			for(int i = 0;	i < libraryNames.length; i++){
				subDataFiles.add(i, null);
			}
		}
		if(subDataFiles.get(libIndex) == null){
			List<String> subDataFile	=	new ArrayList<String>();
			try{
				URL urlRMG			=	new URL(RMG+RMGLibrary+subData);
				URLConnection con	=	urlRMG.openConnection();
				InputStream is		=	con.getInputStream();
				this.reactionReader	=	new BufferedReader(new InputStreamReader(is));

				//Read the sub data file.
				while((line = reactionReader.readLine()) != null){
					subDataFile.add(line);
				}

				reactionReader.close();

			} catch(IOException e){
				e.printStackTrace();
			}
			
			subDataFiles.set(libIndex, subDataFile);
		}
	}

	/**
	 * {@inheritDoc}
	 */
	public boolean read(){
		
		boolean success	=	this.readComponents();
		
		if(success){
			this.setComponentMolFiles();
			this.setReactantCount();
			this.setProductCount();
			return true;
		}else{
			return false;
		}
	}
	
	/**Set the number of reactants
	 */
	private void setReactantCount(){
		
		this.reactantCount	=	reactantLinks.size();	
	}
	
	/**Set the number of products
	 */
	private void setProductCount(){
		
		this.productCount	=	productLinks.size();
	}
	
	/**Set the component mol (MDL format) files
	 */
	private void setComponentMolFiles(){
		
		this.componentMolFiles	=	new String[componentIdentifiers.size()];
		
		for(int i = 0;	i < componentMolFiles.length;	i++){
	    	try {	
	    		File tempFile					=	new File(temp);
	    		FileWriter writer				=	new FileWriter(tempFile);
	    		MDLV2000Writer molFileWriter	=	new MDLV2000Writer(writer);
	    		IChemObjectBuilder builder		=	DefaultChemObjectBuilder.getInstance();
	    		SmilesParser converter 			=	new SmilesParser(builder);
				IAtomContainer component		=	converter.parseSmiles(componentIdentifiers.get(i));
				
				molFileWriter.write(component);
				molFileWriter.close();
				
				TXTFileReader tfr		=	new TXTFileReader(temp);
				String molFile			=	StringConstants.NEW_LINE+
											StringConstants.NEW_LINE+
											StringConstants.NEW_LINE+
											tfr.read(4, '+');
				componentMolFiles[i]	=	molFile.substring(0, molFile.length()-1);
				tempFile.delete();
			
			} catch (CDKException e) {
				e.printStackTrace();
			} catch (Exception e) {
				e.printStackTrace();
			}	    	
		}
	}
	
	/**
	 * {@inheritDoc}
	 */
	public int getProductCount(){
		
		return this.productCount;
	}
	
	/**
	 * {@inheritDoc}
	 */
	public int getReactantCount(){
		
		return this.reactantCount;
	}
	
	/**
	 * {@inheritDoc}
	 */
	public int[] getComponentCoefs(){
		
		if(componentCoefs == null){
			componentCoefs	=	new int[componentLinks.size()];
			for(int i = 0;	i < componentCoefs.length;	i++)
				componentCoefs[i]	=	1;
		}
		
		return componentCoefs;
	}
	
	/**
	 * {@inheritDoc}
	 */
	public String[] getComponentMolFiles(){
		
		return componentMolFiles;
	}
	
	/**Read all the components and return whether reading them was successful.
	 * 
	 * @return success
	 */
	private boolean readComponents(){
		
		boolean getLinkSuccess	=	this.getComponentLinks();
		
		if(!getLinkSuccess){
			return false;
		}
		else{
			for(String componentLink:this.componentLinks){
				boolean success	=	this.readComponentInChIOrSmiles(componentLink);
				
				if(!success){
					return false;
				}
			}
		return true;
		}
	}
	
	/**Get the link to the component site from the reaction site and return whether this opperation 
	 * was successful. Restarts from the last visited line for that subdatabase. 
	 * 
	 * @return success
	 */
	private boolean getComponentLinks(){

		int lineCount	=	subDataFiles.get(libIndex).size();
		int lineNumber	=	restartFromLine[libIndex];
		
		while(lineNumber < lineCount){
			String line	=	subDataFiles.get(libIndex).get(lineNumber);

			if(line.contains(notFound) || line.contains(serverError)){
				logger.info("The entry "+subData+"_"+number+"\" does not exist in or cannot be retrieved from the RMG database.");
				return false;
			}
			else if(getNumber(line) == number){
				boolean continueRead	=	true;
				boolean breakFound		=	false;
				//Continue reading from the reaction line on. Each reactant will now be found via the 
				//<a href="/database/molecule tag
				while(continueRead){
					//read next line
					line 				=	subDataFiles.get(libIndex).get(++lineNumber);
					//check whether it still exists (minus one as one additional loop will always run)
					boolean readNext	=	lineNumber < lineCount-1;
					//check whether reading reactants or products	
					if(!breakFound){
						breakFound	=	line.contains(reactionArrow);
					}

					if(!readNext){
						continueRead	=	false;
					}

					else{
						if(line.contains(htmlLink+RMGMolecule)){
							if(!breakFound){
								reactantLinks.add(line.substring(line.indexOf(htmlLink)+htmlLink.length(), line.indexOf("\"><img")));

							}else{
								productLinks.add(line.substring(line.indexOf(htmlLink)+htmlLink.length(), line.indexOf("\"><img")));
							}
						}	

						continueRead	=	getNumber(line) <= number;
					}
				}

				componentLinks.addAll(reactantLinks);
				componentLinks.addAll(productLinks);
				restartFromLine[libIndex]	=	lineNumber;
				return true;
			}

			else if(getNumber(line) > number){
				logger.info("The reaction with id \""+this.subData+"_"+this.number+" does not exist in the RMG database");
				return false;
			}
			
			lineNumber++;
		}        		

			
		logger.info("The reaction with id \""+this.subData+"_"+this.number+" does not exist in the RMG database");
		return false;
	}
	
	/**Get the number of the entry on the read line
	 * 
	 * @param line
	 * @return number
	 */
	private int getNumber(String line){
		
		if(line.contains(htmlLink+RMGLibrary+subData+"/")){
			int pos0	=	line.indexOf(htmlLink+RMGLibrary+subData+"/");
			int pos1	=	pos0 + (htmlLink+RMGLibrary+subData+"/").length();
			int pos2	=	line.indexOf("/\">", pos1);
			
			if(pos0 == -1 || pos2 == -1){
				return -1;
			}
			
			return Integer.parseInt(line.substring(pos1, pos2));
		}
		
		return -1;
	}
	
	/**From the component site, read the Smiles identifier for this component. Would read InChI's, but RMG contains
	 * [CH] radicals, for which an incorrect InChI is listed.
	 * 
	 * @param componentSearch
	 * @return smiles
	 */
	private boolean readComponentInChIOrSmiles(String componentSearch){
		
		try{
			URL urlRMGMolecule			=	new URL(RMG+componentSearch);
			URLConnection con			=	urlRMGMolecule.openConnection();
			InputStream is				=	con.getInputStream();
			BufferedReader molReader	=	new BufferedReader(new InputStreamReader(is));
			String moleculeLine			=	null;
			String previousLine			=	StringConstants.EMPTY;
			
			while((moleculeLine = molReader.readLine()) != null){
				if(previousLine.contains(smiles)){
					componentIdentifiers.add(moleculeLine.substring(moleculeLine.indexOf('>')+1, moleculeLine.length()-tableLineEnd.length()));
					return true;
				}
				//Not being used as invalid inchis are listed for ao [CH] radicals. If fixed, use inchi!
				/*else if(moleculeLine.contains(inChI)){
					componentIdentifiers.add(moleculeLine.substring(moleculeLine.indexOf(inChI), moleculeLine.length()-tableLineEnd.length()));
					return true;
				}*/
				previousLine	=	moleculeLine;
			}
		} catch(IOException e){
			e.printStackTrace();
		}
		
		return false;
	}
	
	/**For a given sub database, retrieve how many entries it has
	 * 
	 * @param subDatabase
	 * @return entry count
	 */
	public static int getEntryCount(String subDatabase){
		
		if(libraryNames == null)
			readLibraryNames();
		
		for(int i = 0;	i < libraryNames.length;	i++){
			if(libraryNames[i].equals(subDatabase)){
				return entryCount[i];
			}
		}
		
		return -1;
	}
	
	/**Get a list of all the available sub databases
	 * 
	 * @return
	 */
	public static String[] getLibraryNames(){
		
		if(libraryNames == null)
			readLibraryNames();
		return libraryNames;
	}
	
	/**Get a list of the entry count for each sub database
	 * 
	 * @return
	 */
	public static int[] getEntryCounts(){
		
		if(entryCount == null)
			readLibraryNames();
		
		int[] retCount	=	new int[entryCount.length];
		
		for(int i = 0;	i < retCount.length;	i++){
			retCount[i]	=	entryCount[i];
		}
		
		return retCount;
	}
	
	/**Read all the subdatabase names from the overview site
	 */
	private static void readLibraryNames(){
		
		if(libFile == null){
			libFile	=	new ArrayList<String>();
		
			try {
				URL libraryURL			=	new URL(RMG+RMGLibrary);
				URLConnection libCon	=	libraryURL.openConnection();
				InputStream is			=	libCon.getInputStream();
				BufferedReader libReader=	new BufferedReader(new InputStreamReader(is));
				int lineCounter			=	0;
				List<String> libNames	=	new ArrayList<String>();
				List<Integer> entCount	=	new ArrayList<Integer>();
				String line;
				String previousLine		=	StringConstants.EMPTY;

				while((line = libReader.readLine()) != null){
					libFile.add(line);

					if(line.contains(htmlLink+RMGLibrary)){
						if(lineCounter != 0){
							libNames.add(line.substring(line.indexOf(RMGLibrary)+RMGLibrary.length(), line.indexOf("/\">")));
						}

						lineCounter++;
					}

					if(previousLine.contains(htmlLink+RMGLibrary)){
						if(lineCounter - 1 != 0){
							entCount.add(Integer.parseInt(line.substring(line.indexOf('(')+1, line.indexOf('e')-1)));
						}
					}

					previousLine	=	line;
				}

				libraryNames	=	new String[libNames.size()];
				entryCount		=	new Integer[entCount.size()];
				libraryNames	=	libNames.toArray(libraryNames);
				entryCount		=	entCount.toArray(entryCount);

			} catch (MalformedURLException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		else{
			int lineNumber			=	0;
			int lineCount			=	libFile.size();
			List<String> libNames	=	new ArrayList<String>();
			List<Integer> entCount	=	new ArrayList<Integer>();
			String previousLine		=	StringConstants.EMPTY;
			
			while(lineNumber < lineCount){
				
				String line	=	libFile.get(lineNumber);

				if(line.contains(htmlLink+RMGLibrary)){
					if(lineNumber != 0){
						libNames.add(line.substring(line.indexOf(RMGLibrary)+RMGLibrary.length(), line.indexOf("/\">")));
					}
				}

				if(previousLine.contains(htmlLink+RMGLibrary)){
					if(lineNumber - 1 != 0){
						entCount.add(Integer.parseInt(line.substring(line.indexOf('(')+1, line.indexOf('e')-1)));
					}
				}

				previousLine	=	line;
			}

			libraryNames	=	new String[libNames.size()];
			entryCount		=	new Integer[entCount.size()];
			libraryNames	=	libNames.toArray(libraryNames);
			entryCount		=	entCount.toArray(entryCount);
		}
	}

	/**
	 * {@inheritDoc}
	 */
	public String[] getComponentIDs() {
		
		String[] out	=	new String[componentIdentifiers.size()];
		return componentIdentifiers.toArray(out);
	}
	
	private static void initialiseRestart(){
		
		if(libraryNames == null){
			readLibraryNames();
		}
		
		restartFromLine	=	new int[libraryNames.length];
	}
	
	private static int getLibraryIndex(String data){
		
		try{
			return	Integer.parseInt(data);
		}
		catch(NumberFormatException notAnInteger){
			if(libraryNames == null)
				readLibraryNames();
			return Tools.indexOf(data,libraryNames);
		}
		
	}
}