package databaseExtraction.databases;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

import org.apache.log4j.Logger;
import org.openscience.cdk.exception.NoSuchAtomTypeException;
import org.openscience.cdk.interfaces.IReaction;

import constants.StringConstants;
import interfaces.IDatabase;
import interfaces.IEntry;
import io.FileStructure;
import io.RMGEntry;
import io.readers.RMGReader;
import io.readers.RXNFileReader;
import io.writers.GenericFileWriter;
import tools.Tools;

public class RMG implements IDatabase{

	private static Logger logger	=	Logger.getLogger(RMG.class);
	private String defaultOutputDir	=	io.FileStructure.getCurrentDir()+"RMG\\";
	private String outputDir		=	io.FileStructure.getCurrentDir()+"RMG\\";
	private String unbalancedDir	=	"Unbalanced\\";
	private String[] readLibraries;
	private static final String name=	"RMG";
	private int retrieved;
	private int failed;
	private int unbalanced;
	
	/**Constructs the RMG database. The argument limits the database to a sub database if it differs from "ALL".
	 * 
	 * @param readLib
	 */
	public RMG(String readLib){
		
		this.retrieved	=	0;
		this.failed		=	0;
		this.unbalanced	=	0;
		
		if(readLib.equals(StringConstants.ALL)){
			readLibraries	=	RMGReader.getLibraryNames();
		}
		else{
			if(checkArgument(readLib)){
				readLibraries	=	new String[1];
				readLibraries[0]=	getLibName(readLib);
			}
			else{
				logger.fatal("The specified database name is not present in the RMG database. Exiting ...");
				System.exit(-1);
			}
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
	
	private boolean checkArgument(String arg){
		
		return Tools.contains(RMGReader.getLibraryNames(), getLibName(arg));
	}
	/**Return the default output directory
	 * 
	 * @return name of output directory
	 */
	public String defaultOutput(){
		
		return defaultOutputDir;
	}
	
	/**Get the name of the database
	 * 
	 * @return database name
	 */
	public String getName(){
		
		return name;
	}
	
	/**Retrieve all entries of the (specified) (sub)database(s) and write them to a .rxn file.
	 */
	public void retrieve(){
		
		FileStructure.makeFolder(outputDir);
		FileStructure.makeFolder(outputDir+unbalancedDir);
		
		for(int i = 0;	i < readLibraries.length;	i++){
			for(int j = 0;	j < RMGReader.getEntryCount(readLibraries[i]);	j++){
				//RMG entries start counting at 1 not 0!
				IEntry re	=	new RMGEntry(readLibraries[i],j+1);
				if(re.exists()){
					writeEntry(re);
					retrieved++;
					
					if(re.unbalanced()){
						unbalanced++;
					}
				}
				else{
					failed++;
				}
			}
		}
		
		logger.info("Retrieved "+retrieved+" reaction"+(retrieved == 1?"":"s")+", "+
				failed+" reaction"+(failed == 1?"":"s")+" do"+(failed == 1?"es":"")+" not exist, "+
				unbalanced+" reaction"+(unbalanced == 1?"":"s")+" "+(unbalanced == 1?"is":"are")+" unbalanced.");
	}
	
	/**Retrieve a single entry from the database and write it to a .rxn file. This requires a subdatabase to have been
	 * specified in the arguments.
	 * 
	 * @param number of the entry
	 */
	public void retrieve(int number){
		
		if(readLibraries.length != 1){
			logger.fatal("Cannot retrieve one entry for all subdatabases of RMG. Exiting ...");
			System.exit(-1);
		}
		
		if(number > RMGReader.getEntryCount(readLibraries[0])){
			logger.fatal("The specified entry index is out of range for the specified subdatabase. Exiting ...");
			System.exit(-1);
		}
		
		FileStructure.makeFolder(outputDir);
		FileStructure.makeFolder(outputDir+unbalancedDir);
		
		IEntry re		=	new RMGEntry(readLibraries[0],number);
		
		if(re.exists()){
			writeEntry(re);
			retrieved++;
			
			if(re.unbalanced()){
				unbalanced++;
			}
		}
		else{
			failed++;
		}
		
		logger.info("Retrieved "+retrieved+" reaction"+(retrieved == 1?"":"s")+", "+
				failed+" reaction"+(failed == 1?"":"s")+" do"+(failed == 1?"es":"")+" not exist, "+
				unbalanced+" reaction"+(unbalanced == 1?"":"s")+" "+(unbalanced == 1?"is":"are")+" unbalanced.");
	}
	
	/**Retrieve all entries specified by the content of the array and write them to .rxn files. This requires a subdatabase to have been
	 * specified in the arguments.
	 * 
	 * @param array with entry numbers
	 */
	public void retrieve(int[] numbers){
	
		if(readLibraries.length != 1){
			logger.fatal("Cannot retrieve one entry for all subdatabases of RMG. Exiting ...");
			System.exit(-1);
		}
		
		FileStructure.makeFolder(outputDir);
		FileStructure.makeFolder(outputDir+unbalancedDir);
		
		for(int number:numbers){
			IEntry re	=	new RMGEntry(readLibraries[0],number);
			
			if(re.exists()){
				writeEntry(re);
				retrieved++;
				
				if(re.unbalanced()){
					unbalanced++;
				}
			}
			else{
				failed++;
			}
		}
		
		logger.info("Retrieved "+retrieved+" reaction"+(retrieved == 1?"":"s")+", "+
				failed+" reaction"+(failed == 1?"":"s")+" do"+(failed == 1?"es":"")+" not exist, "+
				unbalanced+" reaction"+(unbalanced == 1?"":"s")+" "+(unbalanced == 1?"is":"are")+" unbalanced.");
	}
	
	/**Retrieve all entries between the low and high number and writes them to .rxn files.
	 * 
	 * @param max, min
	 */
	public void retrieve(int lowNumber, int highNumber){
		
		if(readLibraries.length != 1){
			logger.fatal("Cannot retrieve one entry for all subdatabases of RMG. Exiting ...");
			System.exit(-1);
		}
		
		FileStructure.makeFolder(outputDir);
		FileStructure.makeFolder(outputDir+unbalancedDir);
		
		for(int i = lowNumber;	i <= highNumber;	i++){
			IEntry re	=	new RMGEntry(readLibraries[0],i);
			
			if(re.exists()){
				writeEntry(re);
				retrieved++;
				
				if(re.unbalanced()){
					unbalanced++;
				}
			}
			else{
				failed++;
			}
		}
		
		logger.info("Retrieved "+retrieved+" reaction"+(retrieved == 1?"":"s")+", "+
				failed+" reaction"+(failed == 1?"":"s")+" do"+(failed == 1?"es":"")+" not exist, "+
				unbalanced+" reaction"+(unbalanced == 1?"":"s")+" "+(unbalanced == 1?"is":"are")+" unbalanced.");
	}
	
	/**Set the output directory to a specified output
	 * 
	 * @param name of the output directory
	 */
	public void setOutputDir(String output){
		
		if(!(output.endsWith("\\") || output.endsWith("/"))){
			outputDir	=	output+"\\";
		}
		else{
			outputDir	=	output;
		}
		
	}
	
	/**Write an entry of the database to a .rxn file
	 * 
	 * @param re
	 */
	private void writeEntry(IEntry re){
		
		String fileName		=	FileStructure.validateFileName(re.getID());
		
		File balancedFile	=	new File(outputDir+FileStructure.validateFileName(re.getSubDir())+"/"+fileName+".rxn");
		File unbalancedFile	=	new File(outputDir+unbalancedDir+fileName+".rxn");
		
		//If the sub-database folder doesn't yet exist, make it
		if(!balancedFile.getParentFile().exists()){
			FileStructure.makeFolder(balancedFile.getParent());
		}
		
		boolean successCopy	=	false;
		GenericFileWriter.writeFile(balancedFile, re.getRXNFile());
		
		if(moveUnbalancedEntry(balancedFile)){
			re.setUnbalanced(true);
			
			try{
				logger.info("Reaction "+re.getID()+" is unbalanced. Moving to folder .\\"+unbalancedDir+".");
				Files.copy(balancedFile.toPath(),unbalancedFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
				successCopy	=	true;
			}
			catch(IOException e){
				e.printStackTrace();
				logger.error("Failed to move entry "+re.getID()+" to the unbalanced folder.");
			}
			
			if(successCopy){
				try{
					Files.delete(balancedFile.toPath());
				}
				catch(IOException e){
					e.printStackTrace();
					logger.warn("Failed to delete the redundant entry "+ re.getID()+".");
				}
			}
		}
	}
	
	
	/**Method that determines whether an entry should be moved to the unbalanced folder.
	 * 
	 * @param balancedFile
	 * @return unbalanced?
	 */
	private static boolean moveUnbalancedEntry(File balancedFile){
		
		IReaction rxn= (new RXNFileReader(balancedFile.getPath())).toReaction(false);

		try {
			return !rxn.checkAtomBalance();
		} catch (NoSuchAtomTypeException e) {
			e.printStackTrace();
			logger.error("Atom balance check failed, assuming reaction is unbalanced.");
		}
		
		return true;
	}
}