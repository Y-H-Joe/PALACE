package databaseExtraction.databases;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

import org.apache.log4j.Logger;
import org.openscience.cdk.exception.NoSuchAtomTypeException;
import org.openscience.cdk.interfaces.IReaction;

import interfaces.IDatabase;
import interfaces.IEntry;
import io.FileStructure;
import io.KEGGEntry;
import io.writers.GenericFileWriter;
import io.readers.RXNFileReader;

/**Method to read entries from the KEGG database
 * 
 * @author pplehier
 *
 */
public class KEGG implements IDatabase{
	
	private static Logger logger	=	Logger.getLogger(KEGG.class);
	private String defaultOutputDir	=	io.FileStructure.getCurrentDir()+"KEGG\\";
	private String outputDir		=	io.FileStructure.getCurrentDir()+"KEGG\\";
	private String unbalancedDir	=	"Unbalanced\\";
	private static final String name=	"KEGG";
	private int retrieved;
	private int failed;
	private int unbalanced;
	
	public KEGG(){
		
		this.retrieved	=	0;
		this.failed		=	0;
		this.unbalanced	=	0;
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
	
	/**Retrieve all entries of the database and write them to a .rxn file.
	 */
	public void retrieve(){
		
		FileStructure.makeFolder(outputDir);
		FileStructure.makeFolder(outputDir+unbalancedDir);
		
		for(int i = 1;	i < 100000;	i++){
			IEntry ke	=	new KEGGEntry(i);
			if(ke.exists()){
				writeEntry(ke);
				retrieved++;
				
				if(ke.unbalanced()){
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
	
	/**Retrieve a single entry from the database and write it to a .rxn file
	 * 
	 * @param number of the entry
	 */
	public void retrieve(int number){
		
		FileStructure.makeFolder(outputDir);
		FileStructure.makeFolder(outputDir+unbalancedDir);
		
		IEntry ke		=	new KEGGEntry(number);
		
		if(ke.exists()){
			writeEntry(ke);
			retrieved++;
			
			if(ke.unbalanced()){
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
	
	/**Retrieve all entries specified by the content of the array and write them to .rxn files
	 * 
	 * @param array with entry numbers
	 */
	public void retrieve(int[] numbers){
		
		FileStructure.makeFolder(outputDir);
		FileStructure.makeFolder(outputDir+unbalanced);
		
		for(int number:numbers){
			IEntry ke	=	new KEGGEntry(number);
			
			if(ke.exists()){
				writeEntry(ke);
				retrieved++;
				
				if(ke.unbalanced()){
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
		
		FileStructure.makeFolder(outputDir);
		FileStructure.makeFolder(outputDir+unbalancedDir);
		
		for(int i = lowNumber;	i <= highNumber;	i++){
			IEntry ke	=	new KEGGEntry(i);
			
			if(ke.exists()){
				writeEntry(ke);
				retrieved++;
				
				if(ke.unbalanced()){
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
	 * @param ke
	 */
	private void writeEntry(IEntry ke){
		
		File balancedFile	=	new File(outputDir+ke.getID()+".rxn");
		File unbalancedFile	=	new File(outputDir+unbalancedDir+ke.getID()+".rxn");
		boolean successCopy	=	false;
		GenericFileWriter.writeFile(balancedFile, ke.getRXNFile());
		
		if(moveUnbalancedEntry(balancedFile)){
			ke.setUnbalanced(true);
			
			try{	
				logger.info("Reaction "+ke.getID()+" is unbalanced. Moving to folder .\\"+unbalancedDir+".");
				Files.copy(balancedFile.toPath(),unbalancedFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
				successCopy	=	true;
			}
			catch(IOException e){
				e.printStackTrace();
				logger.error("Failed to move entry "+ke.getID()+" to the unbalanced folder.");
			}
			
			if(successCopy){
				try{
					Files.delete(balancedFile.toPath());
				}
				catch(IOException e){
					e.printStackTrace();
					logger.warn("Failed to delete the redundant entry "+ ke.getID()+".");
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