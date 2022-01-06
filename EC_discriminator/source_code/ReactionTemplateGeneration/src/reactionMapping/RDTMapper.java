package reactionMapping;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.apache.log4j.Logger;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IReaction;
import org.openscience.cdk.smiles.SmilesGenerator;

import changes.ChangeDetector;
import constants.StringConstants;
import io.FileStructure;
import io.readers.RXNFileReader;
import manipulators.ReactionBuilder;
import structure.Resonance;

/**User RDT to map a reaction
 * 
 * @author pplehier
 *
 */
public class RDTMapper {
	
	private static Logger logger			=	Logger.getLogger(RDTMapper.class);
	private static int reactionCounter		=	1;
	private static String folderName		=	FileStructure.getCurrentDir()+StringConstants.OUTPUT+"R";
	private static String fileName			=	"\\ECBLAST_R";
	private static String fileSuffix		=	"_AAM.rxn";
	private static final boolean rdtOutput	=	false;
	private static boolean lastMappingSuccess;
	
	public static void setOutputFolder(String folder){
		
		folderName	=	folder+FileStructure.backslash(new File(folder))+"R";
	}
	/**Generate a reaction ID, based on the current reaction count.
	 * 
	 * @return reaction id
	 */
	private static String reactionID(){
		
		if (reactionCounter < 10){
			return "0000"+reactionCounter;
			
		}else if(reactionCounter < 100){
			return "000"+reactionCounter;
			
		}else if(reactionCounter < 1000){
			return "00"+reactionCounter;
			
		}else if(reactionCounter < 10000){
			return "0"+reactionCounter;
			
		}else {
			return ""+reactionCounter;
		}
	}
		
	/**Map a reaction, organize files into a folder for each reaction, clean up reaction, construct new mapped
	 * reaction and go to next.
	 * If the input reaction has a mapping assigned to it, the mapping is not performed, and a mapped reaction is 
	 * constructed based on the provided mapping (this implies that a hydrogen mapping check is still performed)
	 * Via this method resonance structures are separately mapped and the combination with the minimal number of
	 * changes is returned.
	 * 
	 * @param unmapped reaction
	 * @param name of the reaction
	 * @return mapped reaction
	 */
	public static MappedReaction mapReaction(IReaction rxn, String name){
		
		logger.info("Starting Atom-Atom mapping for "+name);
		long start_time			=	System.nanoTime();
		MappedReaction r		=	new MappedReaction(rxn,false);
		Resonance rs			=	new Resonance(r);
		List<MappedReaction> rr	=	rs.getPossibleReactions();
		int resonanceStructures	=	rr.size();
		
		if(resonanceStructures !=1 ){
			logger.info("Resonance detected for "+name+". Performing additional mapping for "+resonanceStructures+" combinations of resonance structures.");
			logger.info("Resonance with minimal number of changes will be selected.");
		}
		
		if(rxn.getMappingCount() == 0){
		
			boolean reactionExists	=	FileStructure.makeReactionFolder(folderName,reactionID(),name);
			//Data for reaction with minimal changes.
			int minCount			=	0;
			MappedReaction mr		=	new MappedReaction();
			boolean first			=	true;
			int resonance			=	0;
			
			for(int i=0;i<rr.size();i++){
				
				if(!reactionExists){
					executeAAM(rr.get(i).getReaction());
					moveFiles(name);
					renameFiles(name,"_"+i);
				}
				else{
					logger.info("Reaction has been previously mapped, using existing mapping.");
				}
				
				RXNFileReader rfr	=	new RXNFileReader(folderName+reactionID()+"_"+name+fileName+reactionID()+"_"+name+"_"+i+fileSuffix);
				IReaction mappedrxn	=	rfr.toReaction(true);

				ReactionBuilder.fixAromaticity(mappedrxn);
				//No longer required, fixed in RDT
				//ReactionBuilder.revertRadicals(mappedrxn);
				
				int count 				= 	0;
				MappedReaction mrtemp	=	new MappedReaction(mappedrxn);
				lastMappingSuccess		=	mrtemp.allAtomsMapped();
				
				if(lastMappingSuccess && resonanceStructures != 1){
					for(IAtom at:mrtemp.getReactantAtoms()){
						ChangeDetector cd = new ChangeDetector(mrtemp,at);
						if(cd.detectChanged())count++;
					}

					if(first||count<minCount){
						minCount	=	count;
						mr			=	mrtemp;
						first		=	false;
						resonance	=	i;
					}
				}
				else if(resonanceStructures == 1)
					mr	=	mrtemp;
			}
			
			mr.setResonanceStructureNumber(resonance);
			nextReaction();
			
			long end_time		=	System.nanoTime();
			double deltat		=	((long)((end_time - start_time) / 1000000)) / 1000.0;
			
			if(mr.getReaction().getMappingCount() == 0){

				logger.warn("Failed to generate a mapping for "+name+". ");
				logger.warn("Reaction is not processed further and will not be added as family.");
				logger.info("Ended Atom-Atom mapping for "+name+".");
				logger.info("Time spent: " + deltat + " seconds.");			
			}
			else if(!lastMappingSuccess){
				logger.warn("Failed to generate complete mapping for "+name+".");
				logger.warn("Reaction is not processed further and will not be added as family.");
			}else{
				logger.info("Ended Atom-Atom mapping for "+name+".");
				logger.info("Time spent: " + deltat + " seconds.");	
			}

			return mr;
		}
		else{
			logger.info("User defined mapping is present, not performing AAM.");
			ReactionBuilder.fixAromaticity(rxn);
			//No longer required, fixed in RDT
			//ReactionBuilder.revertRadicals(rxn);

			MappedReaction mr	=	new MappedReaction(rxn);
			lastMappingSuccess	=	mr.allAtomsMapped();
			nextReaction();

			long end_time		=	System.nanoTime();
			double deltat		=	((long)((end_time - start_time) / 1000000)) / 1000.0;
			
			logger.info("Ended Atom-Atom mapping for "+name+".");
			logger.info("Time spent: " + deltat + " seconds.");	
			return mr;
		}
	}
	
	/**Map a reaction, but based on the smiles representation of the reaction
	 * 
	 * @param rxn smiles
	 * @param name
	 * @return mapped reaction
	 */
	public static MappedReaction mapReaction(String rxn,String name){
		
		logger.info("Starting Atom-Atom mapping for "+rxn);
		long start_time	=	System.nanoTime();
		boolean reactionExists	=	FileStructure.makeReactionFolder(folderName,reactionID(),name);
		
		if(!reactionExists){
			executeAAM(rxn);
			moveFiles(name);
			renameFiles(name,"");
		}
		else{
			logger.info("Reaction has been previously mapped, using existing mapping.");
		}
		
		RXNFileReader rfr	=	new RXNFileReader(folderName+reactionID()+"_"+name+fileName+reactionID()+"_"+name+fileSuffix);
		IReaction mappedrxn	=	rfr.toReaction(true);
		
		ReactionBuilder.fixAromaticity(mappedrxn);
		//No longer required, fixed in RDT
		//ReactionBuilder.revertRadicals(mappedrxn);
		
		MappedReaction mr	=	new MappedReaction(mappedrxn);
		lastMappingSuccess	=	mr.allAtomsMapped();
		nextReaction();
		
		long end_time		=	System.nanoTime();
		double deltat		=	((long)((end_time - start_time) / 1000000)) / 1000.0;
		
		if(mr.getReaction().getMappingCount() == 0){
			
			logger.warn("Failed to generate a mapping for "+name+". ");
			logger.warn("Reaction is not processed further and will not be added as family.");
			logger.info("Ended Atom-Atom mapping for "+name+".");
			logger.info("Time spent: " + deltat + " seconds.");			
		}
		else if(!lastMappingSuccess){
				logger.warn("Failed to generate complete mapping for "+name+".");
				logger.warn("Reaction is not processed further and will not be added as family.");
		}else{
			logger.info("Ended Atom-Atom mapping for "+name+".");
			logger.info("Time spent: " + deltat + " seconds.");	
		}
		
		return mr;
	}
	
	/**Should be used in combination with map Reaction to check the mapping: more specifically before 
	 * nextReaction() and after moveFiles()
	 * 
	 * @param rxn
	 * @param name
	 */
	protected static void annotateReaction(IReaction rxn, String name){
		
		executeAnnotate(name);
		moveFiles(name);
		renameFiles(name,"");
	}
	
	/**Calls the AAM functionality of RDT and retrieves the logging info. This info is turned off by default
	 * but can be turned on by setting the rdtOutput flag to true
	 * 
	 * @param unmapped reaction
	 */
	private static void executeAAM(IReaction rxn){
		//DO NOT pass .aromatic() smiles to RDT!
		SmilesGenerator sg	=	SmilesGenerator.isomeric();
		String command;
		//Make hydrogen explicit so they are mapped (if necessary).
		
		ReactionBuilder.explicitHydrogen(rxn);
		
		try{
			String smiles	=	"";
			
			//Try to make isomeric smiles, if that fails, create generic smiles.
			try{
				smiles	=	sg.createReactionSMILES(rxn);
				
			}catch(Exception e){
				//TODO:back to isomeric if should be stereo!
				sg		=	SmilesGenerator.generic();
				smiles	=	sg.createReactionSMILES(rxn);
			}
			
			String currentDir	=	FileStructure.getCurrentDir();
			command 			=	"java -jar "+currentDir+StringConstants.RDT+" -Q SMI -q \""+smiles+"\" -g -j AAM -f BOTH -m";
			logger.info(command);
			Process proc;
			proc	=	Runtime.getRuntime().exec(command);
			
			//Let the process terminate, unless gets stuck (no reasonable case should be up for 10 min.), then 
			//kill it and report.
			proc.waitFor(10, TimeUnit.MINUTES);
			
			if(proc.isAlive()){
				proc.destroy();
				logger.error(tools.Output.line("!", 50)+ "\nRDT process terminated due to excessive time usage.\n"+tools.Output.line("!",50));
			}
		    
			// Then retrieve the process output
		    InputStream in	=	proc.getInputStream();
		    InputStream err	=	proc.getErrorStream();

		    byte b[]	=	new byte[in.available()];
		    in.read(b,0,b.length);
		    if(rdtOutput){
		    	logger.info(new String(b));
		    }
		    
		    byte c[]	=	new byte[err.available()];
		    err.read(c,0,c.length);
		    if(rdtOutput){
		    	logger.info(new String(c));
		    }
		    
		} catch (IOException e) {
			logger.error("Incorrect input for AAM execution!");
			e.printStackTrace();
		} catch (InterruptedException e) {
			logger.error("Execution of AAM interrupted!");
			e.printStackTrace();
		}catch (CDKException e) {
			logger.error("Failed to create smiles for reaction number "+reactionCounter);
			e.printStackTrace();
		}
	}
	
	/**Calls the AAM functionality of RDT and retrieves the logging info. This info is turned off by default
	 * but can be turned on by setting the rdtOutput flag to true
	 * 
	 * @param reaction smiles
	 */
	private static void executeAAM(String rxn){
		
		String command;
		
		try{
			String currentDir	=	FileStructure.getCurrentDir();
			command 			=	"java -jar "+currentDir+StringConstants.RDT+" -Q SMI -q \""+rxn+"\" -g -j AAM -f BOTH -m";
			
			Process proc;
			proc	=	Runtime.getRuntime().exec(command);
			proc.waitFor();
			
		    // Then retrieve the process output
		    InputStream in	=	proc.getInputStream();
		    InputStream err	=	proc.getErrorStream();

		    byte b[]	=	new byte[in.available()];
		    in.read(b,0,b.length);
		    if(rdtOutput){
		    	logger.info(new String(b));
		    }

		    byte c[]	=	new byte[err.available()];
		    err.read(c,0,c.length);
		    if(rdtOutput){
		    	logger.info(new String(c));
		    }
		    
		} catch (IOException e) {
			logger.error("Incorrect input for AAM execution!");
			e.printStackTrace();
		} catch (InterruptedException e) {
			logger.error("Execution of AAM interrupted!");
			e.printStackTrace();
		}
	}
	
	/**Call the annotation functionality of RDT. Works only with a .rxn file as input. The specified name when
	 * calling this method must refer to such an .rxn file.
	 * 
	 * @param .rxn file name
	 */
	private static void executeAnnotate(String name){

		String command;
		String filePath=folderName+reactionID()+"_"+name+"\\ECBLAST_R"+reactionID()+"_"+name+"_AAM.rxn";
		
		try{
			String currentDir	=	FileStructure.getCurrentDir();
			command				=	"java -jar "+currentDir+StringConstants.RDT+" -Q RXN -q "+filePath+" -g -j ANNOTATE -f BOTH -m";
			
			Process proc;
			proc	=	Runtime.getRuntime().exec(command);
			proc.waitFor();
			
		    // Then retrieve the process output
		    InputStream in	=	proc.getInputStream();
		    InputStream err	=	proc.getErrorStream();

		    byte b[]	=	new byte[in.available()];
		    in.read(b,0,b.length);
		    if(rdtOutput){
		    	logger.info(new String(b));
		    }
		    
		    byte c[]	=	new byte[err.available()];
		    err.read(c,0,c.length);
		    if(rdtOutput){
		    	logger.info(new String(c));
		    }
		    
		} catch (IOException e) {
			logger.error("ERROR: Incorrect input for AAM execution!");
			e.printStackTrace();
		} catch (InterruptedException e) {
			logger.error("ERROR: Execution of AAM interrupted!");
			e.printStackTrace();
		}
		
	}
	
	/**Moves the files from the working directory to a separate file per reaction.
	 * 
	 * @param reaction name
	 */
	private static void moveFiles(String name){
		
		String current	=	FileStructure.getCurrentDir();
		File working	=	new File(current);
		File[]content	=	working.listFiles();
		
		for(int i = 0;	i < content.length;	i++){
			if(content[i].getName().contains("ECBLAST")){
				File destinationFile	=	new File(folderName+reactionID()+"_"+name+"\\"+content[i].getName());
				Path dest	=	destinationFile.toPath();
				
				try {
					Files.move(content[i].toPath(), dest,StandardCopyOption.REPLACE_EXISTING);
				} catch (IOException e) {
					logger.error("Failed to move file "+content[i].getPath());
					e.printStackTrace();
				}
			}
		}
	}
	
	/**Renames the files generated by RDT from the default ECBLAST_smiles_AAM to 
	 * ECBLAST_reationID_reaction name_AAM
	 * 
	 * @param reaction name
	 */
	private static void renameFiles(String name, String suffix){
		
		String current	=	folderName+reactionID()+"_"+name+"\\";
		
		File rxpng		=	new File(current+"ECBLAST_smiles_AAM.png");
		rxpng.renameTo(new File(current+"ECBLAST_R"+reactionID()+"_"+name+suffix+"_AAM.png"));
		
		File rxrxn		=	new File(current+"ECBLAST_smiles_AAM.rxn");
		rxrxn.renameTo(new File(current+"ECBLAST_R"+reactionID()+"_"+name+suffix+"_AAM.rxn"));
		
		File rxtxt		=	new File(current+"ECBLAST_smiles_AAM.txt");
		rxtxt.renameTo(new File(current+"ECBLAST_R"+reactionID()+"_"+name+suffix+"_AAM.txt"));
		
		File rxxml		=	new File(current+"ECBLAST_smiles_AAM.xml");
		rxxml.renameTo(new File(current+"ECBLAST_R"+reactionID()+"_"+name+suffix+"_AAM.xml"));
		
		File rxpng2		=	new File(current+"ECBLAST_smiles_ANNONATE.png");
		rxpng2.renameTo(new File(current+"ECBLAST_R"+reactionID()+"_"+name+suffix+"_ANNOTATE.png"));
		
		File rxrxn2		=	new File(current+"ECBLAST_smiles_ANNONATE.rxn");
		rxrxn2.renameTo(new File(current+"ECBLAST_R"+reactionID()+"_"+name+suffix+"_ANNOTATE.rxn"));
		
		File rxtxt2		=	new File(current+"ECBLAST_smiles_ANNONATE.txt");
		rxtxt2.renameTo(new File(current+"ECBLAST_R"+reactionID()+"_"+name+suffix+"_ANNOTATE.txt"));
		
		File rxxml2		=	new File(current+"ECBLAST_smiles_ANNONATE.xml");
		rxxml2.renameTo(new File(current+"ECBLAST_R"+reactionID()+"_"+name+suffix+"_ANNOTATE.xml"));
	}

	/**Proceed to next reaction
	 * 
	 */
	private static void nextReaction(){
		
		reactionCounter++;
	}
	
	public static boolean lastMappingSucces(){
		
		return lastMappingSuccess;
	}
}