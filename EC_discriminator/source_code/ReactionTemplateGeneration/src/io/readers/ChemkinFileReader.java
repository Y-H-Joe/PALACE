package io.readers;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.log4j.Logger;

/**Read reactions from a chemkin file. Requires the chemkin file to list all species below each other
 * and for them to be identified by a standard identifier (smiles or inchi) as comment in the same row.
 * 
 * @author pplehier
 *
 */
public class ChemkinFileReader {

	private static Logger logger	=	Logger.getLogger(ChemkinFileReader.class);
	
	private String filePath;
	private List<String> speciesNames;
	private List<String> speciesIdentifiers;
	private List<String> reactionNames;
	private Map<String,String> namesToIdentifiers;
	
	public ChemkinFileReader(String path){
		
		this.filePath			=	path;
		this.speciesNames		=	new ArrayList<String>();
		this.speciesIdentifiers	=	new ArrayList<String>();
		this.reactionNames		=	new ArrayList<String>();
		this.namesToIdentifiers	=	new HashMap<String,String>();
	}
	
	public void read(){
		
		try{
			String sCurrentLine;
			BufferedReader reader	=	new BufferedReader(new FileReader(this.filePath));
			boolean species			=	false;
			boolean reactions		=	false;
			while ((sCurrentLine = reader.readLine()) != null) {
			
				//Keep track of which block the reader is in.
				if(species && sCurrentLine.equals("END")){
					species	=	false;
				}
				if(reactions && sCurrentLine.equals("END")){
					reactions	=	false;
				}
				
				//Read species
				if(species){
					readSpecie(sCurrentLine);
				}
				
				if(reactions){
					readReaction(sCurrentLine);
				}
				
				if(sCurrentLine.equals("SPEC")||sCurrentLine.equals("SPECIES")){
					species	=	true;
				}
				
				if(sCurrentLine.contains("REACTION")){
					reactions	=	true;
				}
				
			}
			
			reader.close();
		} 
		catch (IOException e) {}
		
	}
	
	private void readSpecie(String sCurrentLine){

		if(sCurrentLine.equals("")||sCurrentLine.startsWith("!!")){
			//skip if empty or comment line.
		}
		else{
			String name	=	sCurrentLine.split("!!")[0].trim();
			String id	=	sCurrentLine.split("!!")[1].trim();
			//Get rid of all trailing info in the ID.
			id			=	id.split(" ")[0];
			speciesNames.add(name);
			speciesIdentifiers.add(id);
			
			if(namesToIdentifiers.containsKey(name)&&!id.equals(namesToIdentifiers.get(name))){
				logger.warn("Duplicate species entries with different ID's encountered for "+name);
				logger.warn("IDs are "+id+" and "+namesToIdentifiers.get(name)+".");
			}
			
			namesToIdentifiers.putIfAbsent(name, id);
		}
	}
	
	private void readReaction(String sCurrentLine){
		
		if(sCurrentLine.equals("")||sCurrentLine.startsWith("!")||!sCurrentLine.contains("=")){
			//skip if empty line, line is comment or line does not contain reaction equation.
		}
		else{
			String name	=	sCurrentLine.split(" ")[0];
			if(name.equals("")){}
			else reactionNames.add(name);
		}
	}
	
	public List<String> getReactionNames(){
		
		return this.reactionNames;
	}
	
	public List<String> getSpeciesNames(){
		
		return this.speciesNames;
	}
	
	public List<String> getSpeciesIdentifiers(){
		
		return this.speciesIdentifiers;
	}
	
	public Map<String,String> getNamesToIdentifiers(){
		
		return this.namesToIdentifiers;
	}
}
