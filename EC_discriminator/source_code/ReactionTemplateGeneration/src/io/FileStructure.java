package io;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

import org.apache.log4j.Logger;

import tools.Tools;

/**This class contains methods corresponding to file access, manipulation, ...
 * 
 * @author pplehier
 *
 */
public class FileStructure {
	
	private static Logger logger			=	Logger.getLogger(FileStructure.class);
	private static char[] prohibitedChars	=	{'/','\\','\"',':','?','*','<','>','|'};
	
	/**Get the current working directory
	 * 
	 * @return working directory
	 */
	public static String getCurrentDir(){
		
		String current	=	"";
		
		try {
			current	=	new File(".").getCanonicalPath();
		} catch (IOException e) {
			logger.error("Failed to retrieve working directory.");
			e.printStackTrace();
		}
		
		return current+"\\";
	}
	
	/**Make the outputFileGen folder
	 */
	public static void makeOutputFolder(String name){
		
		File output	=	new File(name);
		output.mkdir();
	}
	
	/**Make a separate folder for each reaction, which will contain the output of the RDT
	 * 
	 * @param ReactionID
	 * @param name
	 */
	public static boolean makeReactionFolder(String folderNameR, String ReactionID,String name){

		File output	=	new File(folderNameR+ReactionID+"_"+name);
		if(output.exists() && output.listFiles().length != 0){
			return true;
		}
		else{
			output.mkdirs();
			return false;
		}
	}
	
	/**Make a folder with the given name
	 * 
	 * @param name
	 */
	public static void makeFolder(String name){
		
		File output	=	new File(name);
		output.mkdirs();
	}
	
	/**Move the log file from the working directory to the output directory
	 */
	public static void moveLogFile(String destFolder){
		
		String logFileName	=	getLogFileName();
		String currentDir	=	getCurrentDir();
		
		if(logFileName.equals("")){
			logger.info("No log file found.");
			return;
		}
		
		File logFile	=	new File(currentDir+logFileName);
		File logFileDest=	new File(destFolder+"logfile");
		
		try {
			Files.copy(logFile.toPath(), logFileDest.toPath(), StandardCopyOption.REPLACE_EXISTING);
		} catch (IOException e) {
			logger.warn("Failed to move logFile.\n Logfile still located in: "+currentDir+logFileName);
			e.printStackTrace();
		}
	}
	
	/**Get the (user-defined) name of the logFile. 
	 * 
	 * @return name of the logFile
	 */
	private static String getLogFileName(){
		
		String name			=	"";
		File workingDir		=	new File(getCurrentDir());
		File[] listOfFiles	=	workingDir.listFiles();
		
		for(int i = 0;	i<listOfFiles.length;	i++){
			if(listOfFiles[i].getName().contains("log")
			   &&
			   !listOfFiles[i].getName().contains("log4j")){
				name	=	listOfFiles[i].getName();
			}
		}

		return name;
	}
	
	/**Check whether a directory contains a file
	 * 
	 * @param directory
	 * @param file
	 * @return does directory contain file
	 */
	public static boolean contains(File directory,File file){
		
		File[] filesList	=	directory.listFiles();
		
		for(int i = 0;	i < filesList.length;	i++){
			if(filesList[i].getName().equals(file.getName())){
				return true;
			}
		}
		
		return false;
	}
	
	public static boolean isProhibited(String search){
		
		return Tools.containsAsSub(prohibitedChars, search); 
	}
	
	public static String removeProhibited(String name){
		
		String newName	=	name;
		
		for(int i = 0;	i < prohibitedChars.length;	i++){
			if(newName.contains(""+prohibitedChars[i])){
				newName	=	replace(prohibitedChars[i], '_', newName);
			}
		}
		
		return newName;
	
	}
	
	private static String replace(char ch, char replacement, String name){
		
		int pos			=	name.indexOf(ch);
		String newName	=	name;	
		
		while(pos != -1){
			
			String front	=	newName.substring(0, pos);
			String end		=	newName.substring(pos+1);
			newName			=	front + replacement + end;
			pos				=	newName.indexOf(ch, pos);
		}
		
		return newName;
	}
	
	/**This method checks whether a desired file name does not contain any illegal characters.
	 * If it does, these characters are replaced by '_'.
	 * 
	 * @param fileName
	 */
	public static String validateFileName(String fileName){
		
		String newFileName	=	fileName;
		
		if(FileStructure.isProhibited(newFileName)){
			newFileName		=	FileStructure.removeProhibited(newFileName);
			logger.info("Changed original file name \""+fileName+"\" to \""+newFileName+"\" because it contained illegal characters.");
		}
		
		return newFileName;
	}
	
	/**Returns a backslash if the current path string doesn't contain one
	 * 
	 * @param dir
	 * @return
	 */
	public static String backslach(String dir){
		
		char last	=	dir.charAt(dir.length() - 1);
		
		if(last == '\\')
			return "";
		else
			return "\\";
	}
	
	/**Returns a backslash if the files path does not end on one
	 * 
	 * @param dir
	 * @return
	 */
	public static String backslash(File dir){
		
		char last	=	dir.getPath().charAt(dir.getPath().length()-1);
		
		if(last == '\\')
			return "";
		else
			return "\\";
	}
	
	/** Delete a directory (and all subdirectories)
	 * 
	 * @param file
	 */
	public static void deleteDir(File file){
		
		if(file.isDirectory()){
			for(File subFile:file.listFiles()){
				deleteDir(subFile);
			}
		}
		
		file.delete();
	}
	
	/**Delete a directory (and all subdirectories)
	 * 
	 * @param name
	 */
	public static void deleteDir(String name){
		
		File dir	=	new File(name);
		
		deleteDir(dir);
	}
	
	/**Rename a directory
	 * 
	 * @param originalName
	 * @param newName
	 */
	public static void rename(String originalName, String newName){
		
		File dir	=	new File(originalName);
		File newDir	=	new File(newName);
		
		if(dir.exists() && dir.isDirectory()){
			dir.renameTo(newDir);
		}
	}
	
	/**Construct a filter that finds all folders that start with R0...
	 * 
	 * @return
	 */
	public static FilenameFilter getRXNFolderFilter(){
		return new FilenameFilter() {
			public boolean accept(File dir, String name) {
				
				String lowercaseName = name.toLowerCase();
		
				if (lowercaseName.startsWith("r0") && (new File(dir.getAbsolutePath()+"\\"+name)).isDirectory()) {
					return true;
				} else {
					return false;
				}
			}
		};
	}
	
	/**Construct a filter that finds all .rxn files
	 * 
	 * @return
	 */
	public static FilenameFilter getRXNFileFilter(){
		return new FilenameFilter() {
			public boolean accept(File dir, String name) {
				
				String lowercaseName = name.toLowerCase();
				
				if (lowercaseName.endsWith(".rxn")) {
					return true;
				} else {
					return false;
				}
			}
		};
	}
}