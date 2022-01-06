package io.readers;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import org.apache.log4j.Logger;

import constants.StringConstants;

public class TXTFileReader {

	private static Logger logger	=	Logger.getLogger(TXTFileReader.class);
	private String filePath;
	
	public TXTFileReader(String filePath){
		
		this.filePath	=	filePath;
	}
	
	/**Count the number of lines in a file
	 * 
	 * @return number of lines
	 */
	public int countLines(){
		
		int iCurrentLine	=	0;
		
		try{
			BufferedReader reader	=	new BufferedReader(new FileReader(this.filePath));
		
			while ((reader.readLine())	!=	null){
				iCurrentLine++;
			}
			reader.close();
		} 
		catch(IOException e){}
		
		return iCurrentLine;
	}

	/**Get the line number after the specified index at which the first occurence of the specified keyword is 
	 * encountered
	 * 
	 * @param start index
	 * @param Keyword
	 * @return
	 */
	public int getLineNumberNextKey(int index, String Keyword){
		
		int out	=	this.countLines();
		
		if (index <= 0){
			logger.fatal("Zero or negative index.");
			System.exit(-1);
		}
		else {
			try{
				String sCurrentLine;
				BufferedReader reader	=	new BufferedReader(new FileReader(this.filePath));
				int iCurrentLine		=	0;
				boolean found			=	false;
				
				while ((sCurrentLine = reader.readLine()) != null && !found ){
					found	=	iCurrentLine > index && sCurrentLine.contains(Keyword);
					
					if(found){
						out	=	iCurrentLine;
					}
					
					iCurrentLine++;
				}
				
				reader.close();
			} 
			catch(IOException e){}
		}
		
		return out+1;
		
	}

	/**Read an entire file to string format
	 * 
	 * @return file in string format
	 */
	public String read(){
		
		String out	=	StringConstants.EMPTY;
		
		try{
			String sCurrentLine;
			BufferedReader reader	=	new BufferedReader(new FileReader(this.filePath));
			
			while ((sCurrentLine = reader.readLine()) != null) {
				out	+=	sCurrentLine + "\n";
			}
			
			reader.close();
		} 
		catch (IOException e) {}
		
		return out;
	}

	/**Read a section of a file <br>
	 * -'-' specifies all lines before and including the given index
	 * -'+' specifies all lines after and including the given index
	 * 
	 * @param index
	 * @param option
	 * @return
	 */
	public String read(int index,char option){
		
		String out	=	StringConstants.EMPTY;
		
		if (index <= 0){
			logger.fatal("Zero or negative index; not reading file.");
			System.exit(-1);
		}
		
		else if(option == '+'){
			try{
				String sCurrentLine;
				BufferedReader reader	=	new BufferedReader(new FileReader(this.filePath));
				int iCurrentLine		=	0;
				
				while ((sCurrentLine = reader.readLine()) != null){
					if(iCurrentLine >= index - 1){
						out	+=	sCurrentLine + "\n";
					}
					iCurrentLine++;
				}
				
				reader.close();
			} 
			catch(IOException e){}
		}
		
		else if(option == '-'){
			try{
				String sCurrentLine;
				BufferedReader reader	=	new BufferedReader(new FileReader(this.filePath));
				int iCurrentLine		=	0;
				
				while ((sCurrentLine = reader.readLine()) != null && iCurrentLine <= index - 1){
					out	+=	sCurrentLine + "\n";
					iCurrentLine++;
				}
				
				reader.close();
			} 
			catch(IOException e){}
		}
		
		else{
			logger.fatal("Wrong option specified; not reading file.");
			System.exit(-1);
			}
		
		return out;
	}

	/**Read the lines within a specified range
	 * 
	 * @param start
	 * @param end
	 * @return text between start and end
	 */
	public String read(int start, int end){
		
		String out	=	StringConstants.EMPTY;;
		
		if(start > end){
			logger.fatal("Start index > End index; not reading file.");
			System.exit(-1);
		}
		
		else if (start <= 0 || end <= 0){
			logger.fatal("Zero or negative index; not reading file.");
			System.exit(-1);
		}
		
		else{
			try{
				String sCurrentLine;
				BufferedReader reader	=	new BufferedReader(new FileReader(this.filePath));
				int iCurrentLine		=	0;
				
				while ((sCurrentLine=reader.readLine()) != null){
					if(iCurrentLine>=start-1&&iCurrentLine <= end-1){
					out	+=	sCurrentLine + "\n";
					}
				
				iCurrentLine++;
				}
				
				reader.close();
			} 
			catch(IOException e){}
		}
		
		return out;
	}

	/**Read a line of the file, containing the specified keyword and starting after the specified index
	 * 
	 * @param start index
	 * @param Keyword to be read
	 * @return first line containing keyword after the index
	 */
	public String read(int index,String Keyword){
		
		String out	=	StringConstants.EMPTY;
		
		if (index <= 0){
			logger.fatal("Zero or negative index; not reading file.");
			System.exit(-1);
		}
		
		else {
			try{
				String sCurrentLine;
				BufferedReader reader	=	new BufferedReader(new FileReader(this.filePath));
				int iCurrentLine		=	0;
				boolean found			=	false;
				
				while ((sCurrentLine=reader.readLine()) != null && !found){
					found	=	iCurrentLine >= index - 1 && sCurrentLine.contains(Keyword);
					
					if(found){
						out	=	sCurrentLine;
					}
					
					iCurrentLine++;
				}
				
				reader.close();
			} 
			catch(IOException e){}
		}
		
		return out;
	}

	/**Read an entire file to a string array per line
	 * 
	 * @return The lines of the file
	 */
	public String[] readArray(){
		
		String[] out	=	new String[this.countLines()];
		
		try{
			String sCurrentLine;
			BufferedReader reader	=	new BufferedReader(new FileReader(this.filePath));
			int iCurrentLine		=	0;
			
			while ((sCurrentLine = reader.readLine()) != null) {
				out[iCurrentLine]	=	sCurrentLine;
				iCurrentLine++;
			}
			
			reader.close();
		} 
		catch (IOException e) {}

		return out;
	}
	
	/**Count how many lines contain a given keyword in the text file.
	 * 
	 * @param keyword
	 * @return hits (per line)
	 */
	public int countOccurences(String keyword){
		
		int count	=	0;
		try{
			String sCurrentLine;
			BufferedReader reader	=	new BufferedReader(new FileReader(this.filePath));
			@SuppressWarnings("unused")
			int iCurrentLine		=	0;
			
			while ((sCurrentLine = reader.readLine()) != null) {
				iCurrentLine++;
				if(sCurrentLine.contains(keyword))
					count++;
			}
			
			reader.close();
		} 
		catch (IOException e) {}
		
		return count;
	}
}