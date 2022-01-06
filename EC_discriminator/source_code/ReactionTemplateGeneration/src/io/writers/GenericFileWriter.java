package io.writers;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import org.apache.log4j.Logger;

/**This class defines a basic file writer
 * 
 * @author pplehier
 *
 */
public class GenericFileWriter {

	private static Logger logger	=	Logger.getLogger(GenericFileWriter.class);
	
	/**Write a file from string to file
	 * 
	 * @param file
	 * @param content
	 */
	public static void writeFile(File file, String content){
		
		try {
			FileWriter out_f	=	new FileWriter(file);
			out_f.write(content);
			out_f.close();
		} catch (IOException e) {
			logger.fatal("Failed to write output file!");
			System.exit(-1);
		}
	}
	
	public static void writeFile(File file, StringBuffer content){
		
		try{
			FileWriter out_f	=	new FileWriter(file);
			out_f.write(content.toString());
			out_f.close();
		} catch(IOException e){
			logger.fatal("Failed to write output file!");
			System.exit(-1);
		}
	}
}