package fileGeneration.generators;

import constants.*;
import io.FileStructure;
import io.XMLValidator;
import io.readers.xmlClasses.Config;
import io.writers.GenericFileWriter;
import tools.Output;

import java.io.File;
import java.util.List;

import javax.xml.XMLConstants;
import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Marshaller;
import javax.xml.validation.Schema;
import javax.xml.validation.SchemaFactory;

import org.apache.log4j.Logger;
import org.xml.sax.SAXException;

/**This class contains the actual file generator of the ReactionFamilyGenerationTool.
 * It allows the user to generate Genesys-compatible reaction_families.xml files based on either:
 * Text-based input of reactions (separate smiles for reactants and products
 * Rxn-file input of the reactions, for which the directory containing the files should be specified
 * The output is presented in the outputFileGen folder and requires the ReactionDecoderTool (RDT) to be
 * present in the same directory
 * 
 * @author pplehier
 *
 */
public class FileGenerator {
	
	private static Logger logger=Logger.getLogger(FileGenerator.class);
	private boolean outputCheck	=	true;
	private boolean zeroKinetics=	false;
	private boolean constraint	=	true;
	private boolean specific	=	false;
	private File ReactionFamilies;
	private File workDir;
	private File outputDir;
	private File input;
	private File output;
	private File summary;
	private int fails;
	private int identical;
	private int reverse;
	private String failed;
	private int unique;
	private int total;
	private int reacts;
	
	/**Generate a reaction_families.xml file.<br>
	 * N.B.1:<br>
	 * -Constraints are hard-coded: max 1 radical, max 6 carbon<br>
	 * -Fails to generate for stereo chemistry changes(but changes are detected, <br>
	 * 	as Genesys does not consider this an elementary change<br>
	 * -Input arg can be either a file (.txt) or a folder (containing .rxn files).<br>
	 * 
	 * N.B.2:<br>
	 * -Requires RDT1.4.2.jar to be in the same folder.<br>
	 * -Remove or rename output folder before re-running<br>
	 * 
	 * TODO:Expansion of genesys to allow hydrogen-unbalanced reactions: to solve the problem of many organic
	 * syntheses not reporting hydrogen sources.
	 * @param args
	 */
	public void run(){
		
		String inputPath		=	this.input.getAbsolutePath();
		String outputPath		=	this.output.getAbsolutePath();
		
		this.checkPresence();
		
		logger.info("Writing reaction families to: "+outputPath);
		
		FileStructure.makeOutputFolder(this.outputDir.getPath());
		
		this.generateFile(inputPath,outputPath);
		
		logger.info(Output.line("-",50)+
					"\nGenerated "+unique+" unique famil"+((unique == 1)?"y":"ies")+" from a total of "
									+total+" reaction"+((total == 1)?"":"s")+"."+
					"\nIgnored "+this.reacts+" reaction"+((this.reacts == 1)?"":"s")+" because they contained more than 2 reactants."+
					"\nDetected "+this.reverse+" reversible reaction"+((this.reverse == 1)?"":"s")+
					"\nDetected "+this.identical+" identical reaction"+((this.identical == 1)?"":"s")+
					"\nFailed mapping for "+this.fails+" reaction"+((this.fails == 1)?"":"s")+":"+
					"\n"+this.failed);
		
		FileStructure.moveLogFile(this.outputDir.getPath());
	}
	
	/**Create a new generator
	 * 
	 * @param inputFileName
	 * @param outputFileName
	 */
	public FileGenerator(String inputFileName, String outputFileName){
		
		String workingDir	=	FileStructure.getCurrentDir();
		String outFileName	=	checkOutputValidity(outputFileName);
		this.workDir		=	new File(workingDir);
		this.input			=	new File(workingDir+inputFileName);
		this.output			=	new File(outFileName);
		this.fails			=	0;
		this.failed			=	"";
	}
	
	/**Create a new generator (with specified output directory)
	 * 
	 * @param inputFileName
	 * @param outputFileName
	 */
	public FileGenerator(String inputFileName, String outputFileName, String outputDirectoryName){
		
		String workingDir	=	FileStructure.getCurrentDir();
		String outFileName	=	checkOutputValidity(outputFileName);
		this.workDir		=	new File(workingDir);
		this.input			=	new File(workingDir+inputFileName);
		this.outputDir		=	new File(outputDirectoryName);
		this.output			=	new File(this.outputDir.getPath()+outFileName);
		this.summary		=	new File(this.outputDir.getPath()+"summary.csv");
		this.fails			=	0;
		this.failed			=	"";
	}
	
	/**Set the output directory, overwrites any existing info
	 * 
	 * @param outputDirectory
	 */
	public void setOutputDirectory(String outputDirectory){
		
		this.outputDir		=	new File(outputDirectory);
		this.output			=	new File(this.outputDir.getPath()+FileStructure.backslash(this.outputDir)+this.output.getName());
		this.summary		=	new File(this.outputDir.getPath()+FileStructure.backslash(this.outputDir)+"summary.csv");
	}
	
	/**Set whether or not the existence of the output directory should  be checked.
	 * 
	 * @param check
	 */
	public void setOutputCheck(boolean check){
		
		this.outputCheck	=	check;
	}
	
	public void setZeroKinetics(boolean kin){
		
		this.zeroKinetics	=	kin;
	}
	
	public void setConstraint(boolean constraint){
		
		this.constraint		=	constraint;
	}
	
	public void setSpecific(boolean specific){
		
		this.specific	=	specific;
	}
	/**Generate the .xml file, using the file named input as input and writing to the file named output.
	 * If the arguments pass a directory to this method, it will be assumed that a .rxn file is present in 
	 * this directory for each reaction
	 * If the arguments pas a file to this method, it will be assumed that the input is in text format using
	 * keywords (and tabs) to specify conditions
	 * 
	 * @param input file name
	 * @param output file name
	 */
	private void generateFile(String input,String output){
		
		//Determine input type
		if(new File(input).isDirectory()){
			logger.info("Reading input from folder: "+input+
						"\nInput format is .rxn files.\n");
		}
		else if(input.endsWith(".txt")){
			logger.info("Reading input from file: "+input+
						"\nInput format is .txt file.\n");
		}
		else if(input.endsWith(".rxn")){
			logger.info("Reading input from file: "+input+
						"\nInput format is .rxn file.\n");
		}
		else if(input.endsWith(".inp")){
			logger.info("Reading input from file: "+input+
						"\nInput format is chemkin input file.\n");
		}
		
		ReactionFamilies			=	new File(output);
		Config config				=	new Config();
		List<Object> generationOut	=	ChemistryGenerator.generate(config, ReactionFamilies.getParent(), input, zeroKinetics, constraint, specific);
		
		this.total		=	(int) generationOut.get(1);
		this.unique		=	(int) generationOut.get(2);
		this.fails		=	(int) generationOut.get(3);
		this.failed		=	(String) generationOut.get(4);
		this.reacts		=	(int) generationOut.get(5);
		this.reverse	=	(int) generationOut.get(6);
		this.identical	=	(int) generationOut.get(7);
		GenericFileWriter.writeFile(this.summary, (StringBuffer) generationOut.get(8));
		
		try {
			JAXBContext jaxbContext = JAXBContext.newInstance(StringConstants.RFXMLPACKAGE);
			Marshaller jaxbMarshaller = jaxbContext.createMarshaller();
			//make sure that the output file content is nicely formatted
			jaxbMarshaller.setProperty(Marshaller.JAXB_FORMATTED_OUTPUT, true);
			SchemaFactory schemaFact	=	SchemaFactory.newInstance(XMLConstants.W3C_XML_SCHEMA_NS_URI);
			Schema schema				=	schemaFact.newSchema(XMLValidator.class.getResource(StringConstants.RFDATA+StringConstants.RFSCHEMANAME));

			//make sure the input follows the correct schema
			jaxbMarshaller.setSchema(schema);
			jaxbMarshaller.marshal(config, ReactionFamilies);

		} catch (JAXBException e) {
			e.printStackTrace();
		} catch (SAXException e) {
			e.printStackTrace();
		}
	}
	
	/**Check whether the given output name in the run-arguments does indeed specify an .xml file.<br>
	 * If it doesn't either:<br>
	 * No extension is provided: .xml is added<br>
	 * A wrong extension is provided: the program informs and exits.<br>
	 * 
	 * @param Output file name
	 * @return
	 */
	private static String checkOutputValidity(String Output){
		
		boolean format	=	Output.endsWith(StringConstants.XML);
		
		if(format){
			return Output;
		}
		
		else{
			String[] suffix	=	Output.split(".");
			
			if(suffix.length < 2){
				return Output.concat(StringConstants.XML);
			}
			
			else 
				if(suffix.length > 2){
				logger.fatal("Invalid output file name: "+Output+"!");
				System.exit(-1);
				}
			
				else{
					return suffix[0].concat(StringConstants.XML);
				}
			}
		
		return Output;
	}
	
	/**Check whether there is no outputFileGen folder in the working directory and whether rdt is available 
	 * in the same directory (specifically under the name StringConstants.RDT) <br>
	 * If either condition is not fulfilled, the program exits.
	 */
	private void checkPresence(){
		
		boolean flag	=	false;
		File rdtJar		=	new File(this.workDir.getPath()+"\\"+StringConstants.RDT);
		
		if(!FileStructure.contains(workDir, rdtJar)){
			logger.fatal("Working directory "+workDir.getPath()+" does not contain "+StringConstants.RDT+"... ");
			flag	=	true;
		}
		//Check whether the directory contains the correct RDT jar!
		if(outputCheck){
			if(outputDir.isDirectory()){
				logger.fatal("Output directory for file generation already exists ... "
						   + "\nRemove or rename folder!");
				flag	=	true;
			}
		}
		else{
			logger.warn("Not checking existence of output directory. Might overwrite data.");
		}
		
		if(flag){
			logger.fatal("Aborting ...");
			System.exit(-1);
		}
	}
}