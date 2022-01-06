package constants;

public class StringConstants {

	// env variables
	public static final String DATABASES_HOME = "DATABASES_HOME";
	
	// files:	
	public static final String CALCTRACK = "calculations_track";
	public static final String FAILURES = "failures";
	public static final String DATABASE = "database";
	public static final String JAVA_BINARIES = "JAVA_BINARIES";	
	public static final String KEYWORDS = "keywords";
	public static final String ALL_THERMO = "all_thermo";
	public static final String SPECIES_THERMO = "species_thermo";
	public static final String ALL_KINETICS = "all_kinetics";
	public static final String RUNNING_CALCULATIONS = "running_calculations";
	public static final String TODO_CALCULATIONS = "todo_calculations";
	public static final String FINISHED_CALCULATIONS = "finished_calculations";
	public static final String OPTIONS = "options";
	public static final String SUMMARY = "summary";
	public static final String VARIABLES = "variables";
	public static final String CONSTANTS= "constants";
	public static final String END = "end";
	public static final String IDS = "ids";
	public static final String IMassymInput = "temp_imassym_input";
	public static final String IMassymOutput = "temp_imassym_output";
	public static final String EigenValuesQSCInput = "temp_eigenValuesQSC_input";
	public static final String EigenValuesQSCOutput = "QSC_output";
	public static final String CoordinatesInput = "coordinates_input";
	public static final String RESTART = "_restart";
	public static final String THERMO = "_thermo";
	public static final String SOURCE = "_source";
	public static final String LIBRARY = "library";
	public static final String FAILED = "failed_calculations";
	public static final String EP = "EvansPolanyi";
	public static final String RFXMLPACKAGE = "io.readers.xmlClasses";
	public static final String RFDATA = "data/";
	public static final String RFSCHEMANAME = "reaction-families.xsd";
	public static final String BENSONXMLPACKAGE = "thermo.bensonGA.xmlClasses";
	public static final String BURCATXMLPACKAGE = "thermo.libraries.digesters.xmlClasses.burcat";
	public static final String NINEPOINTSXMLPACKAGE = "thermo.libraries.digesters.xmlClasses.ninePoints";
	public static final String NASAXMLPACKAGE = "thermo.libraries.digesters.xmlClasses.nasa";
	public static final String ANALYZERXMLPACKAGE = "readers.xmlClasses";
	
	// extensions: 
	public static final String XML = ".xml";
	public static final String LOG = ".log";
	public static final String COM = ".com";
	public static final String INP = ".inp";
	public static final String OUT = ".out";
	public static final String TXT = ".txt";
	public static final String PBS = ".pbs";
	public static final String ERR = ".err";
	
	// separators and punctuation marks
	public static final String TAB = "\t";
	public static final String NEW_LINE = "\n";
	public static final String SPACE = " ";
	public static final String EQUAL = "=";
	public static final String SLASH = "/";
	public static final String DASH = "-";
	public static final String BSLASH = "\\";
	public static final String COLON = ":";
	public static final String COMMA = ",";
	public static final String OPEN_BRACKETS = "(";
	public static final String CLOSE_BRACKETS = ")";
	public static final String EXCLAMATION = "!";
	

	// elements
	public static final String HYDROGEN = "H";
	public static final String CARBON = "C";
	public static final String OXYGEN = "O";
	public static final String NITROGEN = "N";
	
	
	public static final String OUTPUT="outputFileGen\\";
	public static final String AEND="/>\n";
	public static final String BEND=">\n";
	public static final String HEADER="<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<config>\n";
	public static final String TEMPERATURE_START="\t<inp-temperature>";
	public static final String TEMPERATURE_END="</inp-temperature>\n";
	public static final String REACTIONFAMILY_START="\t<inp-reaction-family ";
	public static final String REACTIONFAMILY_B_END=BEND;
	public static final String REACTIONFAMILY_END="\t</inp-reaction-family>\n\n";
	public static final String BIMOLECULAR_START="\t\t<bimolecular>";
	public static final String BIMOLECULAR_END="</bimolecular>\n";
	public static final String RECIPE_START="\t\t<inp-recipe>\n";
	public static final String RECIPE_END="\t\t</inp-recipe>\n";
	public static final String TRANSFORMATION_START="\t\t\t<inp-transformation ";
	public static final String TRANSFORMATION_END=AEND;
	public static final String REACTANT_START="\t\t<inp-reactant ";
	public static final String REACTANT_B_END=BEND;
	public static final String REACTANT_END="\t\t</inp-reactant>\n";
	public static final String REACTIVECENTER_START="\t\t\t<inp-reactive-center ";
	public static final String REACTIVECENTER_END=AEND;
	public static final String MOLECULECONSTRAINT_START="\t\t\t<inp-molecule-constraint ";
	public static final String MOLECULECONSTRAINT_END=AEND;
	public static final String KINETICS_START="\t\t<inp-kinetics ";
	public static final String KINETICS_B_END=BEND;
	public static final String KINETICS_END="\t\t</inp-kinetics>\n";
	public static final String RATECOEF_START="\t\t\t<rate-coeff ";
	public static final String RATECOEF_END=AEND;
	public static final String FOOTER="</config>\n";
	
	//ReactionCenterDetection
	public static final String RDT="RDT1.5.jar";
	public static final String MAPPINGPROPERTYATOM="Corresponding Atom";
	public static final String MAPPINGPROPERTY="Corresponding Atom Index";
	public static final String HYDROGENCHANGE="Hydrogens changed";
	public static final String HYDROGENDIFFERENCE="Difference between reactant and product implicit hydrogens";
	public static final String NEIGHBOURCHANGE="Neighbours changed";
	public static final String SINGLEELECTRONCHANGE="Single electrons changed";
	public static final String BECAMERING="Became ring atom";
	public static final String NEIGHBOURSTEREO="Neighbour's stereo has changed";
	public static final String STEREOCHANGE="Stereo changed";
	public static final String BONDCHANGE="Bonds changed";
	public static final String CHARGECHANGE="Charges changed";
	public static final String CHANGED="Atom has changed";
	public static final String SHOULDKEEP="Atom should be in Reaction Center";
	public static final String LINK="Links two reaction centers";
	public static final String RECIPECENTER="Recipe center ID";
	public static final String HYDROGENUSED="Hydrogen has been used for bond formation";
	public static final String AVOIDEMPTY="Avoid empty reactive center in xml";
	public static final String OLDHydrogenCount="Old implicit hydrogencount";
	public static final String EMPTY="";
	public static final String ALL	="ALL";
}
