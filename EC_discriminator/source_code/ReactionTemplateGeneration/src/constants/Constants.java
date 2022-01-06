package constants;

public class Constants {

	// Pi
	public static final double Pi = 3.141592654;
	// speed of light
	public static final double c = 2.99792458E10;
	// Planck constant
	public static final double h = 6.626068E-34;
	// Planck constant
	public static final double h_bar = 1.0545E-34;
	// Boltzmann constant
	public static final double kb = 1.38066E-23;
	// Ideal gas constant
	public static final double R = 8.314472;
	// conversion from bohr to meter
	public static final double bohr2meter=5.29177249E-11;
	// conversion from amu to kg
	public static final double amu2kg = 1.660538E-27;
	// conversion from hartree to joules
	public static final double hartree2joules = 4.35974417E-18;
	// coversion from hartree to kjoules/mol
	public static final double hartree = 2625.5;
	// conversion from atm to pascal
	public static final double atm2pascal = 101325;
	// conversion from calories to joules
	public static final double cal2joules = 4.184;
	// maximum number of Angstroms between two atoms that are not bonded
	// TODO: keep fixed value or let depend on atom sizes?
	public static final double min_not_bonded_distance = 1.0;
	// Conversion from nDyne/A to N/m
	public static final double NDYNEperA_2_NperM = 100;
	// Conversion from amu bohr² to kg m² 
	public static final double IMconvert = 4.6838E-48;
	// number of avogadro
	public static double avogadro = 6.022E23;
	// standard molar volume
	public static final double Vm = 22.4E-3;//m³/mol
	// atm to pascal
	public static final int p0 = 101325;
	// conversion from angstrom to bohr
	public static final double angstrom2bohr = 1.88972;
	// degrees to radians
	public static final double DEG2RAD = 2*Math.PI/360;
	// radians to degrees
	public static final double RAD2DEG = 360/(2*Math.PI);

	
	// temperatures for nasa polynomials
	public final static double TCOMMON = 1000.0;//TODO absolutely arbitrary!
	public final static double THIGH = 1500.0;
	public final static double TLOW = 298.0;
	
	//Reaction changes
	public static final int DECREASE_BOND = 0;
	public static final int INCREASE_BOND = 1;
	public static final int BREAK_BOND = 2;
	public static final int FORM_BOND = 3;
	public static final int GAIN_RADICAL = 4;
	public static final int LOSE_RADICAL = 5;
	public static final int GAIN_CHARGE = 6;
	public static final int LOSE_CHARGE = 7;
	public static final int UNSET = 8;

}
