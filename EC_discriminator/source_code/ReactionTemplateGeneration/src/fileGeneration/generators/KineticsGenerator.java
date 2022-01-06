package fileGeneration.generators;

import org.apache.log4j.Logger;

import io.readers.xmlClasses.InpKinetics;
import io.readers.xmlClasses.KineticsType;
import io.readers.xmlClasses.RateCoeff;
import reactionCenterDetection.ReactionCenter;

public class KineticsGenerator {
	
	private static Logger logger=Logger.getLogger(KineticsGenerator.class);
	private String[] kinetics;
	private boolean reverse;
	private boolean zeroKinetics;
	
	private KineticsGenerator(String[] kinetics, boolean reverse, boolean zeroKinetics){
		
		this.kinetics		=	kinetics;
		this.reverse		=	reverse;
		this.zeroKinetics	=	zeroKinetics;
	}
	
	/**Generate the kinetics block for the reaction family corresponding to reaction[index] stored in the 
	 * ChemistryGenerator
	 * TODO: still needs some work
	 * @param index
	 * @return
	 */
	private InpKinetics generate(){
		
		InpKinetics kin	=	new InpKinetics();
		String kinLable	=	new String(kinetics[1]);
		String kinType	=	new String(kinetics[1]);
		//if revers=true: override the kinetics argument:
		if(reverse){
			kinType		=	"REVERSE";
			kinLable	=	"REVERSE";
		}
		
		if(zeroKinetics){
			kinType		=	"ARRHENIUS";
			kinLable	=	"ARRHENIUS0";
		}
		
		kin.setType(KineticsType.fromValue(kinType));
		
		switch(kinLable){
		
		case "ARRHENIUS0":		{
								RateCoeff rate	=	new RateCoeff();
								rate.setA(0.0);
								rate.setEa(0.0);
								rate.setN(0.0);
								kin.addRateCoeff(rate);
								}
		break;
		
		case "ARRHENIUS": 		{
								RateCoeff rate	=	new RateCoeff();
								rate.setA(Double.parseDouble(kinetics[2]));
								rate.setEa(Double.parseDouble(kinetics[4]));
								rate.setN(Double.parseDouble(kinetics[3]));
								kin.addRateCoeff(rate);
								}
		break;
									
		case "GROUP_ADDITIVITY":{
								kin.setPath(kinetics[2]);
								}
		break;


		case "EVANS_POLANYI":	{
								RateCoeff rate	=	new RateCoeff();
								rate.setA(Double.parseDouble(kinetics[2]));
								rate.setEpAlpha(Double.parseDouble(kinetics[4]));
								rate.setEpConstant(Double.parseDouble(kinetics[3]));
								kin.addRateCoeff(rate);
								}	
		break;
										
		case "BLOWERS_MASEL":	{
								RateCoeff rate	=	new RateCoeff();
								rate.setA(Double.parseDouble(kinetics[2]));
								rate.setVp(Double.parseDouble(kinetics[4]));
								rate.setWbwf(Double.parseDouble(kinetics[3]));
								kin.addRateCoeff(rate);
								}
		break;
							
		case "REVERSE":					break;
		
		case "AB_INITIO":				break;
		
		case "TEMP_INDEPENDENT_RATE":	break;
		
		case "LIBRARY":					break;
		
		default: 						logger.error("Incompatible kinetics type entered.");
										System.exit(-1);
										break;
		}
		
		return kin;
	}
	
	protected static InpKinetics generate(String[] kinetics, ReactionCenter RC, boolean zeroKinetics){
		
		KineticsGenerator kG	=	new KineticsGenerator(kinetics,RC.isReverse(), zeroKinetics);
		
		return kG.generate();
	}
}