package fileGeneration.generators;

import constants.StringConstants;

public class TemperatureGenerator {
	
	private double temperature;
	
	private TemperatureGenerator(double temp){
		
		this.temperature	=	temp;
	}
	/**Generate the line for the temperature
	 * 
	 * @param Input
	 * @return
	 */
	private String generate(){
		
		return StringConstants.TEMPERATURE_START+temperature+StringConstants.TEMPERATURE_END;
	}	
	
	/**Generate the line for the temperature
	 * 
	 * @param temperature
	 * @return
	 */
	protected static String generate(double temperature){
		
		TemperatureGenerator TG	=	new TemperatureGenerator(temperature);
		
		return TG.generate();
	}
}