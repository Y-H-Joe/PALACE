package interfaces;

import java.util.List;

import org.openscience.cdk.interfaces.IReaction;

public interface IInput {
	
	/**Returns the bimolecular property of all reactions
	 * 
	 * @return ?bimolecular
	 */
	public List<Boolean> getBimol();
	
	/**Retrieve the read constraints from the input file
	 * 
	 * @return constraints
	 */
	public List<String[]> getConstraints();
	
	/**Get the number of processed reactions
	 * 
	 * @return number of reactions
	 */
	public int getInputCount();
	
	/**Retrieve the read kinetics from the input file
	 * 
	 * @return kinetics
	 */
	public List<String[]> getKinetics();
	
	/**Get the names of the reactions (the names of the rxn files)
	 * 
	 * @return names
	 */
	public List<String> getNames();
	
	/**Get the reactions in the input
	 * 
	 * @return reactions
	 */
	public List<IReaction> getReactions();
	
	/**Get the temperature specified in the input file
	 * 
	 *@return temperature
	 */
	public double getTemperature();
	
	/**Get the number of reactions in the input for which the number of reactants was > 2
	 * 
	 * @return count
	 */
	public int tooManyReactantCount();
	
	/**Get the specified path where all inputs can be found.
	 * 
	 * @return  input path name
	 */
	public String getInputPath();
}