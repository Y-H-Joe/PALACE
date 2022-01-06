package interfaces;

/**Interface for class types that read data from a database for one single entry
 * 
 * @author pplehier
 *
 */
public interface IDatabaseReader {
	
	/**Read the database entry
	 * 
	 * @return does the entry exist
	 */
	public boolean read();
	
	/**Return the number of reactants that have been read.
	 * 
	 * @return reactant count
	 */
	public int getReactantCount();
	
	/**Return the number of products that have been read.
	 * 
	 * @return product count
	 */
	public int getProductCount();
	
	/**Get the list of read component ID's
	 * 
	 * @return list of component IDs;
	 */
	public String[] getComponentIDs();
	
	/**Get the list of read component mol (MDL format) files
	 * 
	 * @return file strings
	 */
	public String[] getComponentMolFiles();
	
	/**Get the list of the read coefficients of the components in the reaction.
	 * 
	 * @return component coefficients
	 */
	public int[] getComponentCoefs();
}