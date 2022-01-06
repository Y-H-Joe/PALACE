package interfaces;

public interface IEntry {

	/**Checks whether the specified entry number has data in the database
	 * 
	 * @return exists
	 */
	public boolean exists();
	
	/**Get the interpreted data from the database in .rxn format
	 * 
	 * @return String containing rxn format
	 */
	public String getRXNFile();
	
	/**Get the database ID of the entry
	 * 
	 * @return ID
	 */
	public String getID();
	
	/**Retrieve the value of unbalanced. Should return true if the reaction is unbalanced.
	 * 
	 * @return unbalanced
	 */
	public boolean unbalanced();
	
	/**Set the value of unbalanced. 
	 */
	public void setUnbalanced(boolean unbalanced);

	/**Get the name of the sub-directory to which the entry belongs
	 * 
	 * @return sub-directory name
	 */
	public String getSubDir();
}