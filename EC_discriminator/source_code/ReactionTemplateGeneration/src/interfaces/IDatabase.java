package interfaces;

public interface IDatabase {

	/**Retrieve all entries of the database and write them to a .rxn file.
	 */
	public void retrieve();

	/**Return the default output directory
	 * 
	 * @return name of output directory
	 */
	public String defaultOutput();

	/**Set the output directory to a specified output
	 * 
	 * @param name of the output directory
	 */
	public void setOutputDir(String name);

	/**Retrieve a single entry from the database and write it to a .rxn file
	 * 
	 * @param number of the entry
	 */
	public void retrieve(int oneEntry);

	/**Retrieve all entries between the low and high number and writes them to .rxn files.
	 * 
	 * @param max, min
	 */
	public void retrieve(int low, int high);

	/**Retrieve all entries specified by the content of the array and write them to .rxn files
	 * 
	 * @param array with entry numbers
	 */
	public void retrieve(int[] fromFile);

	/**Get the name of the database
	 * 
	 * @return database name
	 */
	public String getName();
}