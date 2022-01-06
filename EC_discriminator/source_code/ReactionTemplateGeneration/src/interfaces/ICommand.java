package interfaces;

public interface ICommand {
	
	/**Returns whether the command has a specified option.
	 * 
	 * @param option in char format
	 */
	public boolean hasOption(char opt);
	
	/**Returns whether the command has a specified option.
	 * 
	 * @param option in String format
	 */
	public boolean hasOption(String opt);
	
	/**Returns the value of a specified option
	 * 
	 * @param option in char format
	 */
	public String getOption(char opt);

	/**Returns the value of a specified option
	 * 
	 * @param option in String format
	 */
	public String getOption(String opt);
	
	/**Execute the command line
	 */
	public void execute();
}