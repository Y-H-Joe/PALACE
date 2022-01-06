package tools;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

public class Output {
	
	/**Creates a line containing amount times the specified symbol:<br>
	 * EG: <br>line('+',10) -> <br>
	 * ++++++++++
	 * 
	 * @param symbol
	 * @param amount
	 * @return amount times symbol
	 */
	public static String line(String symbol,int amount){
		
		String out	=	"";
		
		for(int i = 0;	i < amount;	i++){
			out	+=	symbol;
		}
		
		return out;
	}
	
	/**Returns the time (format HH:mm:ss) in string format
	 * 
	 * @return HH:mm:ss
	 */
	public static String getTime(){
		
		DateFormat dateFormat	=	new SimpleDateFormat("HH:mm:ss");
		Date date				=	new Date();
		
		return dateFormat.format(date);
	}
	
	/**Returns the time and date in string format
	 * 
	 * @return yyyy/MM/dd HH:mm:ss
	 */
	public static String getDate(){
		
		DateFormat dateFormat	=	new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
		Date date				=	new Date();
		
		return dateFormat.format(date);
	}
}