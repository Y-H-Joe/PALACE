package tools;

import java.util.List;

public class Tools {

	/**Searches whether a string array contains a specified string
	 *  
	 * @param Strlist
	 * @param search
	 * @return
	 */
	public static boolean contains(String[] Strlist, String search){
		
		boolean found	=	false;
		
		for(String element:Strlist){
			if(!found){
				found	=	element.equals(search);
			}
		}
		
		return found;
	}
	
	/**Searches whether a string array contains a specified string
	 *  
	 * @param Strlist
	 * @param search
	 * @return
	 */
	public static boolean contains(Iterable<String> Strlist, String search){
		
		boolean found	=	false;
		
		for(String element:Strlist)
			if(!found)
				found	=	element.equals(search);
		
		return found;
	}
	/**Searches whether a specified string contains any of the strings of the array as substring
	 * 
	 * @param Strlist
	 * @param search
	 * @return
	 */
	public static boolean containsAsSub(String[] strList, String search){
		
		boolean found	=	false;
		
		for(String element:strList){
			if(!found){
				found	=	search.contains(element);
			}
		}
		
		return found;
	}
	
	/**Searches whether a specified string contains any of the strings of the array as substring
	 * 
	 * @param Strlist
	 * @param search
	 * @return
	 */
	public static boolean containsAsSub(char[] charList, String search){
		
		boolean found	=	false;
		
		for(char element:charList){
			if(!found){
				found	=	search.contains(""+element);
			}
		}
		
		return found;
	}
	
	/**Checks whether two string arrays contain the same elements, but in a different (or same) order.
	 * 
	 * @param Strlist1
	 * @param Strlist2
	 * @return
	 */
	public static boolean haveSameElements(String[] Strlist1, String[] Strlist2){
		
		boolean forA	=	true;
		boolean forB	=	true;
		boolean length	=	Strlist1.length == Strlist2.length;
		
		for(String element:Strlist1){
			if(!contains(Strlist2,element)){
				return forA	=	false;
			}
		}
		
		for(String element:Strlist2){
			if(!contains(Strlist1,element)){
				return forB	=	false;
			}
		}
		
		return forA && forB && length;
	}
	
	/**Returns the index of an object a in an array. If the object is not present, returns -1;
	 * 
	 * @param a
	 * @param list
	 * @return index of a in list
	 */
	public static int indexOf(Object a,Object[] list){
		
		for(int i = 0;	i< list.length;	i++){
			if(list[i].equals(a))
				return i;
		}
		
		return -1;
	}

	public static int occurences(String search, List<String> list) {
		
		int occurences	=	0;
		
		if(list.isEmpty())
			return 0;
		
		for(String symbol:list){
			if(symbol.equals(search))
				occurences++;
		}
		
		return occurences;
	}
	
	/**Count the number of times a character appears in a string.
	 * 
	 * @param line
	 * @param c
	 * @return #c
	 */
	public static int charCount(String line, char c){
		
		int count	=	0;
		
		for(char cc:line.toCharArray()){
			if(cc==c){
				count++;
			}
		}
		
		return count;
	}
	
	/**Split a string on a given delimiter. Avoids failure for "+", "\" etc.
	 * 
	 * @param line
	 * @param delim
	 * @return
	 */
	public static String[] split(String line, char delim){
		
		String[] out		=	new String[charCount(line,delim) + 1];
		int indexOfDelim	=	line.indexOf(delim);
		int index			=	0;
		//If the delimiter is not present, the entire line should be returned.
		if(indexOfDelim == -1)
			out[0]	=	line;
		//As long as the tail contains the delimiter, keep adding sections.
		while(indexOfDelim != -1){
			String tail		=	line.substring(indexOfDelim+1);
			String head		=	line.substring(0, indexOfDelim);
			out[index]		=	head;
			line			=	tail;
			indexOfDelim	=	line.indexOf(delim);
			index++;
			//If final step: add tail as well.
			if(indexOfDelim == -1)
				out[index]	=	tail;
		}
		
		return out;
	}
	
	/**Check whether a string starts with a number. Return an array of which the first element is the number
	 * (in string, one if no number), and the second the remaining string
	 * @param s
	 * @return
	 */
	public static String[] startsWithNumber(String s){
		
		String number	=	"";
		int index		=	0;
		
		while(s.charAt(0) < 58 && s.charAt(0) > 48){
			number+=s.charAt(index);
			s	=	s.substring(1);
		}
		
		if(number.equals("")){
			String [] ans	=	{"1",s};
			return ans;
		}
		else{
			String [] ans	=	{number,s};
			return ans;
		}
	}
	
	/**Remove any illegal characters for file names
	 * 
	 * @param s
	 * @return
	 */
	public static String removeIllegal(String s){
		s	=	s.replace('<', 'x');
		s	=	s.replace('>', 'x');
		s	=	s.replace(':', 'x');
		s	=	s.replace('\"', 'x');
		s	=	s.replace('/', 'x');
		s	=	s.replace('\\', 'x');
		s	=	s.replace('|', 'x');
		s	=	s.replace('?', 'x');
		s	=	s.replace('*', 'x');
		return s;
	}
	
}