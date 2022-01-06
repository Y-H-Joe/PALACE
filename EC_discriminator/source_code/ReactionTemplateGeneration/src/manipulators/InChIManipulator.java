package manipulators;

import java.util.ArrayList;
import java.util.List;

import constants.StringConstants;

public class InChIManipulator {

	private String inChI;

	/**Constructs a new InChIManipulator
	 * 
	 * @param inChI
	 */
	public InChIManipulator(String inChI){
		
		this.inChI	=	inChI;
	}
	
	public String getInChI(){
		return inChI;
	}
	
	/**Removes the stereolayer(s) from the original InChI.
	 *  
	 * @return InChI without stereo
	 */
	public void removeStereo(){
		
		String[] sections	=	inChI.split("/");
		int index			=	0;
		String newInChI		=	StringConstants.EMPTY;
		List<Integer> indices	=	new ArrayList<Integer>();
		
		for(index = 0;	index < sections.length;	index++){
			if(sections[index].startsWith("b")
			   ||
			   sections[index].startsWith("t")
			   ||
			   sections[index].startsWith("m")
			   ||
			   sections[index].startsWith("s"))
				
				indices.add(index);
		}
		
		for(int i = 0;	i < sections.length;	i++){
			if(!indices.contains(i)){
				newInChI	+=	sections[i] + "/";
			}
		}
		
		if(newInChI.endsWith("/"))
			newInChI	=	newInChI.substring(0, newInChI.length() - 1);
		
		this.inChI	=	newInChI;
	}
	
	/**Removes the charge layer from the original InChI. 
	 *  
	 * @return InChI without stereo
	 */
	public void removeCharges(){
		
		String[] sections	=	inChI.split("/");
		int index			=	0;
		String newInChI		=	StringConstants.EMPTY;
		List<Integer> indices	=	new ArrayList<Integer>();
		
		for(index = 0;	index < sections.length;	index++){
			if(sections[index].startsWith("q")
			   ||
			   sections[index].startsWith("p")
			  )
				indices.add(index);
		}
		
		for(int i = 0;	i < sections.length;	i++){
			if(!indices.contains(i)){
				newInChI	+=	sections[i] + "/";
			}
		}
		
		if(newInChI.endsWith("/"))
			newInChI	=	newInChI.substring(0, newInChI.length() - 1);
		
		this.inChI	=	newInChI;
	}
}