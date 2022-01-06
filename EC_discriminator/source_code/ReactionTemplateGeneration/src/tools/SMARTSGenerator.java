package tools;

import org.apache.log4j.Logger;
import org.openscience.cdk.CDKConstants;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.inchi.InChIGeneratorFactory;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.smiles.SmilesGenerator;

import constants.StringConstants;
import manipulators.SMARTSManipulator;

/**This class defines a tool to generate the smarts representation of a molecule, based on the smiles of that
 * molecule. As a valid smiles is also a valid smarts, the smiles provide an ideal starting point for the
 * generation of the smarts. 
 * 
 * @author pplehier
 *
 */
public class SMARTSGenerator {
	
	private static final String[] Elements			=	{"H","B","C","N","O","F","P","S","Cl","Br","I","b","c","n","o","p","s"};
	private static final String[] ElementNumbers	=	{"#1","#5","#6","#7","#8","#9","#15","#16","#17","#35","#53","#5","#6","#7","#8","#15","#16"};
	private static final String[] unwanted			=	{"H1","H2","H3","H4","H5","H6","H7","H"};//H5-H7 are for higher valent Sulphur and phosphor atoms.
	private static final String[] bondtypes			=	{"-","=","#","@","~",":","/","\\","/?","\\?"};
	private static Logger logger					=	Logger.getLogger(SMARTSGenerator.class);
	private static final String CO					=	"InChI=1S/CO/c1-2";
	private IAtomContainer molecule;
	private int[] order;
	public String smarts;
	private boolean unique;
		
	public SMARTSGenerator(IAtomContainer molecule, int[] order){
		
		this.molecule	=	molecule;
		this.order		=	order;
		this.unique		=	false;
	}
	
	public SMARTSGenerator(IAtomContainer molecule, int[] order, boolean unique){
		this.molecule	=	molecule;
		this.order		=	order;
		this.unique		=	unique;
	}
	
	/**This method constructs a smarts string based on the smiles representation of a molecule. Bonds that contain
	 * at least one atom that is labeled as "LINK", are converted to wildcard bonds "~".
	 * @return
	 */
	public String getSMARTSConnection(){
		
		getSMILES();
		processBrackets();
		placeBrackets();
		changeHydrogen();
		String inChI	=	"";
		try{
			inChI	=	InChIGeneratorFactory.getInstance().getInChIGenerator(molecule).getInchi();
		}
		catch(CDKException e){}
		//Don't include valence or neighbour info for CO
		if(inChI == null || !inChI.equals(CO)){
			updateNeighboursValence();
		}
		insertBonds(true);
		setAromaticLowerCase();
		fixHydrogenRadical();
		
		return this.smarts;
	}
	
	/**This method constructs a smarts string based on the smiles representation of a molecule. 
	 *
	 */
	public String getSMARTS(){
		
		getSMILES();
		processBrackets();
		placeBrackets();
		changeHydrogen();
		String inChI	=	"";
		try{
			inChI	=	InChIGeneratorFactory.getInstance().getInChIGenerator(molecule).getInchi();
		}
		catch(CDKException e){}
		//Don't include valence or neighbour info for CO
		if(inChI == null || !inChI.equals(CO)){
			updateNeighboursValence();
		}
		insertBonds(false);
		setAromaticLowerCase();
		fixHydrogenRadical();
		
		return this.smarts;
	}
	
	/**This method constructs the smiles of a molecule
	 * 
	 */
	public void getSMILES(){
		
		SmilesGenerator sg	=	null;
		if(unique){
			sg	=	SmilesGenerator.unique().aromatic();
		}
		else{
			sg	=	SmilesGenerator.generic().aromatic();
		}
			
		try {
			this.smarts	=	sg.create(molecule,order);
		} catch (CDKException e) {
			logger.error("ERROR: Failed to generate SMILES.");
			e.printStackTrace();
			this.smarts	=	"";
		}
	}
	
	/**Remove the hydrogens that have been added INSIDE brackets of another atom. These hydrogens are recorded as 
	 * implicit hydrogens anyways, and hence are accounted for in {@link #updateXV(String, int, int, AtomContainer)}
	 */
	public void processBrackets(){
		
		String newSmiles=	this.smarts;
		int pos			=	0;

		while(pos < newSmiles.length()){
			if(newSmiles.charAt(pos) == '['){
				int pos1	=	pos;
				
				while(newSmiles.charAt(pos) != ']'){pos++;}
				
				int pos2	=	pos;
				String temp	=	newSmiles.substring(pos1, pos2+1);
				
				for(int i = 0;	i < unwanted.length;	i++){
					if(temp.charAt(1) != 'H'){
						temp	=	temp.replaceAll(unwanted[i], "");
					}
				}
				
				String part1	=	pos1 == 0?"":newSmiles.substring(0, pos1);
				String part2	=	newSmiles.substring(pos2+1);
				newSmiles		=	part1 + temp + part2;
				pos				=	pos1 + temp.length() - 1;
			}
			
			pos++;
		}
		
		this.smarts	=	newSmiles;
	}
	
	/**Places the required brackets around all atoms. Removes possible duplicates at the end of the process
	 * 
	 */
	public void placeBrackets(){
		
		String updatedSmiles=	smarts;
		int pos				=	0;
		
		while(pos < updatedSmiles.length()){
			//skip atoms that are already in brackets
			if(updatedSmiles.charAt(pos) == '['){
				while(updatedSmiles.charAt(pos) != ']'){pos++;}
			}
			
			String element2	=	"";
			
			if(pos < updatedSmiles.length()-1){
				element2	=	updatedSmiles.substring(pos, pos+2);
			}
			
			String element1	=	updatedSmiles.substring(pos,pos+1);
			
			if(Tools.contains(Elements,element2)){
				updatedSmiles	=	updatedSmiles.substring(0,pos)+
									"["+element2+"]"+
									updatedSmiles.substring(pos+2);
				pos	+=	3;
				
			}else if(Tools.contains(Elements,element1)){
				updatedSmiles	=	updatedSmiles.substring(0,pos)+
									"["+element1+"]"+
									updatedSmiles.substring(pos+1);
				pos	+=	2;
				
			}else{pos++;}
		}
		
		updatedSmiles	=	updatedSmiles.replaceAll("\\[\\[\\[","\\[");
		updatedSmiles	=	updatedSmiles.replaceAll("\\]\\]\\]", "\\]");
		updatedSmiles	=	updatedSmiles.replaceAll("\\[\\[","\\[");
		updatedSmiles	=	updatedSmiles.replaceAll("\\]\\]", "\\]");
	
		this.smarts		=	updatedSmiles;
	}
	
	/**Genesys only excepts [#1] as representation of hydrogen. All [H] representations are therefore changed
	 * to [#1]
	 */
	private void changeHydrogen(){
		
		String updatedSmiles	=	this.smarts;
		
		if(this.smarts.contains(Elements[0])){
			updatedSmiles	=	updatedSmiles.replaceAll(Elements[0], ElementNumbers[0]);
			
		}else if(this.smarts.contains(ElementNumbers[0])){
			updatedSmiles	=	updatedSmiles.replaceAll(ElementNumbers[0],Elements[0]);
		}
		
		this.smarts	=	updatedSmiles;
	}
	
	/**Calculate the index in the smiles/smarts string of the neighbours of a given atom 
	 * (only explicit neighbours are taken into account)
	 */
	private int[] getNeighbourIndex(int atomNumber){
		
		int index			=	getAtomIndex(atomNumber);
		int continueat		=	index;
		int j				=	1;
		int [] neighbours	=	new int[countNeighbours(atomNumber)];
		int neighbour		=	0;
		/*For atoms that are not first in the string, first look for the neighbour preceding the atom. 
		 * Is found when no longer between parentheses and the character [ is encountered
		 */
		if(atomNumber !=0 
		   &&
		   this.smarts.charAt(continueat-2) != '>'
		   &&
		   this.smarts.charAt(continueat-2) != '.'){
			int min			=	2;
			int parenthesism=	0;
			boolean stopm	=	false;
			
			while(!stopm){
				if(this.smarts.charAt(continueat-min) == ')'){parenthesism++;}
				if(this.smarts.charAt(continueat-min) == '('&&parenthesism > 0){parenthesism--;}
				if(this.smarts.charAt(continueat-min) == '['&&parenthesism == 0){
					neighbours[neighbour]	=	continueat - min + 1;
					neighbour++;
					stopm	=	true;
				}
				
				min++;
			}			
		}
		/*Ring connections are also considered as neighbours. The index will refer to the linking number (1-9)
		 * (which will be preceded by the bond type) rather than the actual atom. 
		 */
		while(index+j < this.smarts.length()){
			if(this.smarts.charAt(index+j) == ']') break; 
			j++;
		}
		
		continueat+=j;
		
		if(index+j < this.smarts.length()){
			int i				=	1;
			continueat			+=	i;
			boolean nottoolong	=	index + i + j < this.smarts.length();
			
			if(nottoolong){
				boolean betweenOneAndNine	=	(int)this.smarts.charAt(index+i+j) > 48
												&&
												(int)this.smarts.charAt(index+i+j) < 58;
				boolean stop	=	false;
				
				while(this.smarts.charAt(continueat) != '['
					  &&
					  this.smarts.charAt(continueat) != '('
					  &&
					  !stop
					  &&
					  this.smarts.charAt(continueat) != ')'){
					
					if(betweenOneAndNine){
						neighbours[neighbour]	=	getOtherRingAtom(continueat);;
						neighbour++;
					}
					
					i++;
					
					if(index+i+j < this.smarts.length()){continueat	=	index + i + j;}
					
					else{stop	=	true;}
					
					if(index+i+j < this.smarts.length()){
						betweenOneAndNine=(int)this.smarts.charAt(index+i+j) > 48
										  &&
										  (int)this.smarts.charAt(index+i+j) < 58;
					}
				}
			}
		}
		
		/*Finally we search for the remaining neighbours. 
		 */
		boolean bracketfound=	false;
		boolean stop		=	false;
		
		while(continueat < getAtomIndex(atomNumber+1)
			  &&
			  !bracketfound
			  &&
			  !stop
			  &&
			  continueat < this.smarts.length()){
			//If the character following the molecule is ), there are no following neighbours.
			if(this.smarts.charAt(continueat) == ')'){stop	=	true;}
			//If the character is [, we have encountered a first following neighbour.
			else if(this.smarts.charAt(continueat) == '['){
				bracketfound			=	true;
				continueat++;
				neighbours[neighbour]	=	continueat;
				neighbour++;
			}
			//If the character is (, we have entered a parenthesis, which means that only the following [ is a
			//neighbour.
			else if(this.smarts.charAt(continueat) == '('){
					int parenthesis	=	1;
					continueat++;
					
					if(this.smarts.charAt(continueat) == '['){
						neighbours[neighbour]	=	continueat+1;
						neighbour++;
						
					}else{
						neighbours[neighbour]	=	continueat+2;
						neighbour++;
					}
					
					while(parenthesis > 0){
						if(this.smarts.charAt(continueat) == '('){parenthesis++;}
						if(this.smarts.charAt(continueat) == ')'){parenthesis--;}
						if(this.smarts.charAt(continueat) == '['){atomNumber++;}
						continueat++;
					}
				}
			
			else{continueat++;}
		}
		
		return neighbours;
	}

	private int getOtherRingAtom(int posRingNumber){
		
		char ringNumber	=	smarts.charAt(posRingNumber);
		int pos			=	posRingNumber;
		int encounters	=	0;
		
		while(pos > 0){
			pos--;
			
			if(smarts.charAt(pos) == '[' && pos != 0){
				while(smarts.charAt(pos) != ']'){
					if(smarts.charAt(pos) == ringNumber)
						encounters++;
					pos--;
				}
			}
		}
		
		//If even number of encounters: this is the first new encounter, so the corresponding ring will be to the right
		if(encounters % 2 == 0){
			return getAtomIndexFromRingNumber(findForwardsRing(posRingNumber));
		}
		//If odd number of encounters: this is the 'closure' of the ring, so the corresponding ring will be to the left
		else{
			return getAtomIndexFromRingNumber(findBackwardsRing(posRingNumber));
		}
	}
	
	private int findForwardsRing(int fromPos){
		
		int pos		=	fromPos;
		char ring	=	smarts.charAt(pos);
		
		while(pos < smarts.length()-1){
			pos++;
			
			if(smarts.charAt(pos) == ']' && pos != smarts.length()){
				while(smarts.charAt(pos) != '['){
					if(smarts.charAt(pos) == ring)
						return pos;
					pos++;
				}
			}
		}
		logger.error("Failed to find corresponding ring atom! Exiting ...");
		System.exit(-1);
		return -1;
	}
	
	private int findBackwardsRing(int fromPos){
		
		int pos		=	fromPos;
		char ring	=	smarts.charAt(pos);
		
		while(pos > 0){
			pos--;
			
			if(smarts.charAt(pos) == '[' && pos != 0){
				while(smarts.charAt(pos) != ']'){
					if(smarts.charAt(pos) == ring)
						return pos;
					pos--;
				}
			}
		}
		logger.error("Failed to find corresponding ring atom! Exiting ...");
		System.exit(-1);
		return -1;
	}
	
	private int getAtomIndexFromRingNumber(int ringNumberPos){
		
		int pos	=	ringNumberPos;
		
		while(pos > 0){
			pos--;
			
			if(smarts.charAt(pos) == '[')
				return pos + 1;
		}
		
		logger.error("Failed to find corresponding ring atom! Exiting ...");
		System.exit(-1);
		return -1;
	}

	/**Counts the number of neighbours of a given atom, following a similar algorithm as for getNeighbourIndex
	 * 
	 * @param Smiles
	 * @param atomNumber
	 * @return number of neighbours
	 */
	public int countNeighbours(int atomNumber){
		//System.out.println(atomNumber+" "+smarts);
		int count		=	0;//getNeighbourIndex(Smiles, atomNumber).length;
		int index		=	getAtomIndex(atomNumber);
		int continueat	=	index;
		//all atoms, except for the first have at least one neighbour preceding it in the string.
		if(atomNumber != 0
		   &&
		   this.smarts.charAt(continueat-2) != '>'
		   &&
		   this.smarts.charAt(continueat-2) != '.'){count++;}
			
		int j	=	1;
		//rings
		while(index+j < this.smarts.length()){
			if(this.smarts.charAt(index+j) == ']') break;
			j++;
		}
		
		continueat	=	index + j;
		int i		=	0;
		
		while(continueat + i < this.smarts.length() && this.smarts.charAt(continueat + i) != '['){
			boolean betweenOneAndNine	=	(int) this.smarts.charAt(continueat + i) > 48
											&&
											(int) this.smarts.charAt(continueat + i) < 58;
		
			if(betweenOneAndNine){
				count++;
			}
			
			i++;
		}
		/*
		if(index+j < this.smarts.length()){
			int i				=	1;
			boolean nottoolong	=	index + i + j < this.smarts.length();
			
			if(nottoolong){
				boolean betweenOneAndNine=(int)this.smarts.charAt(index+i+j) > 48
										  &&
										  (int)this.smarts.charAt(index+i+j) < 58;
				
				while(betweenOneAndNine&&nottoolong){
					count++;
					i++;
					continueat	=	index + i + j;
					nottoolong	=	index + i + j < this.smarts.length();
					
					if(nottoolong){
						betweenOneAndNine=(int)this.smarts.charAt(index+i+j) > 48
										  &&
										  (int)this.smarts.charAt(index+i+j) < 58;
					}
				}
			}
		}*/
		
		boolean bracketfound=	false;
		boolean stop		=	false;
		//branches
		while(continueat < getAtomIndex(atomNumber+1)
			  &&
			  !bracketfound
			  &&
			  !stop
			  &&
			  continueat < this.smarts.length()){
			if(this.smarts.charAt(continueat) == ')'){stop	=	true;}
			
			else if(this.smarts.charAt(continueat) == '('){
				int parenthesis	=	1;
				continueat++;
				
				while(parenthesis > 0){
					if(this.smarts.charAt(continueat) == '('){parenthesis++;}
					if(this.smarts.charAt(continueat) == ')'){parenthesis--;}
					if(this.smarts.charAt(continueat) == '['){atomNumber++;}
					continueat++;
				}
				
				count++;
			}
			
			else if(this.smarts.charAt(continueat) == '['){
				bracketfound	=	true;
				continueat++;
				count++;
			}
			
			else{continueat++;}
		}
		
		return count;
	}	
	
	/**Determines the number of atoms represented in a given Smiles string.
	 */
	private int countAtoms(){
		
		return molecule.getAtomCount();
	}
	
	/**Updates the valence and number of neighbours of a single atom in the smiles string
	 * 
	 * @param atomNumber in the smiles
	 * @param atomNumber in the molecule
	 */
	private void updateXV(int atomNumberSmiles,int atomNumberMolecule){
		
		IAtom atomInMol	=	this.molecule.getAtom(atomNumberMolecule);
		String out		=	this.smarts;
		boolean link	=	false;
		
		if(atomInMol.getProperty(StringConstants.LINK) != null){
			link	=	(boolean)atomInMol.getProperty(StringConstants.LINK);
		}
		if(atomInMol.getSymbol().equals("H")
			||
		   link){
			/*don't narrow down valence and neighbours*/
			}
		else{
			//int neighbours	=	atomInMol.getFormalNeighbourCount();
			int valence		=	atomInMol.getValency();
			String atom		=	getAtom(atomNumberSmiles);
			int atomIndex	=	getAtomIndex(atomNumberSmiles);
			
			if(!atom.equals(Elements[0]) && !atom.equals(ElementNumbers[0])){
				int atlength	=	0;
			
				while(this.smarts.charAt(atomIndex+atlength) != ']'){atlength++;}
			
				String tempFront	=	this.smarts.substring(0, atomIndex+atlength);
				String tempEnd		=	this.smarts.substring(atomIndex+atlength);
				//Include both neighbour and valence count.
				//tempFront			+=	";X" + neighbours + "v" + valence;
				//Include only valence.
				tempFront			+=	";v"+valence;
				out					=	tempFront+tempEnd;
			}
		}
		
		this.smarts	=	out;
	}
	
	/**Updates the valence and number of neighbours of all atoms in the smiles string.
	 * order[i] is the index in the smiles/ i is the index in the atomcontainer 
	 * 
	 */
	private void updateNeighboursValence(){
		
		for(int i = 0;	i < countAtoms();	i++){
			updateXV(order[i],i);
		}
	}
	
	/**This method adds implicit single bonds to the structure. 
	 * @param link: wild card for links yes or no
	 * TODO: Look into importance of classifying implicit ring bonds as standard single bonds
	 */
	private void insertBonds(boolean link){
		
		int pos			=	0;
		int pos1		=	0;
		int pos2		=	0;
		
		while(pos < this.smarts.length()){
			if(this.smarts.substring(pos).contains("[")){
				while(this.smarts.charAt(pos) != ']' && pos < this.smarts.length()){pos++;}
			}
			
			if(pos != this.smarts.length()-1){
				pos1	=	pos;
				
				if(this.smarts.substring(pos).contains("[")){
					while(this.smarts.charAt(pos) != '[' && pos < this.smarts.length()-2){pos++;} //max pos should be length -1
					pos2	=	pos;
					
					if(!Tools.containsAsSub(bondtypes,this.smarts.substring(pos1, pos2))){
						int atom2		=	getAtomIndexAtPos(pos2);
						String front	=	this.smarts.substring(0,pos2);
						String end		=	this.smarts.substring(pos2);
						if(link)
							this.smarts		=	front + getCorrectBondLink(atom2) + end;
						else
							this.smarts		=	front + getCorrectBond(atom2) + end;
					}
				}
				else{
					break;
				}
			}
			
			pos++;
		}
	}
	
	/**There are two types of bond that will not be added in the smiles of a molecule. Single bonds and aromatic
	 * bonds (for the latter, the smiles representation only requires lowercase atom symbols).
	 * This method determines which of the two should be added.
	 * 
	 * @param smilesAtom1
	 * @return ":" for aromatic, "-" for single.
	 */
	private String getCorrectBond(int smilesAtom1){
		
		int moleculeAtom1	=	-1;
		int moleculeAtom2	=	-1;
		int smilesAtom2		=	getAtomIndexAtPos(getNeighbourIndex(smilesAtom1)[0]);
		
		for(int i = 0;	i < order.length;	i++){
			if(order[i] == smilesAtom1) moleculeAtom1	=	i;
			if(order[i] == smilesAtom2) moleculeAtom2	=	i;
		}
		
		if(moleculeAtom1 == -1 || moleculeAtom2 == -1){
			logger.fatal("Specified atoms do not exist in the atomcontainer. Exiting ...");
			System.exit(-1);
		}
		
		if(molecule.getBond(molecule.getAtom(moleculeAtom1), molecule.getAtom(moleculeAtom2)).getFlag(CDKConstants.ISAROMATIC)){
			return ":";
		}
		else{
			return "-";
		}
	}
	
	/**Also consider atoms that are labeled "Link": these should get the wildcard bond symbol.
	 * 
	 * @param smilesAtom1
	 * @return ":" for aromatic, "-" for single.
	 */
	private String getCorrectBondLink(int smilesAtom1){
		
		int moleculeAtom1	=	-1;
		int moleculeAtom2	=	-1;
		int[] neighbours	=	getNeighbourIndex(smilesAtom1);
		if(neighbours.length == 0){
			return StringConstants.EMPTY;
		}
		else{
			int smilesAtom2		=	getAtomIndexAtPos(getNeighbourIndex(smilesAtom1)[0]);
			
			for(int i = 0;	i < order.length;	i++){
				if(order[i] == smilesAtom1) moleculeAtom1	=	i;
				if(order[i] == smilesAtom2) moleculeAtom2	=	i;
			}

			if(moleculeAtom1 == -1 || moleculeAtom2 == -1){
				logger.fatal("Specified atoms do not exist in the atomcontainer. Exiting ...");
				System.exit(-1);
			}
			
			boolean link1	=	molecule.getAtom(moleculeAtom1).getProperty(StringConstants.LINK) != null?
								molecule.getAtom(moleculeAtom1).getProperty(StringConstants.LINK):false;
			boolean link2	=	molecule.getAtom(moleculeAtom2).getProperty(StringConstants.LINK) != null?
								molecule.getAtom(moleculeAtom2).getProperty(StringConstants.LINK):false;	
			
			if(link1 || link2)
				return "~";
		
			if(molecule.getBond(molecule.getAtom(moleculeAtom1), molecule.getAtom(moleculeAtom2)).getFlag(CDKConstants.ISAROMATIC)){
				return ":";
			}
			else{
				return "-";
			}
		}
	}
	
	/**Get the atom with index atomNumber in the smiles string. Uses the brackets as recognition points.<br>
	 * Brackets are NOT added if missing!
	 * 
	 * @param SmilesWBrackets
	 * @param atomNumber
	 * @return atom symbol
	 */
	private String getAtom(int atomNumber){
		
		String temp	=	this.smarts;
		int count	=	0;
		String out	=	elementTypeOfFirstAtom(temp);
		
		while (count < atomNumber){
			int end	=	temp.indexOf("]");
			temp	=	temp.substring(end+1);
			out		=	elementTypeOfFirstAtom(temp);
			count++;
		}
		
		return out;
	}
	
	/**Get the position in the string of the atom with index atomNumber<br>
	 * EG: getAtomIndex([C][C][C][C],2) will return  7
	 * @param atomNumber
	 * @return string index of the atom
	 */
	private int getAtomIndex(int atomNumber){
		
		String temp	=	this.smarts;
		int count	=	0;
		int out		=	indexOfFirstAtom(temp);
		
		while (count < atomNumber){
			int end	=	temp.indexOf("]");
			out		+=	end - indexOfFirstAtom(temp);
			temp	=	temp.substring(end+1);
			out		+=	indexOfFirstAtom(temp) + 1;
			count++;
		}
		
		return out;
	}
	
	/**Gives the first atom, based on a search for the first bracket.<br>
	 * 
	 * @return first atom symbol
	 */
	private static String elementTypeOfFirstAtom(String smarts){
		
		int index	=	smarts.indexOf("[");
		String out	=	"";
		
		for(int i = 0;	i < Elements.length;	i++){
			if(smarts.substring(index+1,index+2).equals(Elements[i])){
				out	=	Elements[i];
				
			}else if(smarts.substring(index+1, index+3).equals(Elements[i])
					 ||
					 smarts.substring(index+1, index+3).equals(ElementNumbers[i])){
				out	=	Elements[i];
			}
		}
		
		return out;
	}
	
	/**Return the index of the first atom in the string. <br>
	 * EG: indexOfFirstAtom([C][C][C]) will return 1
	 * 
	 * @return index of first atom
	 */
	private static int indexOfFirstAtom(String smarts){
		
		return smarts.indexOf("[")+1;
	}
	
	/**Fix the way a hydrogen radical is represented. The previous methods are made to simply skip H or #1<br>
	 * The only format for the hydrogen radical that genesys accepts is [#1;!X1].<br>
	 * The only reactions in which this could be a problem is when the hydrogen radical is a reactant, for which
	 * the generated smarts will always be either [#1] or [H].
	 * 
	 */
	private void fixHydrogenRadical(){
		
		if(this.smarts.equals("[#1]")||this.smarts.equals("[H]")){
			this.smarts	=	"[#1;!X1]";
		}
	}
	
	/**Returns the atom number of the atom in the brackets to which the position refers. This implies that the pos index
	 * must refer to a character inside brackets or to the brackets themselves.
	 * 
	 * @param posInSmarts
	 * @return atom index
	 */
	private int getAtomIndexAtPos(int posInSmarts){
		
		int pos			=	0;
		//Start from -1 so first encountered atom has index 0
		int atomCount	=	-1;
		
		while(pos < posInSmarts){
			if(pos < this.smarts.length() - 2){
				if(Tools.contains(Elements, this.smarts.substring(pos, pos+2))){
					atomCount++;
				}else if(Tools.contains(Elements, this.smarts.substring(pos, pos+1))){
					atomCount++;
				}else if(this.smarts.substring(pos, pos+2).contains("#1")){
					atomCount++;
				}
			} else if(Tools.contains(Elements, this.smarts.substring(pos, pos+1))){
				atomCount++;
			}
				
			pos++;
		}

		if(this.smarts.charAt(posInSmarts) == '['){
			atomCount++;
		}
		
		if(posInSmarts > 0){
			if(this.smarts.charAt(posInSmarts - 1) == '['){
				atomCount++;
			}
		}
		
		return atomCount;
	}
	
	/**Sets the aromatic atoms to lowercase letters if necessary
	 */
	private void setAromaticLowerCase(){
		
		for(int i = 0;	i < this.countAtoms();	i++){
			//Iterable<IBond> bonds	=	molecule.getConnectedBondsList(molecule.getAtom(i));
			boolean aromaticAtom	=	molecule.getAtom(i).getFlag(CDKConstants.ISAROMATIC)
										||
										molecule.getAtom(i).getProperty(SMARTSManipulator.Aromatic) == null?
													false:
													molecule.getAtom(i).getProperty(SMARTSManipulator.Aromatic);
			
			//Atom can be aromatic without all connected bonds being aromatic
			/*
			for(IBond bond:bonds){
				if(bond.getFlag(CDKConstants.ISAROMATIC)){
					aromaticAtom	=	true;
					break;
				}
			}*/
			
			if(aromaticAtom){
				int atomPos		=	getAtomIndex(order[i]);
				String front	=	this.smarts.substring(0, atomPos);
				String end		=	this.smarts.substring(atomPos+1);
				String newAtom	=	("" + this.smarts.charAt(atomPos)).toLowerCase();
				this.smarts			=	front + newAtom + end;
			}
		}
	}	
}