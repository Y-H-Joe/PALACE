package tools;

import org.apache.log4j.Logger;
import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.exception.InvalidSmilesException;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IChemObjectBuilder;
import org.openscience.cdk.smiles.SmilesParser;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;

/**A reaction class specifically for text-based input reactions.
 * 
 * @author pplehier
 *
 */
public class TxtReaction {
	
	private IAtomContainer[] Reactants;
	private IAtomContainer[] Products;
	private String smiles;
	private static Logger logger=Logger.getLogger(TxtReaction.class);
	
	public TxtReaction(String[] React, String[] Prod){
	
		IChemObjectBuilder builder	=	DefaultChemObjectBuilder.getInstance();
		SmilesParser sp				=	new SmilesParser(builder);
		
		int rind	=	React.length;
		int pind	=	Prod.length;
		Reactants	=	new IAtomContainer[rind];
		Products	=	new IAtomContainer[pind];
		setSmiles(React,Prod);
		
		if (rind >= 3){logger.error("ERROR: Too many reactants specified.");System.exit(-1);}
		else{
			for(int i = 0;	i < rind;	i++){
				try{
					this.Reactants[i]	=	sp.parseSmiles(React[i]); 
					AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(this.Reactants[i]);
				}
				catch(InvalidSmilesException e){
					logger.error("Invalid smiles for reactant: "+React[i]);
					e.printStackTrace();
				} catch (CDKException e) {
					logger.warn("Failed to percieve atom types and configure atoms for reactant: "+React[i]);
					e.printStackTrace();
				}
			}
		}
		
		for (int i = 0;	i < pind;	i++){
			try{
				this.Products[i]	=	sp.parseSmiles(Prod[i]);
				AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(this.Products[i]);
			}
			catch(InvalidSmilesException e){
				logger.error("Invalid smiles for species: "+Prod[i]);
				e.printStackTrace();
			} catch (CDKException e) {
				logger.warn("Failed to percieve atom types and configure atoms for reactant: "+Prod[i]);
				e.printStackTrace();
			}
		}
	}
	
	/**Creates a reaction smiles based on the specified reactant and product smiles.<br>
	 * This ensures that the same information as provided by the user is used to generate the reactants and 
	 * products
	 * 
	 * @param reactants
	 * @param products
	 */
	private void setSmiles(String[] reactants, String[] products){
		String smiles	=	"";
		int lengthR		=	reactants.length;
		int lengthP		=	products.length;
		
		for(int i = 0;	i < lengthR;	i++){
			if(i == lengthR-1){
				smiles	+=	reactants[i] + ">>";
			}
			
			else{
				smiles	+=	reactants[i] + ".";
			}
		}
		
		for(int i = 0;	i < lengthP;	i++){
			if(i == lengthP-1){
				smiles	+=	products[i];
			}
			else{
				smiles	+=	products[i] + ".";
			}
		}
		
		this.smiles=smiles;
	}
	
	/**Retrieve the smiles of this reaction.
	 * 
	 * @return reaction smiles
	 */
	public String getSmiles(){
		
		return this.smiles;
	}
	public IAtomContainer[] getReactants(){
		
		return this.Reactants;
	}
	
	public IAtomContainer[] getProducts(){
		
		return this.Products;
	}
	
	/**Checks whether the reaction is in atom balance, based on the atomic numbers of the present atoms.<br>
	 * Implicit hydrogen atoms are also counted
	 * 
	 * @return is reaction balanced
	 */
	public boolean checkAtomBalance(){
		
		int [] elements	=	new int[118];
		
		for(IAtomContainer react:Reactants){
			for(int atcount = 0;	atcount < react.getAtomCount();	atcount++){
				int atomicNumber	=	react.getAtom(atcount).getAtomicNumber();
				
				if(atomicNumber > 118 || atomicNumber == 0){
					logger.fatal("Unknown element found in reaction");System.exit(-1);
				}
				else{
					elements[atomicNumber-1]++;
				}
				
				elements[0]	+=	react.getAtom(atcount).getImplicitHydrogenCount();
			}
		}
		
		for(IAtomContainer prod:Products){
			for(int atcount = 0;	atcount < prod.getAtomCount();	atcount++){
				int atomicNumber	=	prod.getAtom(atcount).getAtomicNumber();
				
				if(atomicNumber > 118 || atomicNumber == 0){
					logger.fatal("Unknown element found in reaction");System.exit(-1);
				}
				else{
					elements[atomicNumber-1]--;
				}
				
				elements[0]	-=	prod.getAtom(atcount).getImplicitHydrogenCount();
			}
		}
		
		for(int i:elements){
			if(i != 0) return false;
		}
		
		return true;
	}
}