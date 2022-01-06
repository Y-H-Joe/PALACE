package reactionMapping;

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.openscience.cdk.Mapping;
import org.openscience.cdk.Reaction;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IAtomContainerSet;
import org.openscience.cdk.interfaces.IChemObject;
import org.openscience.cdk.interfaces.IMapping;
import org.openscience.cdk.interfaces.IReaction;

import constants.StringConstants;

/**This class facilitates working with mapped reactions, giving more direct access to mapped atoms, containers etc.
 * 
 * @author pplehier
 *
 */
public class MappedReaction {
	
	private static Logger logger	=	Logger.getLogger(MappedReaction.class);
	private IReaction reaction;
	private Iterable<IMapping> mappings;
	private boolean allAtomsMapped;
	private int resonanceStructureNumber;
	
	public MappedReaction clone() throws CloneNotSupportedException{
		
		MappedReaction newRxn	=	new MappedReaction();
		IReaction clone			=	(IReaction) this.reaction.clone();
		if(clone != null)
			newRxn.reaction			=	clone;
		if(this.mappings != null)
			newRxn.mappings			=	clone.mappings();
		
		newRxn.allAtomsMapped	=	this.allAtomsMapped;
		
		return newRxn;
	}
	
	public MappedReaction(IReaction reaction, boolean construct){
		this.reaction	=	reaction;
	}
	
	/**Constructor
	 */
	public MappedReaction(){
		this.reaction	=	new Reaction();
	}
	
	/**Construct a new MappedReaction. Checks whether all atoms have been mapped. Uses the manual mapper to
	 * add mappings for missed atoms
	 * 
	 * @param reaction
	 */
	public MappedReaction(IReaction reaction){
		
		this.reaction		=	reaction;
		this.mappings		=	reaction.mappings();
		ManualMapper mapper	=	new ManualMapper(this);
		//Manually map missed atoms
		mapper.manuallyMap();
		//re-assign the new reaction and mappings
		this.reaction		=	mapper.getReaction().reaction;
		this.mappings		=	mapper.getReaction().mappings;
		this.allAtomsMapped	=	this.checkAllAtomsMappedNoH();
		
		if(allAtomsMapped){
			this.assignMappingAsProperty();
			HydrogenMapper hm	=	new HydrogenMapper(this);
			hm.fixHydrogenMapping();
		}
		
		this.allAtomsMapped	=	this.checkAllAtomsMapped();
	}
	
	
	/**Get the mapping indices in matrix format. The first index refers to the mapping number, the second to the  
	 * index of the Reactant atom container and the third to the index of the atom in the atom container.
	 * 
	 * @param rxn
	 * @return indices
	 */
	public int[][][] getMappingIndices(){
	
		int[][][] maps			=	new int[reaction.getMappingCount()][2][2];
		int a					=	0;
		
		for(IMapping M:mappings){	
			IChemObject R			=	M.getChemObject(0);
			IChemObject P			=	M.getChemObject(1);
			int i1			=	0;
			int i2			=	0;
			boolean found1	=	false;
			
			while(!found1 && i1 < reaction.getReactantCount()){
				int j	=	0;
				
				while(!found1 && j < reaction.getReactants()
											 .getAtomContainer(i1)
											 .getAtomCount()){
					found1	=	reaction.getReactants()
										.getAtomContainer(i1)
										.getAtom(j) == R;
					
					if(found1){
						maps[a][0][0]	=	i1;
						maps[a][0][1]	=	j;
					}
					
					j++;
				}
				
			i1++;
			}
			
			boolean found2	=	false;
			
			while(!found2 && i2 < reaction.getProductCount()){
				int j	=	0;
			
				while(!found2 &&j < reaction.getProducts()
											.getAtomContainer(i2)
											.getAtomCount()){
					found2=reaction.getProducts()
								   .getAtomContainer(i2)
								   .getAtom(j) == P;
					
					if(found2){
						maps[a][1][0]	=	i2;
						maps[a][1][1]	=	j;
					}
					
					j++;
				}
				
			i2++;
			}
			
		a++;	
		}
		
		return maps;
	}


	/**Visualises the mapping indices
	 * 
	 * @param mat
	 */
	public void printMapping(){
		
		int[][][] mat	=	this.getMappingIndices();
		
		for(int i = 0;	i < mat.length;	i++){
			System.out.print("[");
			
			for(int j = 0;	j < mat[i].length;	j++){
				System.out.print("[");
				
				for(int k = 0;	k < mat[i][j].length;	k++){
					System.out.print(mat[i][j][k]);
					
					if(k + 1 < mat[i][j].length){System.out.print(" , ");}
				}
				System.out.print("]");
				
				if(j + 1 < mat[i].length){System.out.print(" , ");}
			}
			System.out.print("]\n");
		}
	}


	/**Assign the mapping as a property of the atom
	 * 
	 * @param mappedReaction
	 */
	protected void assignMappingAsProperty(){
		
		if(this.mappings==null){return;}
		
		int[][][] maps	=	this.getMappingIndices();
		
		for(int i = 0;	i < maps.length;	i++){
			reaction.getReactants()
					.getAtomContainer(maps[i][0][0])
					.getAtom(maps[i][0][1])
					.setProperty(StringConstants.MAPPINGPROPERTY, maps[i][1]);
			reaction.getReactants()
					.getAtomContainer(maps[i][0][0])
					.getAtom(maps[i][0][1])
					.setProperty(StringConstants.MAPPINGPROPERTYATOM, 
								 this.getMappedProductAtom(reaction.getReactants()
								  								   .getAtomContainer(maps[i][0][0])
								  								   .getAtom(maps[i][0][1])));
			
			reaction.getProducts()
					.getAtomContainer(maps[i][1][0])
					.getAtom(maps[i][1][1])
					.setProperty(StringConstants.MAPPINGPROPERTY, maps[i][0]);
			reaction.getProducts()
					.getAtomContainer(maps[i][1][0])
					.getAtom(maps[i][1][1])
					.setProperty(StringConstants.MAPPINGPROPERTYATOM,
								 this.getMappedReactantAtom(reaction.getProducts()
								  			   						.getAtomContainer(maps[i][1][0])
								  			   						.getAtom(maps[i][1][1])));
		}
	}

	/**Checks that all reactant atoms are mapped to a corresponding product atom
	 * 
	 * @param Mappedrxn
	 */
	private boolean checkAllAtomsMapped(){
		
		boolean mapped	=	true;
		
		for(int i = 0;	i < reaction.getReactantCount();	i++){
			IAtomContainer reactant	=	reaction.getReactants().getAtomContainer(i);
			
			for(int j = 0;	j < reactant.getAtomCount();	j++){
				try{
					this.checkIfReactantMapped(reactant.getAtom(j));
				}
				catch(Exception e){
					logger.error("Reactant atom "+reactant.getAtom(j).getAtomTypeName()+" has not been mapped.");
					mapped	=	false;
				}
			}
		}
		
		for(int i = 0;	i < reaction.getProductCount();	i++){
			IAtomContainer product	=	reaction.getProducts().getAtomContainer(i);
			
			for(int j = 0;	j < product.getAtomCount();	j++){
				try{
					this.checkIfProductMapped(product.getAtom(j));
				}
				catch(Exception e){
					logger.error("Product atom "+product.getAtom(j).getAtomTypeName()+" has not been mapped.");
					mapped	=	false;
				}
			}
		}
		
		return mapped;
	}
	
	/**Checks that all reactant atoms (excluding hydrogen) are mapped to a corresponding product atom
	 * 
	 * @param Mappedrxn
	 */
	private boolean checkAllAtomsMappedNoH(){
		
		boolean mapped	=	true;
		
		for(int i = 0;	i < reaction.getReactantCount();	i++){
			IAtomContainer reactant	=	reaction.getReactants().getAtomContainer(i);
			
			for(int j = 0;	j < reactant.getAtomCount();	j++){
				try{
					this.checkIfReactantMapped(reactant.getAtom(j));
				}
				catch(Exception e){
					if(reactant.getAtom(j).getSymbol().equals("H")){}
					else{
					logger.error("Reactant atom "+reactant.getAtom(j).getAtomTypeName()+" has not been mapped.");
					mapped	=	false;
					}
				}
			}
		}
		
		for(int i = 0;	i < reaction.getProductCount();	i++){
			IAtomContainer product	=	reaction.getProducts().getAtomContainer(i);
			
			for(int j = 0;	j < product.getAtomCount();	j++){
				try{
					this.checkIfProductMapped(product.getAtom(j));
				}
				catch(Exception e){
					if(product.getAtom(j).getSymbol().equals("H")){}
					else{
					logger.error("Product atom "+product.getAtom(j).getAtomTypeName()+" has not been mapped.");
					mapped	=	false;
					}
				}
			}
		}
		return mapped;
	}


	/**Checks whether each (reactant) atom has a mapping<br>
	 * Note: reactant atoms should always be mapped as key: "ChemObject[0]"
	 * 
	 * @param atom
	 * @param mappedReaction
	 * @throws Exception 
	 */
	protected void checkIfReactantMapped(IAtom atom) throws Exception{
		
		boolean check	=	false;
		for(IMapping M:mappings){
			check	=	M.getChemObject(0) == atom;
			if(check){
				check	=	M.getChemObject(1) != null;
				if(!check){
					throw new Exception();
				}
			}
			if(check){break;}
		}
			
		if(!check){
			throw new Exception();
		}
	}
	
	/**Checks whether each (product) atom has a mapping<br>
	 * Note: product atoms should always be mapped as value: "ChemObject[1]"
	 * 
	 * @param atom
	 * @param mappedReaction
	 * @throws Exception 
	 */
	protected void checkIfProductMapped(IAtom atom) throws Exception{
		boolean check	=	false;
		for(IMapping M:mappings){
			check	=	M.getChemObject(1) == atom;
			if(check){
				
				check	=	M.getChemObject(0) != null;
				if(!check){
					throw new Exception();
				}
			}
			if(check){break;}
		}
			
		if(!check){
			throw new Exception();
		}
	}
	
	
	/**Add a single mapping to this reaction
	 * 
	 * @param map
	 */
	void addMapping(IMapping map){
		
		this.reaction.addMapping(map);
		this.mappings	=	this.reaction.mappings();
	}
	
	Iterable<IMapping> getMappings(){
		
		return this.mappings;
	}
	
	public boolean allAtomsMapped(){
		
		return this.allAtomsMapped;
	}

	/** Retrieve the index of the atom to which the specified atom has been mapped.
	 * 
	 * @param mappedAtom
	 * @return
	 */
	public int getAtomIndexMappedAtom(IAtom mappedAtom){
		
		return getMappedIndices(mappedAtom)[1];
	}


	/** Retrieve the index of the atomcontainer in which the atom that is mapped to
	 * specified atom mappedAtom can be found
	 * 
	 * @param mappedAtom
	 * @param mappedrxn
	 * @return atomcontainer index
	 */
	public int getContainerIndexMappedAtom(IAtom mappedAtom){
		
		return getMappedIndices(mappedAtom)[0];
	}


	/** Retrieve the indices (0: container, 1: atom) of the product atom that is mapped to the reactant atom
	 * 
	 * @param mappedAtom
	 * @return [container, atom]
	 */
	public int[] getMappedIndices(IAtom mappedAtom){
		
		return (int[])mappedAtom.getProperty(StringConstants.MAPPINGPROPERTY);
	}


	/**Retrieve the atom to which the reactant atom has been mapped.
	 * 
	 * @param mappedReactantAtom
	 * @param mappedrxn
	 * @return
	 */
	public IAtom getMappedProductAtom(IAtom mappedReactantAtom){
		
		return reaction.getProducts()
					   .getAtomContainer(getContainerIndexMappedAtom(mappedReactantAtom))
					   .getAtom(getAtomIndexMappedAtom(mappedReactantAtom));
	}


	public IAtom getMappedReactantAtom(IAtom mappedProductAtom){
		
		return reaction.getReactants()
					   .getAtomContainer(getContainerIndexMappedAtom(mappedProductAtom))
					   .getAtom(getAtomIndexMappedAtom(mappedProductAtom));
	}


	/**Retrieve the index of the container to which the Reactant atom belongs.
	 * 
	 * @param atom
	 * @param rxn
	 * @return
	 */
	public int getProductContainerIndex(IAtom productAtom){
		
		for(int i = 0;	i < reaction.getProductCount();	i++){
			for(int j = 0;	j < reaction.getProducts()
								 		.getAtomContainer(i)
								 		.getAtomCount();	j++){
				if(reaction.getProducts()
						   .getAtomContainer(i)
						   .contains(productAtom)){
					return i;
				}
			}
		}
		return -1;
	}


	/**Retrieve the index of the container to which the Reactant atom belongs.
	 * 
	 * @param atom
	 * @param rxn
	 * @return
	 */
	public int getReactantContainerIndex(IAtom reactantAtom){
		
		for(int i = 0;	i < reaction.getReactantCount();	i++){
			for(int j = 0;	j < reaction.getReactants()
								 		.getAtomContainer(i)
								 		.getAtomCount();	j++){
				if(reaction.getReactants()
						   .getAtomContainer(i)
						   .contains(reactantAtom)){
					return i;
				}
			}
		}
		return -1;
	}
	
	/**Retrieve the mapped reaction
	 * 
	 * @return reaction
	 */
	public IReaction getReaction(){
		
		return this.reaction;
	}
	
	/**Retrieve the reactants as atom container set
	 * 
	 * @return reactants
	 */
	public IAtomContainerSet getReactants(){
		
		return reaction.getReactants();
	}
	
	/**Retrieve the products as atom container set
	 * 
	 * @return products
	 */
	public IAtomContainerSet getProducts(){
		
		return reaction.getProducts();
	}
	
	/**Retrieve the number of reactants in the reaction
	 * 
	 * @return reactant count
	 */
	public int getReactantCount(){
		
		return reaction.getReactantCount();
	}
	
	/**Get the number of products in the reaction
	 * 
	 * @return product count
	 */
	public int getProductCount() {
		return reaction.getProductCount();
	}
	
	/**Retrieve the number of atoms in the reactants.
	 * 
	 * @return atom count
	 */
	public int getReactantAtomCount(){
		
		int atomCount	=	0;
		
		for(IAtomContainer reactant:this.getReactants().atomContainers()){
			atomCount	+=	reactant.getAtomCount();
		}
		
		return atomCount;
	}
	
	/**Get the reactant with the specified index
	 * 
	 * @param index
	 * @return reactant
	 */
	public IAtomContainer getReactant(int index){
		
		return this.reaction.getReactants().getAtomContainer(index);
	}
	
	/**Get the product with the specified index
	 * 
	 * @param index
	 * @return product
	 */
	public IAtomContainer getProduct(int index){
		
		return this.reaction.getProducts().getAtomContainer(index);
	}
	
	/**Get a list of all the atoms in the reactants
	 * 
	 * @return reactant atoms
	 */
	public List<IAtom> getReactantAtoms(){
		
		List<IAtom> atoms	=	new ArrayList<IAtom>();
		
		for(int i = 0;	i < reaction.getReactantCount();	i++){
			for(IAtom atom : this.getReactant(i).atoms()){
				atoms.add(atom);
			}
		}
		
		return atoms;
	}
	
	/**Get a list of all the atoms in the products
	 * 
	 * @return product atoms
	 */
	public List<IAtom> getProductAtoms(){
		
		List<IAtom> atoms	=	new ArrayList<IAtom>();
		
		for(int i = 0;	i < reaction.getProductCount();	i++){
			for(IAtom atom : this.getProduct(i).atoms()){
				atoms.add(atom);
			}
		}
		
		return atoms;
	}
	
	/**Set which resonance structure has been selected to determine the reaction center
	 * 
	 * @param number
	 */
	public void setResonanceStructureNumber(int number){
		
		this.resonanceStructureNumber	=	number;
	}
	
	/**Get which resonance structure has been selected to determine the reaction center
	 * 
	 * @return resonance structure number
	 */
	public int getResonanceStructureNumber(){
		
		return this.resonanceStructureNumber;
	}
	
	/**Reassign the mappings in case changes have taken place
	 */
	public void reassignMappings(){
		this.assignMappingAsProperty();
	}
	
	/**Make a new mapping for this reaction, based on the mapping of a duplicate reaction.
	 * This means that each element of the duplicate reaction has the same coordinates in this reaction and both
	 * reactions represent the same chemical reaction.
	 * 
	 * @param duplicate
	 */
	public void makeNewMapping(MappedReaction duplicate){
		
		if(duplicate.mappings == null){return;}
		
		for(int pos = 0; pos < this.getMappingCount(); pos++)
			this.reaction.removeMapping(pos);
		
		for(IMapping map:duplicate.mappings){
			IAtom react			=	(IAtom) map.getChemObject(0);
			IAtom prod			=	(IAtom) map.getChemObject(1);
			int reactContainer	=	duplicate.getReactantContainerIndex(react);
			int prodContainer	=	duplicate.getProductContainerIndex(prod);
			int reactAtom		=	duplicate.getReactant(reactContainer).getAtomNumber(react);
			int prodAtom		=	duplicate.getProduct(prodContainer).getAtomNumber(prod);
			IAtom thisReact		=	this.getReactant(reactContainer).getAtom(reactAtom);
			IAtom thisProd		=	this.getProduct(prodContainer).getAtom(prodAtom);
			IMapping newMap		=	new Mapping(thisReact,thisProd);
			
			this.reaction.addMapping(newMap);			
		}
		
		this.mappings	=	this.reaction.mappings();
		
	}

	/**Check whether the specified molecule is a reactant of the reaction
	 * 
	 * @param mol
	 * @return
	 */
	public boolean isReactant(IAtomContainer mol){
		
		for(IAtomContainer react:reaction.getReactants().atomContainers()){
			if(react.equals(mol))
				return true;
		}
		
		return false;
	}
	
	/**Check whether the specified molecule is a product of the reaction
	 * 
	 * @param mol
	 * @return
	 */
	public boolean isProduct(IAtomContainer mol){
		
		for(IAtomContainer prod:reaction.getProducts().atomContainers()){
			if(prod.equals(mol))
				return true;
		}
		
		return false;
	}
	
	/**Get the index of the reactant specified. Errors if not a reactant.
	 * 
	 * @param mol
	 * @return
	 */
	public int getReactantIndex(IAtomContainer mol){
		
		int counter	=	0;
		
		for(IAtomContainer react:reaction.getReactants().atomContainers()){
			if(react == mol){
				return counter;
			}
			counter++;
		}
		logger.fatal("Molecule is not a reactant of the reaction");
		System.exit(-1);
		return -1;
	}
	
	/**Get the index of the product specified. Errors if not a product.
	 * 
	 * @param mol
	 * @return
	 */
	public int getProductIndex(IAtomContainer mol){
		
		int counter	=	0;
		
		for(IAtomContainer prod:reaction.getProducts().atomContainers()){
			if(prod == mol){
				return counter;
			}
			counter++;
		}
		logger.fatal("Molecule is not a reactant of the reaction");
		System.exit(-1);
		return -1;
	}
	
	/**Identify a molecule in the reaction.
	 * Returns whether the specified molecule is a reactant (output[0]=1) or a product (output[0]=0) and
	 * to which container it belongs (output[1]).
	 * 
	 * @param mol
	 * @return [is reactant?1:0 ; container index]
	 */
	public int[] identifyReactant(IAtomContainer mol){
		
		int[] output	=	new int[2];
		
		if(this.isReactant(mol)){
			output[1]	=	this.getReactantIndex(mol);
			output[0]	=	1;
		}
		else if(this.isProduct(mol)){
			output[1]	=	this.getProductIndex(mol);
			output[0]	=	0;
		}
		else{
			logger.fatal("Specified molecule is not a reactant or product of the reaction");
			System.exit(-1);
		}
		
		return output;
	}

	/**Get the number of mappings connected to this reaction
	 * 
	 * @return number of mappings
	 */
	public int getMappingCount() {
		return reaction.getMappingCount();
	}
	
	/**Add a reactant to the reaction
	 * 
	 * @param reactant
	 */
	public void addReactant(IAtomContainer reactant){
		reaction.addReactant(reactant);
	}
	
	/**Add a product to the reaction
	 * 
	 * @param product
	 */
	public void addProduct(IAtomContainer product){
		reaction.addProduct(product);
	}
}