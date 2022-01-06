package reactionMapping;

import org.apache.log4j.Logger;
import org.openscience.cdk.Mapping;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IMapping;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;

import changes.ChangeDetector;
import constants.StringConstants;
import manipulators.HydrogenAdder;

public class HydrogenMapper {

	private static Logger logger	=	Logger.getLogger(HydrogenMapper.class);
	private MappedReaction mappedReaction;
	
	protected HydrogenMapper(MappedReaction reaction){
		this.mappedReaction	=	reaction;
	}
	
	/**If hydrogen is an explicit reactant or product (either as H2 or H*) the hydrogens should be 
	 * mapped to the correct atom. This method determines a mapping for these hydrogens.
	 *  
	 * @param mappedrxn
	 */
	protected void fixHydrogenMapping(){
	
		int reactionAtomCounta	=	0;
		int reactionAtomCountb	=	0;
		int mappingCount		=	mappedReaction.getMappingCount();
		
		for(int i = 0;	i < mappedReaction.getReactantCount();	i++){
			reactionAtomCounta	+=	mappedReaction.getReactants()
												  .getAtomContainer(i)
												  .getAtomCount();
		}		
		
		for(int i = 0;	i < mappedReaction.getProductCount();	i++){
			reactionAtomCountb	+=	mappedReaction.getProducts()
											.getAtomContainer(i)
											.getAtomCount();
		}	
		//if all atoms (both reactant and product) are mapped: 
		//no fixing required (less rigorous check than with check all atoms mapped)
		if(reactionAtomCounta == mappingCount && reactionAtomCountb == mappingCount){return;}
		
		//Check all reactant atoms, if at this stage hydrogen is already present,
		//and it has not been mapped: it was present as explicit reactant, but becomes implicit in the product
		if(reactionAtomCounta != mappingCount){
			this.mapReactantHydrogen();
		}
		
		//same for the products
		if(reactionAtomCountb != mappingCount){
			this.mapProductHydrogen();
		}
		
		mappedReaction.assignMappingAsProperty();
		//If still difference: incorrect interpretation of molecules by RDT.
		int newMappingCount	=	mappedReaction.getMappingCount();
		//if in reactant: hydrogen present where it shouldn't be.
		//so remove the unmapped atom and add a sinlge electron (radical)
		if(newMappingCount != reactionAtomCounta){
			logger.warn("RDT misinterpreted input reactants, removing hydrogen atom.");
			fixRDTReactant();
		}
		//Same for product
		if(newMappingCount != reactionAtomCountb){
			logger.warn("RDT misinterpreted input products, removing hydrogen atom.");
			fixRDTProduct();	
		}
		
	}

	/**Generates a mapping for explicitly present H2 or H* in the products
	 * 
	 * @param mappedrxn
	 */
	private void mapProductHydrogen(){
		
		for(int i = 0;	i < mappedReaction.getProductCount();	i++){
			for(int j = 0;	j < mappedReaction.getProducts()
											  .getAtomContainer(i)
											  .getAtomCount();	j++){
				IAtom hydrogen	=	mappedReaction.getProducts()
												  .getAtomContainer(i)
												  .getAtom(j);
				
				if(!hydrogen.getSymbol().equals("H")
				   ||
				   hydrogen.getProperty(StringConstants.MAPPINGPROPERTY) != null){}
				
				else{
					for(int k = 0;	k < mappedReaction.getProductCount();	k++){
						if(k != i){
							IAtomContainer product	=	mappedReaction.getProducts().getAtomContainer(k);
							boolean added			=	false;
							
							for(int l = 0;	l < product.getAtomCount();	l++){
								IAtom mappedAtom	=	product.getAtom(l);
								
								if(!mappedAtom.getSymbol().equals("H")){
									int hydChange	=	ChangeDetector.hydrogenHasDecreased(mappedReaction,mappedAtom,false);
									//If the number of hydrogens on this atom decreases and there are implicit hydrogens,
									//make one of them explicit and add map the product hydrogen to this one.
									if(hydChange < 0 && this.reactantHydrogensLeft(mappedAtom)){
										//Map product hydrogen to a (new) hydrogen on a reactant atom of which the hydrogen count has changed
										HydrogenAdder HA		=	new HydrogenAdder(1);
										IAtom reactantAtom		=	mappedReaction.getMappedReactantAtom(mappedAtom);
										IAtomContainer reactant	=	mappedReaction.getReactants().getAtomContainer(mappedReaction.getContainerIndexMappedAtom(mappedAtom));
										
										//returned array will only contain 1 element as hydrogenAdder was constructed with arg 1.
										int bondIndex			=	HA.addHydrogens(reactant, reactantAtom)[0];
										IMapping newHydrogen	=	new Mapping(reactant.getBond(bondIndex).getConnectedAtom(reactantAtom),hydrogen);
										
										mappedReaction.addMapping(newHydrogen);
										added	=	true;
										break;
									}
								}
							}
							
							if(added){
								break;
							}
						}
					}
				}
			}
		}
	}

	/**Generates a mapping for explicitly present H2 or H* in the reactants
	 * 
	 * @param mappedrxn
	 */
	private void mapReactantHydrogen(){
		
		for(int i = 0;	i < mappedReaction.getReactantCount();	i++){
			for(int j = 0;	j < mappedReaction.getReactants()
											  .getAtomContainer(i)
											  .getAtomCount();	j++){
				IAtom hydrogen	=	mappedReaction.getReactants().getAtomContainer(i).getAtom(j);
				boolean mapped;
				
				try{
					mappedReaction.checkIfReactantMapped(hydrogen);
					mapped	=	true;
				}
				catch(Exception e){
					mapped	=	false;
				}
				
				if(!hydrogen.getSymbol().equals("H") || mapped){}
				else{
					for(int k = 0;	k < mappedReaction.getReactantCount();	k++){
						if(k != i){
							IAtomContainer reactant	=	mappedReaction.getReactants().getAtomContainer(k);
							boolean added			=	false;
							
							for(int l = 0;	l < reactant.getAtomCount();	l++){
								IAtom mappedAtom	=	reactant.getAtom(l);
								
								if(!mappedAtom.getSymbol().equals("H")){
									int hydChange	=	ChangeDetector.hydrogenHasDecreased(mappedReaction,mappedAtom,true);
									
									if(hydChange < 0 && this.productHydrogensLeft(mappedAtom)){
										//Map reactant hydrogen to a (new) hydrogen on a product atom of which the hydrogen count has changed
										HydrogenAdder HA		=	new HydrogenAdder(1);
										IAtom productAtom		=	mappedReaction.getMappedProductAtom(mappedAtom);
										IAtomContainer product	=	mappedReaction.getProducts().getAtomContainer(mappedReaction.getContainerIndexMappedAtom(mappedAtom));
										int bondIndex			=	HA.addHydrogens(product, productAtom)[0];
										IMapping newHydrogen	=	new Mapping(hydrogen,product.getBond(bondIndex).getConnectedAtom(productAtom));
										
										mappedReaction.addMapping(newHydrogen);
										added	=	true;
										break;
									}
								}
							}
							
							if(added){
								break;
							}
						}
					}
				}
			}
		}
	}

	/**Check whether there are still implicit hydrogens left for conversion in the mapped product atom
	 * 
	 * @param atom
	 * @return
	 */
	private boolean productHydrogensLeft(IAtom atom){
		
		int reactantImplicit		=	atom.getImplicitHydrogenCount();
		int productImplicit			=	mappedReaction.getMappedProductAtom(atom).getImplicitHydrogenCount();
		int reactantExplicit		=	0;
		
		for(IAtom neighbour:mappedReaction.getReactant(mappedReaction.getReactantContainerIndex(atom)).getConnectedAtomsList(atom)){
			if(neighbour.getSymbol().equals("H")){
				reactantExplicit++;
			}
		}
		
		return productImplicit > reactantImplicit + reactantExplicit;
	}

	/**Check whether there are still implicit hydrogens left for conversion in the mapped reactant atom
	 * 
	 * @param atom
	 * @return
	 */
	private boolean reactantHydrogensLeft(IAtom atom){
		
		int productImplicit		=	atom.getImplicitHydrogenCount();
		int reactantImplicit	=	mappedReaction.getMappedReactantAtom(atom).getImplicitHydrogenCount();
		int productExplicit		=	0;
		
		for(IAtom neighbour:mappedReaction.getProduct(mappedReaction.getProductContainerIndex(atom)).getConnectedAtomsList(atom)){
			if(neighbour.getSymbol().equals("H")){
				productExplicit++;
			}
		}
		
		return reactantImplicit > productImplicit + productExplicit;
	}

	/**Same as for reactant.
	 * 
	 */
	private void fixRDTProduct(){
		for(int i = 0;	i < mappedReaction.getProductCount();	i++){
			IAtomContainer product	=	mappedReaction.getProducts().getAtomContainer(i);
			for(int j = 0;	j < product.getAtomCount();	j++){
				IAtom hydrogen	=	product.getAtom(j);
				boolean mapped;
				
				try{
					mappedReaction.checkIfProductMapped(hydrogen);
					mapped	=	true;
				}
				catch(Exception e){
					mapped	=	false;
				}
				if(!mapped && product.getAtomCount() != 1){
					/*IAtom reactantNeighbour		=	this.getMappedReactantAtom(product.getConnectedAtomsList(hydrogen).get(0));
					IAtomContainer reactant		=	this.getReactant(this.getReactantContainerIndex(reactantNeighbour));
					HydrogenAdder ha			=	new HydrogenAdder(1);
					int bondIndex				=	ha.addHydrogens(reactant, reactantNeighbour)[0];
					IMapping newH				=	new Mapping(reactant.getBond(bondIndex).getConnectedAtom(reactantNeighbour),hydrogen);
					
					this.addMapping(newH);
					*/
					int neighbour	=	product.getAtomNumber(product.getConnectedAtomsList(hydrogen).get(0));
					product.addSingleElectron(neighbour);
					product.removeAtomAndConnectedElectronContainers(hydrogen);
					mappedReaction.reassignMappings();
					try {
						AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(product);
					} 
					catch (CDKException e) {}
				}
			}
		}
	}
	
	/**If the difference in atomcount-mappingcount is in the reactants, there is a hydrogen atom too much here.
	 * Remove it.
	 * The actual origin can be: hydrogen too many in the reactants, or hydrogen too little in the products. To avoid
	 * searching which it is (especially to which of the two reactants it belongs), assume too many in the reactant 
	 * and remove.
	 */
	private void fixRDTReactant(){
		for(int i = 0;	i < mappedReaction.getReactantCount();	i++){
			IAtomContainer reactant	=	mappedReaction.getReactants().getAtomContainer(i);
			for(int j = 0;	j < reactant.getAtomCount();	j++){
				IAtom hydrogen	=	reactant.getAtom(j);
				boolean mapped;
				
				try{
					mappedReaction.checkIfReactantMapped(hydrogen);
					mapped	=	true;
				}
				catch(Exception e){
					mapped	=	false;
				}
				if(!mapped && reactant.getAtomCount() != 1){
					int neighbour	=	reactant.getAtomNumber(reactant.getConnectedAtomsList(hydrogen).get(0));
					reactant.addSingleElectron(neighbour);
					reactant.removeAtomAndConnectedElectronContainers(hydrogen);
					mappedReaction.reassignMappings();
					try {
						AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(reactant);
					} 
					
					catch (CDKException e) {}
				}
			}
		}
	}
}
