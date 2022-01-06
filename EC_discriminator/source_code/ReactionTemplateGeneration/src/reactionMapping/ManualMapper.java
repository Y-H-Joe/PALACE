package reactionMapping;

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.openscience.cdk.Mapping;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IMapping;

import constants.StringConstants;
import reactionCenterDetection.ReactionCenter;
import reactionCenterDetection.ReactionCenterDetector;
import structure.StructureAnalyser;
import tools.Tools;

/**This class manually adds a mapping in some relatively simple cases. For some small molecules, RDT has too little
 * structural recognition points to derive an accurate AAM. Luckily, in these small molecules the mapping is actually
 * quite simple.
 * <br/>
 * A few distinctions are made. The first is when all missed atoms are of a different type. In this case it is safe
 * to say that each atom can be mapped to the corresponding type in the products.
 * <br/>
 * A second distinction is when the mapping left two atoms unmapped. If these two atoms belong to identical reactants
 * it doesn't matter which atom is mapped to which product, as no distinction can be made anyways. The same is valid
 * when the product atoms belong to identical products.
 * <br/>
 * Finally, when again only two atoms are missed, but the missed atoms in the product are part of a symmetrical molecule,
 * there will be no difference in which atom is assigned to which product atom. This can again be turned around for 
 * the reactants
 * @author pplehier
 *
 */
public class ManualMapper {

	private static Logger logger	=	Logger.getLogger(ManualMapper.class);
	
	private MappedReaction mappedReaction;
	private List<IAtom> mappedReactant;
	private List<IAtom> mappedProduct;
	private List<IAtom> unmappedReactant;
	private List<String> unmappedReactantSymbol;
	private List<String> unmappedProductSymbol;
	private List<IAtom> unmappedProduct;
	private List<IAtom> uniqueReactants;
	private List<IAtom> uniqueProducts;
	private List<IAtom> nonUniqueReactants;
	private List<IAtom> nonUniqueProducts;
	
	/**Create a new mapper
	 * 
	 * @param mappedReaction
	 */
	public ManualMapper(MappedReaction mappedReaction){
		
		this.mappedReaction			=	mappedReaction;
		this.mappedProduct			=	new ArrayList<IAtom>();
		this.mappedReactant			=	new ArrayList<IAtom>();
		this.unmappedProduct		=	new ArrayList<IAtom>();
		this.unmappedReactant		=	new ArrayList<IAtom>();
		this.uniqueReactants		=	new ArrayList<IAtom>();
		this.uniqueProducts			=	new ArrayList<IAtom>();
		this.nonUniqueReactants		=	new ArrayList<IAtom>();
		this.nonUniqueProducts		=	new ArrayList<IAtom>();
		this.unmappedReactantSymbol	=	new ArrayList<String>();
		this.unmappedProductSymbol	=	new ArrayList<String>();
	}
	
	/**Manually add mappings for the described cases.
	 */
	public void manuallyMap(){
		
		this.findUnmapped();
		this.findUnique();
		this.findNonUnique();
		this.mapDifferentAtoms();
		this.mapPairedAtoms();
	}
	
	/**Get the mapped reaction
	 * 
	 * @return mapped reaction
	 */
	public MappedReaction getReaction(){
		
		return this.mappedReaction;
	}
	
	/**Create a mapping for the atoms in the nonUnique list with two occurrences.
	 */
	private void mapPairedAtoms(){
		
		//if there are only two atoms in the non unique reactants and products lists, easy case: they can be 
		//coupled as such (can't be different types because they'd be unique.
		if(nonUniqueReactants.size() == 2 && nonUniqueProducts.size() == 2){
			try {
				mapTwoSameAtoms(nonUniqueReactants, nonUniqueProducts);
			} catch (CDKException e) {}
		}
		//if there's more than two: find pairs of same types. and map them
		else{
			List<List<List<IAtom>>> pairedAtoms	=	createPairs();
			List<List<IAtom>> pairsReact		=	pairedAtoms.get(0);
			List<List<IAtom>> pairsProd			=	pairedAtoms.get(1);
			
			for(int i = 0;	i < pairsReact.size();	i++){
				try {
					mapTwoSameAtoms(pairsReact.get(i),pairsProd.get(i));
				} catch (CDKException e) {}
			}
		}
	}
	
	/**Create a mapping for the two pairs of atoms specified. Both lists should contain only two atoms. The mapping
	 * will only be performed if the two atoms can be interchanged (i.e either the reactants are identical or the 
	 * products are identical or the reactant is symmetrical or the product is symmetrical)
	 * @throws CDKException 
	 */
	private void mapTwoSameAtoms(List<IAtom> reacts, List<IAtom> prods) throws CDKException{

		StructureAnalyser sa	=	new StructureAnalyser(mappedReaction);
		
		if(reacts.size() == 2 && prods.size() == 2){
			/*Commented:previous version in which method reactantSymmetrical..() only worked for atoms in the same molecule
			if(sa.reactantsIdentical(reacts.get(0),reacts.get(1))
			   ||
			   sa.productsIdentical(prods.get(0), prods.get(1))
			  ||
			   (sa.unmappedInSameReactant(reacts) && sa.reactantSymmetricalOrIdentical(reacts))
			   ||
			   (sa.unmappedInSameProduct(prods) && sa.productSymmetricalOrIdentical(prods))
			  )*/
			if(sa.reactantSymmetricalOrIdentical(reacts) || sa.productSymmetricalOrIdentical(prods)){
				for(int i = 0;	i < reacts.size();	i++){
					
					int retNeighbourIndex	=	0;
					int retNeighbourMax		=	0;
					List<Integer> taken		=	new ArrayList<Integer>();
					for(int j = 0;	j < prods.size();	j++){
						int retNeighbourCount	=	this.findRetainedNeighborCount(reacts.get(i), 
																			   	   prods.get(j), 
																			   	   mappedReaction.getReactants().getAtomContainer(mappedReaction.getReactantContainerIndex(reacts.get(i))),
																			   	   mappedReaction.getProducts().getAtomContainer(mappedReaction.getProductContainerIndex(prods.get(i))));
						if(retNeighbourCount > retNeighbourMax && !taken.contains(j)){
							retNeighbourMax		=	retNeighbourCount;
							retNeighbourIndex	=	j;	
						}
					}
					IMapping newMap	=	new Mapping(reacts.get(i),prods.get(retNeighbourIndex));
					
					if(newMap != null){
						logger.info("Mapping two "+ reacts.get(i).getSymbol()+" atoms.");
						mappedReaction.addMapping(newMap);
					}
				}
			}
		}
	}
	
	/**Search for pairs of reactants and products of the same type
	 * The returned list contains the pairs of reactant atoms at index 0 and the pairs of product atoms at index 1.
	 * @return
	 */
	private List<List<List<IAtom>>> createPairs(){
		
		List<String> encounteredSymbols	=	new ArrayList<String>();
		List<List<IAtom>> pairsReact	=	new ArrayList<List<IAtom>>();
		List<List<IAtom>> pairsProd		=	new ArrayList<List<IAtom>>();
		
		for(IAtom atom:nonUniqueReactants){
			List<IAtom> pairReactant	=	new ArrayList<IAtom>();
			List<IAtom> pairProduct		=	new ArrayList<IAtom>();
			int occurenceOfAtomType		=	Tools.occurences(atom.getSymbol(), unmappedReactantSymbol);
			//only create a pair if there are two of that type!
			if( occurenceOfAtomType == 2 
			   && 
			   !encounteredSymbols.contains(atom.getSymbol()))
			{
				encounteredSymbols.add(atom.getSymbol());
				pairReactant.add(atom);
				for(int i = 0;	i < nonUniqueReactants.size();	i++){
					if(nonUniqueReactants.get(i).getSymbol().equals(atom.getSymbol())){
						pairReactant.add(nonUniqueReactants.get(i));
						break;
					}
				}
			}
			
			//inside the reactant loop: find product pairs with same symbol such that pairs with same 
			//symbol are at same index. 
			for(IAtom prodAtom:uniqueProducts){
				if(prodAtom.getSymbol().equals(atom.getSymbol()) 
				   && 
				   Tools.occurences(prodAtom.getSymbol(), unmappedProductSymbol) == 2)
					pairProduct.add(prodAtom);
			}
			
			if(pairReactant.size() == 2 && pairProduct.size() == 2){
				pairsReact.add(pairReactant);
				pairsProd.add(pairProduct);
			}
		}
		
		List<List<List<IAtom>>> ans	=	new ArrayList<List<List<IAtom>>>();
		ans.add(pairsReact);
		ans.add(pairsProd);
		
		return ans;
	}
	
	/**Create a mapping in the case that all atoms in the atom container are of a different type
	 */
	private void mapDifferentAtoms(){
		for(IAtom reactant:uniqueReactants){
			IMapping newMap	=	new Mapping(reactant,findCorrespondingAtom(reactant));
			if(newMap != null){
				logger.info("Mapping unique "+ reactant.getSymbol()+" atoms.");
				mappedReaction.addMapping(newMap);
			}
		}
	}
	
	/**Find an atom in the products with the same symbol as the reactant
	 * 
	 * @param reactant atom
	 * @return product atom with same symbol
	 */
	private IAtom findCorrespondingAtom(IAtom atom){
		
		for(IAtom product:uniqueProducts){
			if(product.getSymbol().equals(atom.getSymbol()))
				return product;
		}
		
		logger.error("Could not find corresponding atom type in unmapped products.");
		return null;
	}
	
	/**Find all unmapped atoms in the mapped reaction
	 * 
	 */
	private void findUnmapped(){
		
		int[][][] map	=	mappedReaction.getMappingIndices();
		
		for(int i = 0;	i < map.length;	i++){
			int rContainer	=	map[i][0][0];
			int rAtom		=	map[i][0][1];
			int pContainer	=	map[i][1][0];
			int pAtom		=	map[i][1][1];
			IAtom reactAtom	=	mappedReaction.getReactant(rContainer).getAtom(rAtom);
			IAtom prodAtom	=	mappedReaction.getProduct(pContainer).getAtom(pAtom);
			
			if(!mappedReactant.contains(reactAtom)){
				mappedReactant.add(reactAtom);
			}
			
			if(!mappedProduct.contains(prodAtom)){
				mappedProduct.add(prodAtom);
			}
		}
		
		List<IAtom> allReactant	=	mappedReaction.getReactantAtoms();
		List<IAtom> allProduct	=	mappedReaction.getProductAtoms();
		
		for(IAtom atom : allReactant){
			if(!mappedReactant.contains(atom) && !atom.getSymbol().equals("H")){
				unmappedReactant.add(atom);
				unmappedReactantSymbol.add(atom.getSymbol());
			}
		}
		
		for(IAtom atom : allProduct){
			if(!mappedProduct.contains(atom) && !atom.getSymbol().equals("H")){
				unmappedProduct.add(atom);
				unmappedProductSymbol.add(atom.getSymbol());
			}
		}
		
		//If different numbers of atoms are not mapped in the reactants and products, do not try to map them manually.
		//Reset the lists to empty so no mapping is attempted.
		if(unmappedProduct.size() != unmappedReactant.size()){
			logger.error("Different numbers of heavy atoms in reactants and products are not mapped. Not mapping ...");
			unmappedProduct	=	new ArrayList<IAtom>();
			unmappedReactant=	new ArrayList<IAtom>();
		}
		
		if(!unmappedReactant.isEmpty() || !unmappedProduct.isEmpty()){
			logger.info("Found unmapped atoms, perfoming additional mapping step.");
		}
	}
	
	/**Of the unmapped atoms, find which ones are unique (i.e. atom type only occurs once).
	 */
	private void findUnique(){
		
		for(IAtom reactant:unmappedReactant){
			if(Tools.occurences(reactant.getSymbol(),unmappedReactantSymbol) == 1){
				uniqueReactants.add(reactant);
			}
		}
		
		for(IAtom reactant:unmappedProduct){
			if(Tools.occurences(reactant.getSymbol(),unmappedProductSymbol) == 1){
				uniqueProducts.add(reactant);
			}
		}
	}
	
	/**Of the unmapped atoms, find which ones occur more than once
	 */
	private void findNonUnique(){
		
		for(IAtom react:unmappedReactant){
			if(!uniqueReactants.contains(react))
				nonUniqueReactants.add(react);
		}
		
		for(IAtom prod:unmappedProduct){
			if(!uniqueProducts.contains(prod))
				nonUniqueProducts.add(prod);
		}
	}
	
	/**Find the number of common neighbours between the two atoms, one in reactants, other in products
	 * 
	 * @param react
	 * @param prod
	 * @return
	 */
	private int findRetainedNeighborCount(IAtom react, IAtom prod, IAtomContainer reactant, IAtomContainer product){
		
		List<IAtom> neigboursR	=	reactant.getConnectedAtomsList(react);
		int ret					=	0;
		
		for(IAtom at:neigboursR){
			IAtom neigh	=	null;
			for(IMapping map : mappedReaction.getMappings()){
				if(map.getChemObject(0) == at){
					neigh	=	(IAtom) map.getChemObject(1);
					break;
				}
			}
			ret	=	product.getBond(prod, neigh) == null?ret:ret+1;
		}
		return ret;
	}
	
	public static boolean peroxyRadicals(ReactionCenter center){
		
		center.setOldHydrogenCounts();
		
		IAtom oxygen1	=	null;
		IAtom oxygen2	=	null;
		
		for(IAtom atom:center.getReaction().getReactantAtoms()){
			boolean found	=	false;
			if((boolean)atom.getProperty(StringConstants.CHANGED) && atom.getSymbol().equals("O")){
				Iterable<IAtom> neighbours	=	center.getReaction()
													  .getReactant(center.getReaction().getReactantContainerIndex(atom))
													  .getConnectedAtomsList(atom);
				for(IAtom neighbour:neighbours){
					if(neighbour.getSymbol().equals("O") && neighbour.getValency() == 1){
						oxygen2	=	neighbour;
						oxygen1	=	atom;
						found	=	true;
						break;
					}
				}
			}
			if(found)
				break;
		}
		
		if(oxygen2 != null){
			logger.info("Detected peroxide radicals. Adjusting mapping to find correct center.");
			IAtom mapped1	=	center.getReaction().getMappedProductAtom(oxygen1);
			IAtom mapped2	=	center.getReaction().getMappedProductAtom(oxygen2);
			int map1		=	-1;
			int map2		=	-1;
			int i			=	0;
			
			for(IMapping map:center.getReaction().getReaction().mappings()){
				if(map.getChemObject(1) == mapped1 && map.getChemObject(0) == oxygen1){
					map1	=	i;
					break;
				}
				
				i++;
			}

			center.getReaction().getReaction().removeMapping(map1);
			i=0;
			
			for(IMapping map:center.getReaction().getReaction().mappings()){
				if(map.getChemObject(1) == mapped2 && map.getChemObject(0) == oxygen2){
					map2	=	i;
					break;
				}
				i++;
			}
			
			center.getReaction().getReaction().removeMapping(map2);
			
			center.getReaction().getReaction().addMapping(new Mapping(oxygen1,mapped2));
			center.getReaction().getReaction().addMapping(new Mapping(oxygen2,mapped1));
			
			center.getReaction().reassignMappings();
			
			ReactionCenterDetector rcd	=	new ReactionCenterDetector(center.getReaction());
			ReactionCenter newCenter	=	rcd.detectReactionCenters();
			if(newCenter.noChanges() || !newCenter.radicalMechanismOK()){
				return false;
			}
			else{
				center.changeTo(newCenter);
				return true;
			}
		}else{
			return false;
		}
	}
}