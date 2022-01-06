package changes;

import java.util.List;

import org.openscience.cdk.CDKConstants;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IBond;
import org.openscience.cdk.interfaces.IDoubleBondStereochemistry;
import org.openscience.cdk.interfaces.IStereoElement;
import org.openscience.cdk.interfaces.ITetrahedralChirality;
import org.openscience.cdk.ringsearch.RingSearch;

import constants.StringConstants;
import reactionMapping.MappedReaction;

/**Class to detect the type of changes that an atom undergoes during a reaction. Unless explicitly stated otherwise,
 * all methods assume the mappedAtom to be a reactant!
 * 
 * @author pplehier
 *
 */
public class ChangeDetector {

	private MappedReaction mappedrxn;
	private IAtom mappedAtom;
	
	public ChangeDetector(MappedReaction mappedrxn, IAtom mappedAtom2){
		
		this.mappedrxn	=	mappedrxn;
		this.mappedAtom	=	mappedAtom2;
	}

	/**An atom is considered to have changed if either of the following is applicable:<br>
	 * - the number of (implicit) hydrogens has changed<br>
	 * - the stereo chemical configuration of the atom has changed<br>
	 * - the bond order of any of the connected bonds has changed<br>
	 * - the number of neighbours of the atom has changed<br>
	 * - the charge on the atom has changed<br>
	 * - the number of single electrons (radicals) on the atom has changed<br>
	 * 
	 * @return changed
	 */
	public boolean detectChanged(){
				
		boolean neighbours		=	!neighboursAreEqual();
		boolean bonds			=	!bondsAreEqual();
		boolean hydrogen		=	!hydrogensAreEqual();
		boolean singleElectrons	=	!singleElectronsAreEqual();
		boolean charges			=	!chargesAreEqual();
		boolean stereo			=	!stereoIsEqual(mappedAtom);
		boolean neighbourStereo	=	neighbourStereoChanged();
		boolean hasChanged		=	neighbours||bonds||hydrogen||singleElectrons||charges;
		
		mappedAtom.setProperty(StringConstants.BONDCHANGE, 			bonds);
		mappedAtom.setProperty(StringConstants.CHARGECHANGE, 		charges);
		mappedAtom.setProperty(StringConstants.HYDROGENCHANGE, 		hydrogen);
		mappedAtom.setProperty(StringConstants.NEIGHBOURCHANGE, 	neighbours);
		mappedAtom.setProperty(StringConstants.SINGLEELECTRONCHANGE,singleElectrons);
		mappedAtom.setProperty(StringConstants.STEREOCHANGE, 		stereo);
		mappedAtom.setProperty(StringConstants.CHANGED,				hasChanged);
		mappedAtom.setProperty(StringConstants.HYDROGENDIFFERENCE, 	hydrogenHasDecreased(true));
		mappedAtom.setProperty(StringConstants.NEIGHBOURSTEREO,		neighbourStereo);
		
		return hasChanged;
	}
	/**Check whether the bond orders of a reactant atom have been changed by the reaction
	 * 
	 * @return
	 */
	private boolean bondsAreEqual(){
		
		List<IBond> bondsR	=	mappedrxn.getReactants()
										 .getAtomContainer(mappedrxn.getReactantContainerIndex(mappedAtom))
										 .getConnectedBondsList(mappedAtom);
		List<IBond> bondsP	=	mappedrxn.getProducts()
										 .getAtomContainer(mappedrxn.getContainerIndexMappedAtom(mappedAtom))
										 .getConnectedBondsList(mappedrxn.getMappedProductAtom(mappedAtom));
		boolean equal		=	false;
		
		if(bondsR.size() >= bondsP.size()){
			for(int i = 0;	i<bondsR.size();	i++){
				equal	=	false;
				for(int j = 0;	j < bondsP.size();	j++){
				
					boolean match	=	mappedrxn.getMappedProductAtom(bondsR.get(i)
																			 .getConnectedAtom(mappedAtom))
										==
										bondsP.get(j)
											  .getConnectedAtom(mappedrxn.getMappedProductAtom(mappedAtom));
					
					int orderN		=	bondsR.get(i)
											  .getOrder()
											  .compareTo(bondsP.get(j).getOrder());
					
					//Consider all bond changes, even aromatic to aromatic. In case of aromaticity, the bonds will
					//be resonated, so there will always be a case in which initially non-matching double and
					//single bonds will match.
					boolean order;
						if(orderN != 0){order	=	false;}
						else{order	=	true;}
					
					equal	=	(equal)||(match && order);
	
				}
	
				if(!equal){return equal;}
			}
			
		}else{
			for(int j = 0;	j < bondsP.size();	j++){
				equal	=	false;
				for(int i = 0;	i < bondsR.size();	i++){
				
					boolean match	=	mappedrxn.getMappedProductAtom(bondsR.get(i)
																		  	 .getConnectedAtom(mappedAtom))
										==
										bondsP.get(j)
											  .getConnectedAtom(mappedrxn.getMappedProductAtom(mappedAtom));
					
					int orderN		=	bondsR.get(i)
											  .getOrder()
											  .compareTo(bondsP.get(j).getOrder());
					
					boolean bothAromatic	=	bondsR.get(i).getFlag(CDKConstants.ISAROMATIC) && bondsP.get(j).getFlag(CDKConstants.ISAROMATIC);
					

					//If both bonds are aromatic, don't consider the order change (because the aromaticity is 
					//made explicit, the position of the double and single bonds can shift.
					boolean order;
					if(orderN != 0 && !bothAromatic){order	=	false;}
						else{order	=	true;}
					
					equal	=	(equal) || (match && order);
	
				}
	
				if(!equal){return equal;}
			}
		}
		return true;
	}

	/** Check whether the reactant atom's charge has been changed by the reaction
	 * 
	 * @param mappedAtom
	 * @return
	 */
	private boolean chargesAreEqual(){
		
		double chargeR	=	mappedAtom.getFormalCharge();
		double chargeP	=	mappedrxn.getMappedProductAtom(mappedAtom)
									 .getFormalCharge();
		return chargeR == chargeP;
	}

	/** This method checks whether a non-ring reactant atom has become part of a ring in the product or 
	 * vice-versa
	 * Useful before adding method to ensure connected centers. Can result in over-specification of reaction
	 * centers
	 * 
	 * @param mappedAtom
	 * @return
	 * @deprecated
	 */
	@SuppressWarnings("unused")
	private boolean hasBecomeRingAtom(){
				
		RingSearch rsR	=	new RingSearch(mappedrxn.getReactants()
													.getAtomContainer(mappedrxn.getReactantContainerIndex(mappedAtom)));
		RingSearch rsP	=	new RingSearch(mappedrxn.getProducts()
													.getAtomContainer(mappedrxn.getContainerIndexMappedAtom(mappedAtom)));
		
		boolean inreactant	=	rsR.cyclic(mappedAtom);
		boolean inproduct	=	rsP.cyclic(mappedrxn.getMappedProductAtom(mappedAtom));
		
		return inreactant != inproduct;
	}

	/**For defining the correct mechanism, missing hydrogen atoms must be added if the number of implicit hydrogens 
	 * decreases. This method determines by how much the implicit hydrogencount changes for [true] a reactant
	 * or [false] a product.
	 * 
	 * @param mappedAtom
	 * @param RP: flag whether reactants (true) or products (false) are to be considered
	 * @return change in hydrogen atoms
	 */
	private int hydrogenHasDecreased(boolean RP){
	
		if(RP){return hydrogenHasDecreasedReactant();}
		else{return hydrogenHasDecreasedProduct();}
	}

	/**For defining the correct mechanism, missing hydrogen atoms must be added if the number of implicit hydrogens 
	 * decreases. This method determines by how much the implicit hydrogencount changes for a product.
	 * 
	 * @param mappedAtom
	 * @return change in hydrogen atoms between reactant and product
	 */
	private int hydrogenHasDecreasedProduct(){
	
		int hydrP	=	mappedAtom.getImplicitHydrogenCount();
		
		List<IBond> pBonds	=	mappedrxn.getProducts()
										 .getAtomContainer(mappedrxn.getProductContainerIndex(mappedAtom))
										 .getConnectedBondsList(mappedAtom);
	
		for(IBond bond:pBonds){
			if(bond.getConnectedAtom(mappedAtom).getSymbol().equals("H")){
				hydrP++;
			}
		}
		
		int hydrR	=	mappedrxn.getMappedReactantAtom(mappedAtom)
							 	 .getImplicitHydrogenCount();
	
		List<IBond> rBonds	=	mappedrxn.getReactants()
			 	 						 .getAtomContainer(mappedrxn.getContainerIndexMappedAtom(mappedAtom))
			 	 						 .getConnectedBondsList(mappedrxn.getMappedReactantAtom(mappedAtom));
			 	 						
			 	 						 
	
		for(IBond bond:rBonds){
			if(bond.getConnectedAtom(mappedrxn.getMappedReactantAtom(mappedAtom)).getSymbol().equals("H")){
				hydrR++;
			}
		}
		
		return hydrP - hydrR;
	}

	/**For defining the correct mechanism, missing hydrogen atoms must be added if the number of implicit hydrogens 
		 * decreases. This method determines by how much the implicit hydrogen count changes for a reactant.
		 * 
		 * @param mappedAtom
		 * @return change in hydrogen atoms between product and reactant
		 */
		private int hydrogenHasDecreasedReactant(){
			
			int hydrR	=	mappedAtom.getImplicitHydrogenCount();
			int hydrP	=	mappedrxn.getMappedProductAtom(mappedAtom)
									 .getImplicitHydrogenCount();
			
			return hydrR - hydrP;
		}

	/**Checks whether the number of hydrogens connected to the reactant atom has been changed during the reaction
	 * 
	 * @param mappedAtom
	 * @return 
	 */
	private boolean hydrogensAreEqual(){
		
		int hydrR	=	mappedAtom.getImplicitHydrogenCount();
		int hydrP	=	mappedrxn.getMappedProductAtom(mappedAtom)
								 .getImplicitHydrogenCount();
		
		return hydrR == hydrP;
	}

	/**Check whether the number of neighbours of a reactant atom has been changed by the reaction
	 * 
	 * @param mappedAtom
	 * @return
	 */
	private boolean neighboursAreEqual(){
		
		List<IAtom> neighboursR	=	mappedrxn.getReactants()
											 .getAtomContainer(mappedrxn.getReactantContainerIndex(mappedAtom))
											 .getConnectedAtomsList(mappedAtom);
		List<IAtom> neighboursP	=	mappedrxn.getProducts()
											 .getAtomContainer(mappedrxn.getContainerIndexMappedAtom(mappedAtom))
											 .getConnectedAtomsList(mappedrxn.getMappedProductAtom(mappedAtom));
		
		if(neighboursR.size() == neighboursP.size()){
			for(int i = 0;	i < neighboursR.size();	i++){
				boolean equal	=	false;
			
				for(int j = 0;	j < neighboursP.size();	j++){
					//neighbours are equal if there is a product neighbour that maps to a reactant neighbour.
	
					equal	=	equal
								||
								mappedrxn.getMappedProductAtom(neighboursR.get(i))
								== neighboursP.get(j);
				
				}
				
				if(!equal){return equal;}
			}
			
		}else{
			for(int j = 0;	j < neighboursP.size();	j++){
				boolean equal	=	false;
			
				for(int i = 0;	i < neighboursR.size();	i++){
					//neighbours are equal if there is a product neighbour that maps to a reactant neighbour.
	
					equal	=	equal
								||
								mappedrxn.getMappedProductAtom(neighboursR.get(i))
								== neighboursP.get(j);
				
				}
				
				if(!equal){return equal;}
			}
		}
		
		return true;
	}

	/** Method to determine whether the stereo configuration of a neighbouring atom has changed. If this is the case
	 * the atom will be included in the reactive center.
	 * 
	 * @param atom
	 * @return
	 */
	private boolean neighbourStereoChanged(){
		
		List<IAtom> neighbours	=	mappedrxn.getReactants()
				 							 .getAtomContainer(mappedrxn.getReactantContainerIndex(mappedAtom))
				 							 .getConnectedAtomsList(mappedAtom);
		
		boolean stereo	=	false;
		
		for(int i = 0;	i < neighbours.size();	i++){
			stereo	=	stereo || !stereoIsEqual(neighbours.get(i));
		}
		
		return stereo;
	}

	/** This method checks whether the number of single electrons on a reactant atom has been changed by the
	 * reaction
	 * 
	 * @param mappedAtom
	 * @return
	 */
	private boolean singleElectronsAreEqual(){
		
		int singleER	=	mappedrxn.getReactants()
									 .getAtomContainer(mappedrxn.getReactantContainerIndex(mappedAtom))
									 .getConnectedSingleElectronsCount(mappedAtom);
		
		int singleEP	=	mappedrxn.getProducts()
									 .getAtomContainer(mappedrxn.getContainerIndexMappedAtom(mappedAtom))
									 .getConnectedSingleElectronsCount(mappedrxn.getMappedProductAtom(mappedAtom));
		
		return singleER == singleEP;
	}

	/**Check whether the stereo configuration of the reactant atom has been changed by the reaction
	 * Only checks whether stereo has changed, not whether the connected stereo atoms have changed.
	 * TODO:necessary when stereo is formed?
	 * @param mappedAtom
	 * @return
	 */
	private boolean stereoIsEqual(IAtom mappedAtom){
		
		int AcIndex	=	mappedrxn.getReactantContainerIndex(mappedAtom);
		
		for(IStereoElement el:mappedrxn.getReactants()
									   .getAtomContainer(AcIndex)
									   .stereoElements()){
			
			if(el instanceof ITetrahedralChirality){
				if(mappedAtom.equals(((ITetrahedralChirality) el).getChiralAtom())){
					
					for(IStereoElement prodEl:mappedrxn.getProducts()
													   .getAtomContainer(mappedrxn.getContainerIndexMappedAtom(mappedAtom))
													   .stereoElements()){
						if(prodEl instanceof ITetrahedralChirality){
							
							boolean containsAtom	=	((ITetrahedralChirality) prodEl).getChiralAtom()
														==
														mappedrxn.getMappedProductAtom(((ITetrahedralChirality) el).getChiralAtom());
							boolean sameStereo		=	((ITetrahedralChirality) prodEl).getStereo()
														==
														((ITetrahedralChirality) el).getStereo();
							
							if(containsAtom && !sameStereo){
								return false;
							}
						}
					}
					
					return false;
				}
			}
			else 
				if(el instanceof IDoubleBondStereochemistry){
					
					Iterable<IAtom> atoms	=	((IDoubleBondStereochemistry) el).getStereoBond().atoms();
					boolean isPartOfBond	=	false;
					//Check whether mappedAtom is part of this stereochemical bond.
					for(IAtom atom:atoms){
						isPartOfBond	=	atom == mappedAtom;
						if(isPartOfBond){break;}
					}
					//If it is part of the bond, find the corresponding bond in the products and compare the 
					//stereo property of that bond.
					if(isPartOfBond){
						for(IStereoElement elP:mappedrxn.getProducts()
														.getAtomContainer(mappedrxn.getContainerIndexMappedAtom(mappedAtom))
														.stereoElements()){
							
							if(elP instanceof IDoubleBondStereochemistry){
								boolean containsMapped	=	((IDoubleBondStereochemistry) elP).getStereoBond()
															.contains(mappedrxn.getMappedProductAtom(mappedAtom));
								boolean hasSameStereo	=	((IDoubleBondStereochemistry) el).getStereo()
															==
															((IDoubleBondStereochemistry) elP).getStereo();
								
								if(containsMapped){
								
									return hasSameStereo;
								}
							}
							//If this line is reached, it means that there is no corresponding stereochemical (double
							//bond in the products. The stereochemistry has been lost or has changed to chiral -> false;
							return false;
						}
						//if no stereobond elements exist in the products: stereo lost
						return false;
					}
				}
		}
	return true;
	}
	
	/**Public access to the change in hydrogen atoms for a specified atom
	 * 
	 * @param reaction
	 * @param atom
	 * @param reactantOrProduct
	 * @return hydrogen change
	 */
	public static int hydrogenHasDecreased(MappedReaction reaction, IAtom atom, boolean reactantOrProduct){
		
		ChangeDetector cd	=	new ChangeDetector(reaction, atom);
		
		return cd.hydrogenHasDecreased(reactantOrProduct);
	}
}