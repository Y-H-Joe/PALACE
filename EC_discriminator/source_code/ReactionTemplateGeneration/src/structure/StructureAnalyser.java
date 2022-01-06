package structure;

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.inchi.InChIGenerator;
import org.openscience.cdk.inchi.InChIGeneratorFactory;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IBond;
import org.openscience.cdk.smsd.Isomorphism;
import org.openscience.cdk.smsd.interfaces.Algorithm;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;

import manipulators.RadicalHandler;
import reactionCenterDetection.ReactionCenter;
import reactionMapping.MappedReaction;

public class StructureAnalyser {
	
	private static Logger logger	=	Logger.getLogger(StructureAnalyser.class);
	private MappedReaction mappedReaction;
	
	public StructureAnalyser(MappedReaction mappedReaction){
		
		this.mappedReaction	=	mappedReaction;
	}
	
	public StructureAnalyser(ReactionCenter RC){
		
		this.mappedReaction	=	RC.getReaction();
	}
	
	/**Assess whether a product molecule to which the prods belong is symmetrical whith respect to the 
	 * two specified atoms.
	 * 
	 * @param prods (contains 2 atoms)
	 * @return symmetrical?
	 * @throws CDKException
	 */
	public boolean productSymmetricalOrIdentical(List<IAtom> prods) throws CDKException{
		
		if(prods.size() != 2){
			logger.error("Expected two atoms, received "+prods.size()+".");
			return false;
		}
		
		IAtomContainer mol1	=	mappedReaction.getProduct(mappedReaction.getProductContainerIndex(prods.get(0)));
		IAtomContainer mol2	=	mappedReaction.getProduct(mappedReaction.getProductContainerIndex(prods.get(1)));
		
		return isSymmetrical(prods.get(0), prods.get(1), mol1, mol2);
	}

	/**Assess whether a reactant molecult to which the reacts belong is symmetrical whith respect to the
	 * two specified atoms.
	 * 
	 * @param reacts (contains 2 atoms)
	 * @return symmetrical?
	 * @throws CDKException
	 */
	public boolean reactantSymmetricalOrIdentical(List<IAtom> reacts) throws CDKException{
		
		if(reacts.size() != 2){
			logger.error("Expected two atoms, received "+reacts.size()+".");
			return false;
		}
		
		IAtomContainer mol1	=	mappedReaction.getProduct(mappedReaction.getReactantContainerIndex(reacts.get(0)));
		IAtomContainer mol2	=	mappedReaction.getProduct(mappedReaction.getReactantContainerIndex(reacts.get(1)));
		return isSymmetrical(reacts.get(0), reacts.get(1), mol1, mol2);
	}

	/**Assess whether a molecule is symmetrical with respect to the two atoms specified.
	 * 
	 * @param atoms
	 * @param mol
	 * @return
	 * @throws CDKException
	 */
	public static boolean isSymmetrical(IAtom atom1, 
										IAtom atom2, 
										IAtomContainer mol1, 
										IAtomContainer mol2) throws CDKException{
	
		if(mol1.contains(atom1) && mol2.contains(atom2)){
			int depth			=	1;
			boolean[] result	=	sameEnvironment(atom1, atom2, mol1, mol2,depth++);

			while(result[0] && !result[1]){
				result	=	sameEnvironment(atom1, atom2, mol1, mol2,depth++);
			}

			return result[0];
		}
		else if(mol1.contains(atom2) && mol2.contains(atom1)){
			int depth			=	1;
			boolean[] result	=	sameEnvironment(atom2, atom1, mol1, mol2,depth++);

			while(result[0] && !result[1]){
				result	=	sameEnvironment(atom2, atom1, mol1, mol2,depth++);
			}

			return result[0];
		}
		else{
			logger.error("Wrong atoms and molecules specified: atoms not present in given molecules.");
			return false;
		}
	}

	/**Assess whether the environment around two atoms (in the same molecule) is the same.
	 * 
	 * @return same environment?
	 * @throws CDKException 
	 */
	private static boolean[] sameEnvironment(IAtom atom1, 
											 IAtom atom2, 
											 IAtomContainer mol1, 
											 IAtomContainer mol2, 
											 int depth) throws CDKException{
		
		boolean[] ans	=	{false,false};
		//if the two compared atoms aren't of the same type, they aren't in equivalent centra.
		if(!atom1.getSymbol().equals(atom2.getSymbol())){
			return ans;
		}
		
		IAtomContainer[] environments	=	findEnvironments(atom1, atom2, mol1, mol2, depth);
		Isomorphism iso					=	new Isomorphism(Algorithm.CDKMCS,true);
		
		iso.init(environments[0], environments[1], false, false);
		
		ans[0]	=	iso.getFirstMapping().size() == environments[0].getAtomCount()
					&&
					iso.getFirstAtomMapping().size() == environments[1].getAtomCount();
		ans[1]	=	environments[0].getProperty("Molecule size reached");
		
		return ans; 
	}

	/**Create an environment for two given atoms in the same molecule. The molecule is searched until the specified
	 * depth is reached or the environment has the same number of atoms as the molecule.<br>
	 * The returned array contains one environment for each of the two specified atoms
	 * 
	 * @param atoms
	 * @param mol
	 * @param searchDepth
	 * @return both environments
	 */
	private static IAtomContainer[] findEnvironments(IAtom atom1, 
													 IAtom atom2, 
													 IAtomContainer mol1, 
													 IAtomContainer mol2, 
													 int searchDepth){
		
		IAtomContainer env1		=	DefaultChemObjectBuilder.getInstance().newInstance(IAtomContainer.class);
		IAtomContainer env2		=	DefaultChemObjectBuilder.getInstance().newInstance(IAtomContainer.class);
		//Initiate the first environment: minimal search depth is 1, environment contains the specified atom
		List<IAtom> envAtoms1	=	mol1.getConnectedAtomsList(atom1);
		envAtoms1.add(atom1);
		//Initiate the list of atoms that will be searched for the next layer (exludes the specified atom)
		List<IAtom> search1		=	mol1.getConnectedAtomsList(atom1);
		//Do the same for the second atom.
		List<IAtom> envAtoms2	=	mol2.getConnectedAtomsList(atom2);
		envAtoms2.add(atom2);
		List<IAtom> search2		=	mol2.getConnectedAtomsList(atom2);
		//The bonds
		List<IBond> envBonds1	=	mol1.getConnectedBondsList(atom1);
		List<IBond> envBonds2	=	mol2.getConnectedBondsList(atom2);
		//Max depth from arguments
		int maxSearchDepth		=	searchDepth;
		//Current depth
		int depth				=	1;
		//Stop condition
		boolean stop			=	envAtoms1.size() >= mol1.getAtomCount() || envAtoms2.size() >= mol2.getAtomCount();
		
		while(depth < maxSearchDepth && !stop){
			List<IAtom> newSearch1	=	new ArrayList<IAtom>();
			List<IAtom> tempAtEnv1	=	new ArrayList<IAtom>(envAtoms1);
			List<IBond> tempBoEnv1	=	new ArrayList<IBond>(envBonds1);
			List<IAtom> tempAtEnv2	=	new ArrayList<IAtom>(envAtoms2);
			List<IBond> tempBoEnv2	=	new ArrayList<IBond>(envBonds2);
			
			for(IAtom atom:search1){
				//Connected atoms to the search atom are candidates to be added to the environment.
				List<IAtom> envAtoms1i	=	mol1.getConnectedAtomsList(atom);
				List<IBond> envBonds1i	=	mol1.getConnectedBondsList(atom);
				//-1 because one of the atoms will always be one that is already present
				if(envAtoms1i.size() + envAtoms1.size() - 1 < mol1.getAtomCount()){
					for(IAtom candAt:envAtoms1i){
						//Only add a candidate if it isn't already present.
						if(!envAtoms1.contains(candAt)){
							envAtoms1.add(candAt);
							newSearch1.add(candAt);
						}
					}
					//same for the bonds
					for(IBond candBo:envBonds1i){
						if(!envBonds1.contains(candBo)){
							envBonds1.add(candBo);
						}
					}
				}
				//if the environment reaches the same size as the molecule: stop
				else{
					stop	=	true;
					break;
				}
			}
			//set the new list of atoms that should be searched: doesn't contain the already encountered ones
			search1	=	newSearch1;
			
			List<IAtom> newSearch2	=	new ArrayList<IAtom>();
			//repeat for the second atom
			for(IAtom atom:search2){
				List<IAtom> envAtoms2i	=	mol2.getConnectedAtomsList(atom);
				List<IBond> envBonds2i	=	mol2.getConnectedBondsList(atom);
	
				if(envAtoms2i.size() + envAtoms2.size() - 1 < mol2.getAtomCount()){
					for(IAtom candAt:envAtoms2i){
						if(!envAtoms2.contains(candAt)){
							envAtoms2.add(candAt);
							newSearch2.add(candAt);
						}
					}
					for(IBond candBo:envBonds2i){
						if(!envBonds2.contains(candBo)){
							envBonds2.add(candBo);
						}
					}
				}
				else{
					stop	=	true;
					break;
				}
			}
			
			//in case a stop occurred, undo the search results of this loop!
			if(stop){
				envAtoms1	=	tempAtEnv1;
				envBonds1	=	tempBoEnv1;
				envAtoms2	=	tempAtEnv2;
				envBonds2	=	tempBoEnv2;
			}
			
			search2	=	newSearch2;
			//Go deeper
			depth++;
		}
		//Create molecules from the lists of atoms and bonds.
		env1.setAtoms(envAtoms1.toArray(new IAtom[envAtoms1.size()]));
		env1.setBonds(envBonds1.toArray(new IBond[envBonds1.size()]));
		env2.setAtoms(envAtoms2.toArray(new IAtom[envAtoms2.size()]));
		env2.setBonds(envBonds2.toArray(new IBond[envBonds2.size()]));
		
		try{
			AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(env1);
			new RadicalHandler().correctMultipleRadicals(env1);
			AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(env2);
			new RadicalHandler().correctMultipleRadicals(env2);
		}catch(CDKException e){}
		
		//necessary for symmetry determination
		if(stop){
			env1.setProperty("Molecule size reached", true);
			env2.setProperty("Molecule size reached", true);
		}
		else{
			env1.setProperty("Molecule size reached", false);
			env2.setProperty("Molecule size reached", false);
		}
		
		IAtomContainer[] ans	=	new IAtomContainer[2];
		ans[0]	=	env1;
		ans[1]	=	env2;
		
		return ans;
	}

	/**Check whether the products to which the specified atoms belong are the same molecule
	 * 
	 * @param atom1
	 * @param atom2
	 * @return
	 */
	public boolean productsIdentical(IAtom atom1, IAtom atom2){
		
		IAtomContainer reactant1	=	mappedReaction.getProduct(mappedReaction.getProductContainerIndex(atom1));
		IAtomContainer reactant2	=	mappedReaction.getProduct(mappedReaction.getProductContainerIndex(atom2));
		
		return containersIdentical(reactant1, reactant2);
	}

	/**Check whether the reactants to which the specified atoms belong are the same molecule
	 * 
	 * @param atom1
	 * @param atom2
	 * @return
	 */
	public boolean reactantsIdentical(IAtom atom1, IAtom atom2){
		
		IAtomContainer reactant1	=	mappedReaction.getReactant(mappedReaction.getReactantContainerIndex(atom1));
		IAtomContainer reactant2	=	mappedReaction.getReactant(mappedReaction.getReactantContainerIndex(atom2));
		
		return containersIdentical(reactant1, reactant2);
	}

	/**Assess whether the two containers are identical.
	 * 
	 * @return identical
	 */
	private boolean containersIdentical(IAtomContainer atomContainer1, IAtomContainer atomContainer2){
		
		IAtomContainer[] molecules	=	new IAtomContainer[2];
		InChIGenerator[] inChIs		=	new InChIGenerator[2];
		molecules[0]				=	atomContainer1;
		molecules[1]				=	atomContainer2;
		
		for(int j = 0;	j < inChIs.length;	j++){
			try {
				inChIs[j]	=	InChIGeneratorFactory.getInstance().getInChIGenerator(molecules[j]);
			} catch (CDKException e) {
				logger.warn("Failed to generate InChI! Cannot assess equality of molecules.");
				return false;
			}
		}
		
		if(inChIs[0].getInchi().equals(inChIs[1].getInchi())){
			return true;
		}
		
		return false;
	}

	/**Check whether the two unmapped reactant atoms are in the same reactant molecule.
	 * 
	 * @return
	 */
	public boolean unmappedInSameReactant(List<IAtom> reacts){
		
		if(reacts.size() != 2){
			logger.error("Expected two atoms, received "+reacts.size()+".");
			return false;
		}
		
		int index1	=	mappedReaction.getReactantContainerIndex(reacts.get(0));
		int index2	=	mappedReaction.getReactantContainerIndex(reacts.get(1));
		
		return index1 == index2;
	}

	/**Check whether the two unmapped product atoms are in the same product molecule.
	 * 
	 * @return
	 */
	public boolean unmappedInSameProduct(List<IAtom> prods){
		
		if(prods.size() != 2){
			logger.error("Expected two atoms, received "+prods.size()+".");
			return false;
		}
		
		int index1	=	mappedReaction.getProductContainerIndex(prods.get(0));
		int index2	=	mappedReaction.getProductContainerIndex(prods.get(1));
		
		return index1 == index2;		
	}

}
