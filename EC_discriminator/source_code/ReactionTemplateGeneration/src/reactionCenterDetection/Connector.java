package reactionCenterDetection;

import java.util.ArrayList;
import java.util.List;

import org.openscience.cdk.CDKConstants;
import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.graph.PathTools;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IBond;

import constants.StringConstants;
import tools.MyMath;

/**
 * This class looks ensures that a reaction center will be connected (ie comprises only one single molecular 
 * entity). It uses the depth-first minimal path algorithm of cdk to determine different paths between the
 * different fragments, which are identified first.
 * The shortest total connection is then chosen to be used, and all atoms are flagged as "to be removed" are 
 * changed to "should keep".
 * 
 * The first step in this method is generating a matrix (representation of the graph in which nodes are atoms that 
 * have changed and edges are the number of atoms on the shortest path between them) of the connections between the
 * different "keeper" atoms (that are changed by the reaction).
 * 
 * 
 * @author pplehier
 *
 */
public class Connector {

	private IAtomContainer container;
	private List<IAtom> notCheckedAtoms;
	private List<List<IAtom>> fragmentKeepers;
	private List<List<IAtomContainer>> possibleConnections;
	private List<IAtomContainer> shortestConnections;
	private int fragment;

	/**
	 * @category constructor
	 * @param removeSetContainer
	 */
	public Connector(IAtomContainer removeSetContainer){
		
		this.container			=	removeSetContainer;
		this.notCheckedAtoms	=	new ArrayList<IAtom>();
		for(IAtom atom:container.atoms()){
			notCheckedAtoms.add(atom);
		}
		this.fragmentKeepers	=	new ArrayList<List<IAtom>>();
		this.fragment			=	0;
		this.possibleConnections=	new ArrayList<List<IAtomContainer>>();
		this.shortestConnections=	new ArrayList<IAtomContainer>();
	}
	
	/**This method sets the properties of all atoms such that the remaining atoms will form a connected structure
	 * This is necessary to come to a valid SMARTS representation of the molecule.
	 * 
	 * @throws CDKException
	 */
	public void connectAndSetProperties() throws CDKException{
		
		this.fillFragments();
		
		if(this.fragmentKeepers.size() == 1){return;}
		else{
			this.initiateConnections();
			this.findPossibleConnections();
			this.findShortestConnections();
			this.setKeeperConnections();
		}
	}
	
	/**Build an new array of possible connections
	 * 
	 */
	private void initiateConnections(){
		
		for(int i = 0;	i < fragmentKeepers.size();	i++){
			possibleConnections.add(new ArrayList<IAtomContainer>());
			
			for(int j = 0;	j < fragmentKeepers.size(); j++){
				possibleConnections.get(i).add(DefaultChemObjectBuilder.getInstance().newInstance(IAtomContainer.class));
			}
		}
	}
	
	/**Set the atom properties of the atoms that should be kept. Only atoms in the shortest connections list 
	 * will be kept
	 */
	private void setKeeperConnections(){
		
		for(int i = 0;	i < shortestConnections.size();	i++){
			for(int j = 0;	j < shortestConnections.get(i).getAtomCount();	j++){
				IAtom atom	=	shortestConnections.get(i).getAtom(j);
				
				if(!(boolean)atom.getProperty(StringConstants.SHOULDKEEP)){
					atom.setProperty(StringConstants.SHOULDKEEP, true);
					atom.setProperty(StringConstants.LINK, true);
				}
			}
		}
	}
	
	/**Find the shortest connection between all reactive centers, by determining the maximum spanning tree of 
	 * the graph that consists of all centers and connections. The edge weights are the bond counts of the connection.
	 * 
	 * @throws CDKException
	 */
	private void findShortestConnections() throws CDKException{
		
		List<List<Double>> bondCounts	=	 new ArrayList<List<Double>>();
		for(int i = 0;	i < possibleConnections.size(); i++){
			List<Double> bondCount	=	new ArrayList<Double>();
			for(int j = 0;	j < possibleConnections.get(i).size();	j++){
				int aC	=	possibleConnections.get(i).get(j).getBondCount();
				int aC2	=	possibleConnections.get(j).get(i).getBondCount();
			
				//PossibleConnections matrix is constructed as above diagonal matrix only: spanning tree needs full
				//matrix: mirror
				if(aC == 0)
					bondCount.add((double)aC2);
				else
					bondCount.add((double)aC);
				
			}
			//Fragments contain separate atoms, atoms of the same center will have a link containing no atoms!
			if(!bondCount.isEmpty())
				bondCounts.add(bondCount);
		}
		
		bondCounts						=	MyMath.removeZeroRowAndColumn(bondCounts);
		List<List<Integer>> shortest	=	MyMath.minimalSpanningTree(bondCounts);
		
		for(List<Integer> pair:shortest){
			shortestConnections.add(possibleConnections.get(pair.get(0)).get(pair.get(1)));
		}
	}
	
	/**Find all possible connections, between all atoms in the center. Search via breadth first search on the full
	 * molecular graph.
	 * 
	 */
	private void findPossibleConnections(){

		for(int i = 0;	i < fragmentKeepers.size();	i++){
			for(int j = 0;	j < fragmentKeepers.get(i).size();	j++){
				IAtom rootAtom	=	fragmentKeepers.get(i).get(j);
				
				for(int k = i+1;	k < fragmentKeepers.size();	k++){
					for(int l = 0;	l < fragmentKeepers.get(k).size(); l++){
						IAtom targetAtom			=	fragmentKeepers.get(k).get(l);
						IAtomContainer temp			=	DefaultChemObjectBuilder.getInstance().newInstance(IAtomContainer.class);
						IAtomContainer breadthCont	=	DefaultChemObjectBuilder.getInstance().newInstance(IAtomContainer.class);
						List<IAtom> sphere			=	new ArrayList<IAtom>();
						
						unsetVisited(container);
						sphere.add(rootAtom);
						
						//Do a breadth first search first (ensures that the path found in the consequent depth first
						//search will be the shortest.
						PathTools.breadthFirstTargetSearch(container, sphere, targetAtom, breadthCont, 0, container.getAtomCount());
						unsetVisited(breadthCont);
						if(breadthCont.contains(targetAtom))
							PathTools.depthFirstTargetSearch(breadthCont, 
															 rootAtom,
															 targetAtom, 
															 temp);
						else
							PathTools.depthFirstTargetSearch(container, 
									 						 rootAtom,
									 						 targetAtom, 
									 						 temp);
						
						if(possibleConnections.get(i).get(k).getAtomCount() == 0){
							possibleConnections.get(i).set(k, temp);
						}
						else if(temp.getAtomCount() != 0){
							if(possibleConnections.get(i).get(k).getAtomCount() > temp.getAtomCount()){
								possibleConnections.get(i).set(k, temp);
							}
						}
					}
				}
			}
		}
	}
	
	/**Add all reaction centra to a list.
	 * Each fragment corresponds to a single reaction center
	 */
	private void fillFragments(){
		
		while(this.findFirstKeeper()){
			int counter	=	0;
			
			while(counter < fragmentKeepers.get(fragment).size()){
				
				this.findConnectedKeepers(fragmentKeepers.get(fragment).get(counter));
				counter++;
			}
		
			nextFragment();
		}
	}
	
	/**Go to the next fragment.
	 * 
	 */
	private void nextFragment(){
		
		this.fragment++;
	}
	
	/*
	private void resetFragment(){
		
		this.fragment	=	0;
	}*/
	
	/**Find an atom to be kept. Will pick the first encountered atom of each non-connected reaction center
	 * in the molecule.
	 * 
	 * @return has one been found?
	 */
	private boolean findFirstKeeper(){
		
		fragmentKeepers.add(new ArrayList<IAtom>());
		
		for(IAtom atom:notCheckedAtoms){
			if((boolean)atom.getProperty(StringConstants.SHOULDKEEP)
				//Replaced by keeping a list of atoms that have not yet been checked.
					/*&&
				this.notKept(atom)*/){
				fragmentKeepers.get(fragment).add(atom);
				notCheckedAtoms.remove(atom);
				return true;
			}
		}
		
		return false;
	}
	
	/**Unset the "visited" flags set by the path search algorithm.
	 * 
	 * @param container
	 */
	private void unsetVisited(IAtomContainer container){
		for(IAtom atom:container.atoms())
			atom.setFlag(CDKConstants.VISITED, false);
	}
	
	/**Find the atoms that should be kept that are connected to this atom
	 * 
	 * @param keeper
	 */
	private void findConnectedKeepers(IAtom keeper){
		
		for(IBond bond:container.getConnectedBondsList(keeper)){
			IAtom candidate	=	bond.getConnectedAtom(keeper);
			
			if((boolean)candidate.getProperty(StringConstants.SHOULDKEEP)
				&&
			   !fragmentKeepers.get(fragment).contains(candidate)){
				fragmentKeepers.get(fragment).add(candidate);
				notCheckedAtoms.remove(candidate);
			}
		}
	}
}