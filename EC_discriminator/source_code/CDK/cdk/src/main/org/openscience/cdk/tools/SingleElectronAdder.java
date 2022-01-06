package org.openscience.cdk.tools;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.tools.manipulator.BondManipulator;

/**
 * This type adds Single Electron objects to the IAtomContainer, by checking
 * the atom connectivity of the chemical species.
 * 
 * This tool is required to properly parse identifiers, such as SMILES and InChI
 * were explicit information on single electron's is not stored, but where
 * this information can be derived from the atom connectivity of the molecule.
 * 
 * The present algorithm iterates over all atoms and compares the current valency 
 * of that atom with the expected valency of that atom.
 * 
 * This tool assumes that atom types have not yet been detected, and thus does
 * not make use of any atom type perception utilities.
 * 
 * @author nmvdewie
 *
 */
public class SingleElectronAdder {
	
	Map<String, List<Integer>> expected_valencies;
	
	private static SingleElectronAdder INSTANCE;
	
	public SingleElectronAdder(){
		loadValencies();
	}
	
	public static synchronized SingleElectronAdder getInstance() {
		if(INSTANCE == null){
			INSTANCE = new SingleElectronAdder();	
		}
		return INSTANCE;
	}
	/**
	 * Map of the expected valencies for the elements.
	 * For now only one value is given, although e.g. S has more than 
	 * one expected valency (2): in sulfoxides the valency is 4 instead of 2!
	 * @return
	 */
	private Map<String, List<Integer>> loadValencies() {
		expected_valencies = new HashMap<String, List<Integer>>();
		String input_path = "org/openscience/cdk/tools/data/expected_valencies.txt";

		try {
			InputStream ins = this.getClass().getClassLoader().getResourceAsStream(input_path);
			BufferedReader in = new BufferedReader(new InputStreamReader(ins));
			while(in.ready()){
				String[] pieces = in.readLine().split("\t");
				String symbol = pieces[0];
				Integer valency = Integer.parseInt(pieces[1]);
				List<Integer> list = new ArrayList<Integer>();
				list.add(new Integer(valency));
				expected_valencies.put(symbol, list);
			}
			in.close();
			
		} catch (FileNotFoundException e) {
		} catch (IOException e) {
		}
		
		return expected_valencies;
		
	}
	/**
	 * adds Single Electron objects to the AtomContainer by comparing the
	 * expected valency of the atom with the current valency.
	 * @param container
	 * @param atom
	 * @throws CDKException 
	 */
	public void perceiveSingleElectrons(IAtomContainer container, IAtom atom) throws CDKException {
		//implicit hydrogens will not be counted in bond iterator!
		Integer iH = atom.getImplicitHydrogenCount();// returns null instead of 0
		int hCount = (iH == null)? 0 : iH ;
		
		int totaldegree = BondManipulator.getSingleBondEquivalentSum(container.getConnectedBondsList(atom));
		
		int real_valency = hCount + totaldegree;
		int expected_valency = expected_valencies.get(atom.getSymbol()).get(0);
		//add electrons equal to the difference:
		for(int i = 0; i < expected_valency - real_valency; i++){
			container.addSingleElectron(container.getAtomNumber(atom));
		}


	}
}
