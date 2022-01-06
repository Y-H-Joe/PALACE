/* $Revision$ $Author$ $Date$
 *
 * Copyright (C) 2006-2007  Sam Adams <sea36@users.sf.net>
 *
 * Contact: cdk-devel@lists.sourceforge.net
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 */
package org.openscience.cdk.inchi;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import javax.vecmath.Point2d;
import javax.vecmath.Point3d;

import net.sf.jniinchi.INCHI_BOND_STEREO;
import net.sf.jniinchi.INCHI_BOND_TYPE;
import net.sf.jniinchi.INCHI_KEY;
import net.sf.jniinchi.INCHI_OPTION;
import net.sf.jniinchi.INCHI_PARITY;
import net.sf.jniinchi.INCHI_RADICAL;
import net.sf.jniinchi.INCHI_RET;
import net.sf.jniinchi.INCHI_STEREOTYPE;
import net.sf.jniinchi.JniInchiAtom;
import net.sf.jniinchi.JniInchiBond;
import net.sf.jniinchi.JniInchiException;
import net.sf.jniinchi.JniInchiInput;
import net.sf.jniinchi.JniInchiOutput;
import net.sf.jniinchi.JniInchiOutputKey;
import net.sf.jniinchi.JniInchiStereo0D;
import net.sf.jniinchi.JniInchiWrapper;

import org.openscience.cdk.CDKConstants;
import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.interfaces.ISingleElectron;
import org.openscience.cdk.annotations.TestClass;
import org.openscience.cdk.annotations.TestMethod;
import org.openscience.cdk.config.Isotopes;
import org.openscience.cdk.config.IsotopeFactory;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IBond;
import org.openscience.cdk.interfaces.IChemObjectBuilder;
import org.openscience.cdk.interfaces.IDoubleBondStereochemistry;
import org.openscience.cdk.interfaces.IDoubleBondStereochemistry.Conformation;
import org.openscience.cdk.interfaces.IStereoElement;
import org.openscience.cdk.interfaces.ITetrahedralChirality;
import org.openscience.cdk.interfaces.ITetrahedralChirality.Stereo;

/**
 * <p>This class generates the IUPAC International Chemical Identifier (InChI) for
 * a CDK IAtomContainer. It places calls to a JNI wrapper for the InChI C++ library.
 *
 * <p>If the atom container has 3D coordinates for all of its atoms then they
 * will be used, otherwise 2D coordinates will be used if available.
 *
 * <p><i>Spin multiplicities and some aspects of stereochemistry are not
 * currently handled completely.</i>
 *
 * <h3>Example usage</h3>
 *
 * <code>// Generate factory - throws CDKException if native code does not load</code><br>
 * <code>InChIGeneratorFactory factory = new InChIGeneratorFactory();</code><br>
 * <code>// Get InChIGenerator</code><br>
 * <code>InChIGenerator gen = factory.getInChIGenerator(container);</code><br>
 * <code></code><br>
 * <code>INCHI_RET ret = gen.getReturnStatus();</code><br>
 * <code>if (ret == INCHI_RET.WARNING) {</code><br>
 * <code>  // InChI generated, but with warning message</code><br>
 * <code>  System.out.println("InChI warning: " + gen.getMessage());</code><br>
 * <code>} else if (ret != INCHI_RET.OKAY) {</code><br>
 * <code>  // InChI generation failed</code><br>
 * <code>  throw new CDKException("InChI failed: " + ret.toString()</code><br>
 * <code>    + " [" + gen.getMessage() + "]");</code><br>
 * <code>}</code><br>
 * <code></code><br>
 * <code>String inchi = gen.getInchi();</code><br>
 * <code>String auxinfo = gen.getAuxInfo();</code><br>
 * <p><tt><b>
 * TODO: distinguish between singlet and undefined spin multiplicity<br/>
 * TODO: double bond and allene parities<br/>
 * TODO: problem recognising bond stereochemistry<br/>
 * </b></tt>
 *
 * @author Sam Adams
 *
 * @cdk.module inchi
 * @cdk.githash
 */
@TestClass("org.openscience.cdk.inchi.InChIGeneratorTest")
public class InChIGenerator {

    protected JniInchiInput input;

    protected JniInchiOutput output;

    /**
     * AtomContainer instance refers to.
     */
    protected IAtomContainer atomContainer;

    
    /**
     * <p>Constructor. Generates InChI from CDK AtomContainer.
     *
     * <p>Reads atoms, bonds etc from atom container and converts to format
     * InChI library requires, then calls the library.
     *
     * @param atomContainer      AtomContainer to generate InChI for.
     * @param ignoreAromaticBonds if aromatic bonds should be treated as bonds of type single and double
     * @throws org.openscience.cdk.exception.CDKException if there is an
     * error during InChI generation
     */
    @TestMethod("testGetInchiFromChlorineAtom,testGetInchiFromLithiumIon,testGetStandardInchiFromChlorine37Atom")
    protected InChIGenerator(IAtomContainer atomContainer, boolean ignoreAromaticBonds) throws CDKException {
        try {
            input = new JniInchiInput("");
            generateInchiFromCDKAtomContainer(atomContainer, ignoreAromaticBonds);
        } catch (JniInchiException jie) {
            throw new CDKException("InChI generation failed: " + jie.getMessage(), jie);
        }
    }

    /**
     * <p>Constructor. Generates InChI from CDK AtomContainer.
     *
     * <p>Reads atoms, bonds etc from atom container and converts to format
     * InChI library requires, then calls the library.
     *
     * @param atomContainer      AtomContainer to generate InChI for.
     * @param options   Space delimited string of options to pass to InChI library.
     *                  Each option may optionally be preceded by a command line
     *                  switch (/ or -).
     * @param ignoreAromaticBonds if aromatic bonds should be treated as bonds of type single and double
     * @throws CDKException
     */
    protected InChIGenerator(IAtomContainer atomContainer, String options, boolean ignoreAromaticBonds) throws
            CDKException {
        try {
            input = new JniInchiInput(options);
            generateInchiFromCDKAtomContainer(atomContainer, ignoreAromaticBonds);
        } catch (JniInchiException jie) {
            throw new CDKException("InChI generation failed: " + jie.getMessage(), jie);
        }
    }


    /**
     * <p>Constructor. Generates InChI from CDK AtomContainer.
     *
     * <p>Reads atoms, bonds etc from atom container and converts to format
     * InChI library requires, then calls the library.
     *
     * @param atomContainer     AtomContainer to generate InChI for.
     * @param options           List of INCHI_OPTION.
     * @param ignoreAromaticBonds if aromatic bonds should be treated as bonds of type single and double
     * @throws CDKException
     */
    protected InChIGenerator(IAtomContainer atomContainer, List<INCHI_OPTION> options,
                             boolean ignoreAromaticBonds) throws CDKException {
        try {
            input = new JniInchiInput(options);
            generateInchiFromCDKAtomContainer(atomContainer, ignoreAromaticBonds);
        } catch (JniInchiException jie) {
            throw new CDKException("InChI generation failed: " + jie.getMessage(), jie);
        }
    }


    /**
     * <p>Reads atoms, bonds etc from atom container and converts to format
     * InChI library requires, then places call for the library to generate
     * the InChI.
     *
     * @param atomContainer      AtomContainer to generate InChI for.
     * @throws CDKException
     */
    private void generateInchiFromCDKAtomContainer(IAtomContainer atomContainer, boolean ignore) throws CDKException {
        this.atomContainer = atomContainer;

        Iterator<IAtom> atoms = atomContainer.atoms().iterator();

        // Check for 3d coordinates
        boolean all3d = true;
        boolean all2d = true;
        while (atoms.hasNext()) {
            IAtom atom = (IAtom)atoms.next();
            if (atom.getPoint3d() == null) {
                all3d = false;
            }
            if (atom.getPoint2d() == null) {
                all2d = false;
            }
        }

        // Process atoms
        IsotopeFactory ifact = null;
        try {
            ifact = Isotopes.getInstance();
        } catch (Exception e) {
            // Do nothing
        }

        Map<IAtom, JniInchiAtom> atomMap = new HashMap<IAtom, JniInchiAtom>();
        atoms = atomContainer.atoms().iterator();
        while (atoms.hasNext()) {
        	IAtom atom = atoms.next();

            // Get coordinates
            // Use 3d if possible, otherwise 2d or none
            double x, y, z;
            if (all3d) {
                Point3d p = atom.getPoint3d();
                x = p.x;
                y = p.y;
                z = p.z;
            } else if (all2d) {
                Point2d p = atom.getPoint2d();
                x = p.x;
                y = p.y;
                z = 0.0;
            } else {
                x = 0.0;
                y = 0.0;
                z = 0.0;
            }

            // Chemical element symbol
            String el = atom.getSymbol();

            // Generate InChI atom
            JniInchiAtom iatom = input.addAtom(new JniInchiAtom(x, y, z, el));
            atomMap.put(atom, iatom);

            // Check if charged
            if(atom.getFormalCharge() == null){
            	atom.setFormalCharge(0);
            }
            int charge = atom.getFormalCharge();
            if (charge != 0) {
                iatom.setCharge(charge);
            }

            // Check whether isotopic
            Integer isotopeNumber = atom.getMassNumber();
            if (isotopeNumber != CDKConstants.UNSET && ifact != null) {
                IAtom isotope = atomContainer.getBuilder().newInstance(IAtom.class,el);
                ifact.configure(isotope);
                if (isotope.getMassNumber().intValue() == isotopeNumber.intValue()) {
                    isotopeNumber = 0;
                }
            }
            if (isotopeNumber != CDKConstants.UNSET) {
                iatom.setIsotopicMass(isotopeNumber);
            }

            // Check for implicit hydrogens
            // atom.getHydrogenCount() returns number of implict hydrogens, not
            // total number
            // Ref: Posting to cdk-devel list by Egon Willighagen 2005-09-17
            Integer implicitH = atom.getImplicitHydrogenCount();
            
            // set implicit hydrogen count, -1 tells the inchi to determine it 
            iatom.setImplicitH(implicitH != null ? implicitH : -1);
            
            // Check if radical
            int count = atomContainer.getConnectedSingleElectronsCount(atom);
            if (count == 0) {
                // TODO - how to check whether singlet or undefined multiplicity
            } else if (count == 1) {
                iatom.setRadical(INCHI_RADICAL.DOUBLET);
            } else if (count == 2) {
                iatom.setRadical(INCHI_RADICAL.TRIPLET);
            } else {
                throw new CDKException("Unrecognised radical type");
            }
        }


        // Process bonds
        Map<IBond, JniInchiBond> bondMap = new HashMap<IBond, JniInchiBond>();
        Iterator<IBond> bonds =  atomContainer.bonds().iterator();
        while (bonds.hasNext()) {
            IBond bond = bonds.next();

            // Assumes 2 centre bond
            JniInchiAtom at0 = (JniInchiAtom) atomMap.get(bond.getAtom(0));
            JniInchiAtom at1 = (JniInchiAtom) atomMap.get(bond.getAtom(1));

            // Get bond order
            INCHI_BOND_TYPE order;
            IBond.Order bo = bond.getOrder();
            if (!ignore && bond.getFlag(CDKConstants.ISAROMATIC)) {
            	order = INCHI_BOND_TYPE.ALTERN;
            } else if (bo == CDKConstants.BONDORDER_SINGLE) {
                order = INCHI_BOND_TYPE.SINGLE;
            } else if (bo == CDKConstants.BONDORDER_DOUBLE) {
                order = INCHI_BOND_TYPE.DOUBLE;
            } else if (bo == CDKConstants.BONDORDER_TRIPLE) {
                order = INCHI_BOND_TYPE.TRIPLE;
            } else {
                throw new CDKException("Failed to generate InChI: Unsupported bond type");
            }

            // Create InChI bond
            JniInchiBond ibond = new JniInchiBond(at0, at1, order);
            bondMap.put(bond, ibond);
            input.addBond(ibond);

            // Check for bond stereo definitions
            IBond.Stereo stereo = bond.getStereo();
            // No stereo definition
            if (stereo == IBond.Stereo.NONE) {
                ibond.setStereoDefinition(INCHI_BOND_STEREO.NONE);
            }
            // Bond ending (fat end of wedge) below the plane
            else if (stereo == IBond.Stereo.DOWN) {
                ibond.setStereoDefinition(INCHI_BOND_STEREO.SINGLE_1DOWN);
            }
            // Bond ending (fat end of wedge) above the plane
            else if (stereo == IBond.Stereo.UP) {
                ibond.setStereoDefinition(INCHI_BOND_STEREO.SINGLE_1UP);
            }
            // Bond starting (pointy end of wedge) below the plane
            else if (stereo == IBond.Stereo.DOWN_INVERTED) {
                ibond.setStereoDefinition(INCHI_BOND_STEREO.SINGLE_2DOWN);
            }
            // Bond starting (pointy end of wedge) above the plane
            else if (stereo == IBond.Stereo.UP_INVERTED) {
                ibond.setStereoDefinition(INCHI_BOND_STEREO.SINGLE_2UP);
            }
            else if (stereo == IBond.Stereo.E_OR_Z) {
                ibond.setStereoDefinition(INCHI_BOND_STEREO.DOUBLE_EITHER);
            }
            else if (stereo == IBond.Stereo.UP_OR_DOWN) {
                ibond.setStereoDefinition(INCHI_BOND_STEREO.SINGLE_1EITHER);
            }
            else if (stereo == IBond.Stereo.UP_OR_DOWN_INVERTED) {
                ibond.setStereoDefinition(INCHI_BOND_STEREO.SINGLE_2EITHER);
            }
            // Bond with undefined stereochemistry
            else if (stereo == CDKConstants.UNSET) {
                if (order == INCHI_BOND_TYPE.SINGLE) {
                    ibond.setStereoDefinition(INCHI_BOND_STEREO.SINGLE_1EITHER);
                } else if (order == INCHI_BOND_TYPE.DOUBLE) {
                    ibond.setStereoDefinition(INCHI_BOND_STEREO.DOUBLE_EITHER);
                }
            }
        }
        // Process tetrahedral stereo elements
        for (IStereoElement stereoElem : atomContainer.stereoElements()) {
        	if (stereoElem instanceof ITetrahedralChirality) {
        		ITetrahedralChirality chirality = (ITetrahedralChirality)stereoElem;
        		IAtom[] surroundingAtoms = chirality.getLigands();
        		Stereo stereoType = chirality.getStereo();
        		JniInchiAtom atC = (JniInchiAtom) atomMap.get(chirality.getChiralAtom());
                JniInchiAtom at0 = (JniInchiAtom) atomMap.get(surroundingAtoms[0]);
                JniInchiAtom at1 = (JniInchiAtom) atomMap.get(surroundingAtoms[1]);
                JniInchiAtom at2 = (JniInchiAtom) atomMap.get(surroundingAtoms[2]);
                JniInchiAtom at3 = (JniInchiAtom) atomMap.get(surroundingAtoms[3]);
                INCHI_PARITY p = INCHI_PARITY.UNKNOWN;
                if (stereoType == Stereo.ANTI_CLOCKWISE) {
                    p = INCHI_PARITY.ODD;
                } else if (stereoType == Stereo.CLOCKWISE) {
                    p = INCHI_PARITY.EVEN;
                } else {
                    throw new CDKException("Unknown tetrahedral chirality");
                }

                JniInchiStereo0D jniStereo = new JniInchiStereo0D(atC, at0, at1, at2, at3,
                        INCHI_STEREOTYPE.TETRAHEDRAL, p);
                input.addStereo0D(jniStereo);
        	} else if (stereoElem instanceof IDoubleBondStereochemistry) {
        		IDoubleBondStereochemistry dbStereo = (IDoubleBondStereochemistry)stereoElem;
        		IBond[] surroundingBonds = dbStereo.getBonds();
        		if (surroundingBonds[0] == null || surroundingBonds[1] == null)
        			throw new CDKException("Cannot generate an InChI with incomplete double bond info");
        		org.openscience.cdk.interfaces.IDoubleBondStereochemistry.
        		    Conformation stereoType = dbStereo.getStereo();
        		IBond stereoBond = dbStereo.getStereoBond();
                JniInchiAtom at0 = null;
                JniInchiAtom at1 = null;
                JniInchiAtom at2 = null;
                JniInchiAtom at3 = null;
                // TODO: I should check for two atom bonds... or maybe that should happen when you
                //    create a double bond stereochemistry
        		if (stereoBond.contains(surroundingBonds[0].getAtom(0))) {
        			// first atom is A
                    at1 = (JniInchiAtom) atomMap.get(surroundingBonds[0].getAtom(0));
                    at0 = (JniInchiAtom) atomMap.get(surroundingBonds[0].getAtom(1));
        		} else {
        			// first atom is X
                    at0 = (JniInchiAtom) atomMap.get(surroundingBonds[0].getAtom(0));
                    at1 = (JniInchiAtom) atomMap.get(surroundingBonds[0].getAtom(1));
        		}
        		if (stereoBond.contains(surroundingBonds[1].getAtom(0))) {
        			// first atom is B
                    at2 = (JniInchiAtom) atomMap.get(surroundingBonds[1].getAtom(0));
                    at3 = (JniInchiAtom) atomMap.get(surroundingBonds[1].getAtom(1));
        		} else {
        			// first atom is Y
                    at2 = (JniInchiAtom) atomMap.get(surroundingBonds[1].getAtom(1));
                    at3 = (JniInchiAtom) atomMap.get(surroundingBonds[1].getAtom(0));
        		}
                INCHI_PARITY p = INCHI_PARITY.UNKNOWN;
                if (stereoType == org.openscience.cdk.interfaces.IDoubleBondStereochemistry.Conformation.TOGETHER) {
                    p = INCHI_PARITY.ODD;
                } else if (stereoType == org.openscience.cdk.interfaces.IDoubleBondStereochemistry.Conformation.OPPOSITE) {
                    p = INCHI_PARITY.EVEN;
                } else if(stereoType == org.openscience.cdk.interfaces.IDoubleBondStereochemistry.Conformation.UNSPECIFIED){ 
                	//unspecified double bond stereo should just result in inchi without double bond stereo.
        		}else{
                    throw new CDKException("Unknown double bond stereochemistry");
                }

                JniInchiStereo0D jniStereo = new JniInchiStereo0D(
                	null, at0, at1, at2, at3, INCHI_STEREOTYPE.DOUBLEBOND, p
                );
                input.addStereo0D(jniStereo);
        	}
        }

        try {
            output = JniInchiWrapper.getInchi(input);
        } catch (JniInchiException jie) {
            throw new CDKException("Failed to generate InChI: " + jie.getMessage(), jie);
        }
    }


    /**
     * Gets return status from InChI process.  OKAY and WARNING indicate
     * InChI has been generated, in all other cases InChI generation
     * has failed.
     */
    @TestMethod("testGetInchiFromLandDAlanine3D,testGetInchiEandZ12Dichloroethene2D")
    public INCHI_RET getReturnStatus() {
        return(output.getReturnStatus());
    }

    /**
     * Gets generated InChI string.
     * For "problem radicals", i.e. radicals on a double bond, a stereo layer is added separately.
     */
    @TestMethod("testGetInchiEandZ12Dichloroethene2D,testGetInchiFromEthyne,testGetInchiFromEthene")
    public String getInchi() {
    	//Radicals on a double bond are stereogenic, however, jni-inchi fails to add a stereo layer for this type of 
    	//atoms. For these types of atoms, an additional stereo layer is manually added.
    	//Method currently assumes only one radical on a double bond.
        String InChI=output.getInchi();
        if(this.checkIfProblemRadical()){
        	InChI=this.addCorrectStereoLayer(InChI);
        }
        
        return InChI;
    }
    
    /**Adds the correct stereo layer to the existing InChI
     * 
     * @param InChI
     * @return corrected InChI
     */
    private String addCorrectStereoLayer(String InChI){
    	List<Integer> atoms=getNumbers(InChI);
    	String newInChI=InChI;
    	//Iterate over all heavy atoms in the InChI
    	for(int i=0;i<atoms.size();i++){
    		List<Integer> connected=getConnectedNumbers(InChI,i);
    		//Iterate over all connected atoms in the InChI, if the couple already has InChi
    		//go to next, otherwise, add the stereolayer for this couple and check if this results 
    		//in a valid InChI. If it does, return the InChI with the new stereo layer, otherwise
    		//return the old InChI
    		for(int j=0;j<connected.size();j++){
    			if(!alreadyHasStereo(InChI,atoms.get(i),connected.get(j))){
    				String tempInChI=addAStereoLayer(InChI,atoms.get(i),connected.get(j),this.getStereoChar());
    				try {
						if(isValidInChI(tempInChI)){
							return tempInChI;
						}
					} catch (CDKException e) {
						e.printStackTrace();
					}
    			}
    	
    		}
    	}
    	return newInChI;
    }
    
    /**Get the correct character representing the double bond stereo in the InChI
     * Trans 	(opposite) configurations are represented by a '-'
     * Cis		(together) configurations are represented by a '+'	
     * @return
     */
    private char getStereoChar(){
    	Iterable<IStereoElement> stereoElements=this.atomContainer.stereoElements();
    	List<IDoubleBondStereochemistry> stereoDouble=new ArrayList<IDoubleBondStereochemistry>();
    	char stereo='o';
    	//Only check DoubleBondStereoChemistry
    	for(IStereoElement el:stereoElements){
    		if(el instanceof IDoubleBondStereochemistry){
    			stereoDouble.add((IDoubleBondStereochemistry) el);
    		}
    	}
  
    	for(IDoubleBondStereochemistry bond:stereoDouble){
    		int sE1=this.atomContainer.getConnectedSingleElectronsCount(bond.getStereoBond().getAtom(0));
    		int sE2=this.atomContainer.getConnectedSingleElectronsCount(bond.getStereoBond().getAtom(1));
    		if((sE1+sE2)==1){
    			if(bond.getStereo().equals(Conformation.TOGETHER)){stereo='+';}
    			if(bond.getStereo().equals(Conformation.OPPOSITE)){stereo='-';}
    		}
    	}
    	return stereo;
    }
    
    /**Add a stereo layer to an existing InChI. a and b specify the numbers in the InChI between 
     * which the stereo bond is situated.  
     * This method does nothing to check the validity of the generated InChI, nor whether there
     * is actually a stereo bond between the specified atoms.
     * 
     * @param InChI
     * @param a
     * @param b
     * @param stereo
     * @return
     */
    private String addAStereoLayer(String InChI,int a, int b, char stereo){
    	String[] sections=InChI.split("/");
    	String layer=""+Math.max(a, b)+"-"+Math.min(a, b)+stereo;
    	String newInChI="";
    	int i=0;
    	boolean flag=false;
    	//find section for stereo
    	for(i=0;i<sections.length;i++){
    		if(sections[i].startsWith("b")){flag=true;break;}
    	}
    	//if a stereo layer is already present: add new section in right position
    	if(flag){
    		String[] stereoSections=sections[i].substring(1).split(",");
    		String tempLayer="";
    		int[] order=getStereoNumbers(sections[i]);
    		int k=0;
    		for(k=0;k<order.length;k++){
    			if(order[k]>Math.max(a, b)){break;}
    		}
    		for(int j=0;j<k;j++){
    			tempLayer+=stereoSections[j]+",";
    		}
    		tempLayer+=layer+",";
    		for(int j=k;j<stereoSections.length;j++){
    			tempLayer+=stereoSections[j]+",";
    		}
    		
    		if(tempLayer.endsWith(",")){
    			layer="/b"+tempLayer.substring(0, tempLayer.length()-1);
    		}
    		else{layer="/b"+tempLayer;}
    	}
    	//new stereo layer
    	else{
    		layer="/b"+layer;
    	}
    	//insert the new stereo layer: in the InChI protocol, the double bond stereo layer can be
    	//preceded by: the atomic formula (starting with C), the heavy atoms (starting with c), the
    	//hydrogens (starting with h), and the charges (starting with q and/or p).
    	int elements=1;
    	for(int j=0;j<sections.length;j++){
    		if(sections[j].startsWith("C")){elements++;}
        	if(sections[j].startsWith("c")){elements++;}
        	if(sections[j].startsWith("h")){elements++;}
        	if(sections[j].startsWith("p")){elements++;}
        	if(sections[j].startsWith("q")){elements++;}
        }
    	for(int j=0;j<elements;j++){
    		if(j==0){newInChI+=sections[j];}
    		else{newInChI+="/"+sections[j];}
    	}
    	newInChI+=layer;
    	for(int j=elements+1;j<sections.length;j++){
    		newInChI+="/"+sections[j];
    	}
    	return newInChI;
    	
    }
    /**return a list of the first number of each stereo element of the InChI stereo layer
     * This is required for determining the correct position of a new stereo layer as the layers
     * must be listed in increasing order of the first number (which is the highest of the two 
     * numbers in the stereo element: 5-3 is valid, 3-5 is not!)
     * 
     * @param stereoLayer
     * @return
     */
    private int[] getStereoNumbers(String stereoLayer){
    	String newStereoLayer="";
    	if(stereoLayer.startsWith("b")){newStereoLayer=stereoLayer.substring(1);}
    	String[] sections=newStereoLayer.split(",");
    	int[] numbers=new int[sections.length];
    	for(int i=0;i<sections.length;i++){
    		if(sections[i].charAt(1)!='-'){
    			numbers[i]=Integer.parseInt(sections[i].substring(0,2));
    		}
    		else{
    			numbers[i]=Integer.parseInt(sections[i].substring(0,1));
    		}
    	}
    	return numbers;
    }
    
    /**Checks whether the InChI stereo layer already has an element containing a and b
     * 
     * @param InChI
     * @param a
     * @param b
     * @return
     */
    private boolean alreadyHasStereo(String InChI, int a, int b){
    	String[] sections=InChI.split("/");
    	int k=0;
    	boolean flag=false;
    	for(k=0;k<sections.length;k++){
    		if(sections[k].startsWith("b")){flag=true;break;}
    	}
    	String newStereoLayer="";
    	if(flag){
    	newStereoLayer=sections[k];
    	}else{return false;}
    	
    	if(newStereoLayer.startsWith("b")){newStereoLayer=newStereoLayer.substring(1);}

    	sections=newStereoLayer.split(",");
    	int number1=0;
    	int number2=0;
    	boolean found=false;
    	for(int i=0;i<sections.length;i++){
    		if(sections[i].charAt(1)!='-'){
    			number1=Integer.parseInt(sections[i].substring(0,2));
    			number2=Integer.parseInt(sections[i].substring(3,sections[i].length()-1));
    		}
    		else{
    			number1=Integer.parseInt(sections[i].substring(0,1));
    			number2=Integer.parseInt(sections[i].substring(2,sections[i].length()-1));
    		}
    		found=(number1==a||number1==b)&&(number2==a||number2==b);
    		if(found){break;}
    	}
    	return found;
    }	
    
    /**Checks whether the given InChI is valid. For an invalid InChI, the InChIToStructure 
     * converter returns an empty atomcontainer.
     *
     * @param InChI
     * @return
     * @throws CDKException
     */
    private boolean isValidInChI(String InChI) throws CDKException{
    	IChemObjectBuilder builder=DefaultChemObjectBuilder.getInstance();
    	InChIToStructure converter=new InChIToStructure(InChI,builder);
    	IAtomContainer comp=converter.getAtomContainer();
    	return comp.getAtomCount()!=0;
    	
    }
    
    /**Returns a list of all numbers in the InChI that are connected to the index_th number.
     * 
     * @param InChI
     * @param index
     * @return
     */
    private static List<Integer> getConnectedNumbers(String InChI,int index){
    	String heavyBonds=InChI.split("/")[2];
    	List<Integer> numbers=new ArrayList<Integer>();
    	int number=getNumbers(InChI).get(index);
    	String heavyBondsNoC=heavyBonds;
    	if(heavyBonds.startsWith("c")){
    		heavyBondsNoC=heavyBonds.substring(1);
    	}
    	
    	int pos=heavyBondsNoC.indexOf(""+number);
    	if(pos<heavyBondsNoC.length()-2){
    		while(heavyBondsNoC.charAt(pos+1)!='-'&&heavyBondsNoC.charAt(pos+1)!='('&&heavyBondsNoC.charAt(pos+1)!=')'&&heavyBondsNoC.charAt(pos+1)!=','
    				&&heavyBondsNoC.charAt(pos+2)!='-'&&heavyBondsNoC.charAt(pos+2)!='('&&heavyBondsNoC.charAt(pos+2)!=')'&&heavyBondsNoC.charAt(pos+2)!=','){
    			pos=heavyBondsNoC.indexOf(""+number,pos+2);
    			if(pos>heavyBonds.length()-2){break;}
    		}
    	}
    	int posS=pos;
    	//Find connected atom in front of target
    	if(pos!=0){
    		boolean found=false;
    		int bracketLevel=0;
    		while(!found){
    		
    			posS--;
    			if(heavyBondsNoC.charAt(posS)=='('&&bracketLevel==0){
    				numbers.add(getNumberAtPosDown(heavyBondsNoC,posS-1));
    				found=true;
    			}
    			else if(heavyBondsNoC.charAt(posS)==')'){
    				bracketLevel++;
    				posS--;
    				while(bracketLevel>0){
    					if(heavyBondsNoC.charAt(posS)==')'){bracketLevel++;}
    					if(heavyBondsNoC.charAt(posS)=='('){bracketLevel--;}
    					posS--;
    				}

    				numbers.add(getNumberAtPosDown(heavyBondsNoC,posS));
    				found=true;
    				
    			}
    			else if(heavyBondsNoC.charAt(posS)=='-'){
    				numbers.add(getNumberAtPosDown(heavyBondsNoC,posS-1));
    				found=true;
    			}
    		}
    	}
    	//reset posS
    	posS=pos+1;
    	//find connected atoms behind target, if not last atom
    	if(posS<heavyBondsNoC.length()){
    		if(heavyBondsNoC.charAt(posS)=='-'){
    			numbers.add(getNumberAtPosUp(heavyBondsNoC,posS+1));
    		}
    		else{ 
    			if(heavyBondsNoC.charAt(posS)=='('){
    				boolean end=false;
    				numbers.add(getNumberAtPosUp(heavyBondsNoC,posS+1));
    				posS+=2;
    				while(!end){
    				
    					if(heavyBondsNoC.charAt(posS)!='-'&&heavyBondsNoC.charAt(posS)!='('&&heavyBondsNoC.charAt(posS)!=')'&&heavyBondsNoC.charAt(posS)!=','){
    						posS++;
    					}
    					switch(heavyBondsNoC.charAt(posS)){
    					case '-': posS++;break;
    					case ')': posS++;end=true;break;
    					case ',': posS++;numbers.add(getNumberAtPosUp(heavyBondsNoC,posS));break;
    					default: posS++;break;
    					}
    		
    				}
    				numbers.add(getNumberAtPosUp(heavyBondsNoC,posS));
    			}	
    		}
    	}
    	return numbers;
    }
    
    /**Return the number at the given posS, for a backwards search. This implies that for a double
     * digit number, the parsed string must be between posS-1 and posS instead of just posS
     * 
     * @param heavyBondsNoC
     * @param posS
     * @return
     */
    private static int getNumberAtPosDown(String heavyBondsNoC,int posS){
  
    	if(posS<1){
    		return Integer.parseInt(heavyBondsNoC.substring(posS, posS+1));
    	}
    	else if(heavyBondsNoC.charAt(posS-1)!='-'&&heavyBondsNoC.charAt(posS-1)!='('&&heavyBondsNoC.charAt(posS-1)!=')'&&heavyBondsNoC.charAt(posS-1)!=','){
			return Integer.parseInt(heavyBondsNoC.substring(posS-1, posS+1));
		}
		else{
			return Integer.parseInt(heavyBondsNoC.substring(posS, posS+1));
			
		}
    }
    
    /**Return the number at the given posS, for a forwards search. This implies that for a double
     * digit number, the parsed string must be between posS and posS+1 instead of just posS
     * 
     * @param heavyBondsNoC
     * @param posS
     * @return
     */
    private static int getNumberAtPosUp(String heavyBondsNoC,int posS){
    	if(posS>=heavyBondsNoC.length()-2){
    		return Integer.parseInt(heavyBondsNoC.substring(posS));
    	}
    	else if(heavyBondsNoC.charAt(posS+1)!='-'&&heavyBondsNoC.charAt(posS+1)!='('&&heavyBondsNoC.charAt(posS+1)!=')'&&heavyBondsNoC.charAt(posS+1)!=','){
    		return Integer.parseInt(heavyBondsNoC.substring(posS, posS+2));
		}
		else{
			return Integer.parseInt(heavyBondsNoC.substring(posS, posS+1));
			
		}
    }
    
    /**Returns a list of all the numbers in the heavy atom section of the InChI
     * 
     * @param InChI
     * @return
     */
    private static List<Integer> getNumbers(String InChI){
    	String heavyBonds=InChI.split("/")[2];
    	List<Integer> atoms=new ArrayList<Integer>();
    	String heavyBondsNoC=heavyBonds;
    	
    	if(heavyBonds.startsWith("c")){
    		heavyBondsNoC=heavyBonds.substring(1);
    	}
    	
    	String[] split1=heavyBondsNoC.split("-");
    	for(int i=0;i<split1.length;i++){
    		String[] split2=splitB(split1[i],'(');
    		for(int j=0;j<split2.length;j++){
    			String[] split3=split2[j].split(",");
    			for(int k=0;k<split3.length;k++){
    				String[] split4=splitB(split3[k],')');
    				for(int l=0;l<split4.length;l++){
    					atoms.add(Integer.parseInt(split4[l]));
    				}
    			}
    		}
    	}
    	return atoms;
    }
    
    /**Splits a string on a specified character, instead of a regular expression (in the standard
     * split method). Necessary for splits on (,),[,]
     * 
     * @param s
     * @param c
     * @return
     */
    private static String[] splitB(String s,char c){
    	List<String> fracs=new ArrayList<String>();
    	int pos=s.indexOf(c);
    	int posNew;
    	String search=s;
    	int begin=0;
    	boolean end=false;
    	if(pos==-1){
    		String[] out={s};
    		return out;
    	}
    	else{
    		while(!end){
    			fracs.add(search.substring(begin, pos));
    			search=search.substring(pos+1);
    			posNew=search.indexOf(c);
    			if(posNew==-1){
    				fracs.add(search);
    				end=true;
    			}
    			pos=posNew;
    		}
    	}
    	Object[] out= fracs.toArray();
    	String[] strings=new String[out.length];
    	for(int i=0;i<out.length;i++){
    		strings[i]=(String) out[i];
    	}
    	return strings;
    }

    /**
     * Gets generated InChIKey string.
     */
    @TestMethod("testGetInchiFromEthane")
    public String getInchiKey() throws CDKException {
        JniInchiOutputKey key;
        try {
            key = JniInchiWrapper.getInchiKey(output.getInchi());
            if (key.getReturnStatus() == INCHI_KEY.OK) {
                return key.getKey();
            } else {
                throw new CDKException("Error while creating InChIKey: " +
                                       key.getReturnStatus());
            }
        } catch (JniInchiException exception) {
            throw new CDKException("Error while creating InChIKey: " +
                                   exception.getMessage(), exception);
        }
    }

    /**
     * Gets auxillary information.
     */
    @TestMethod("testGetAuxInfo")
    public String getAuxInfo() {
        return(output.getAuxInfo());
    }

    /**
     * Gets generated (error/warning) messages.
     */
    @TestMethod("testGetMessage,testGetWarningMessage")
    public String getMessage() {
        return(output.getMessage());
    }

    /**
     * Gets generated log.
     */
    @TestMethod("testGetLog")
    public String getLog() {
        return(output.getLog());
    }
    
    private boolean checkIfProblemRadical(){
    	if(this.atomContainer.getSingleElectronCount()==0||this.atomContainer.getStereoCenterCount()==0){return false;}
    	else{
    		Iterable<ISingleElectron> radicals=this.atomContainer.singleElectrons();
    		Iterable<IStereoElement> stereos=this.atomContainer.stereoElements();
    		for(ISingleElectron SE:radicals){
    			IAtom SEAtom=SE.getAtom();
    			Iterable<IBond> atomBonds=this.atomContainer.getConnectedBondsList(SEAtom);
    			for(IStereoElement el:stereos){
    				if(el instanceof IDoubleBondStereochemistry){
    					IBond stereoBond=((IDoubleBondStereochemistry)el).getStereoBond();
    					
    					for(IBond bond:atomBonds){
    						if(bond==stereoBond){return true;}
    					}
    					
    				}
    			}
    		}
    		
    	}
    	return false;
    }
}
