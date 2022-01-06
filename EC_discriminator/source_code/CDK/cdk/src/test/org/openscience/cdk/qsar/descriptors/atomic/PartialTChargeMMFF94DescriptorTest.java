/* $RCSfile$
 * $Author$
 * $Date$
 * $Revision$
 * 
 *  Copyright (C) 2004-2007  Miguel Rojas <miguel.rojas@uni-koeln.de>
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
package org.openscience.cdk.qsar.descriptors.atomic;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.openscience.cdk.CDKConstants;
import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.config.Elements;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IBond;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IChemObjectBuilder;
import org.openscience.cdk.qsar.IAtomicDescriptor;
import org.openscience.cdk.qsar.result.DoubleResult;

/**
 * TestSuite that runs all QSAR tests.
 *
 * @cdk.module test-qsaratomic
 * @cdk.bug    1627763
 */
public class PartialTChargeMMFF94DescriptorTest extends AtomicDescriptorTest {
	
	private final double METHOD_ERROR = 0.16;
	
	private final IChemObjectBuilder builder = DefaultChemObjectBuilder.getInstance();
	
	/**
	 *  Constructor for the PartialTChargeMMFF94DescriptorTest object
	 *
	 */
	public  PartialTChargeMMFF94DescriptorTest() {}
	
    @Before
    public void setUp() throws Exception {
    	setDescriptor(PartialTChargeMMFF94Descriptor.class);
    }
    
	/**
	 *  A unit test suite for JUnit
	 *
	 *@return    The test suite
	 */
	/**
	 *  A unit test for JUnit with Methanol
	 */
	@Test
    public void testPartialTotalChargeDescriptor_Methanol() throws ClassNotFoundException, CDKException, java.lang.Exception {
		double [] testResult={0.28,-0.67,0.0,0.0,0.0,0.4};/* from Merck Molecular Force Field. II. Thomas A. Halgren*/
		IAtomicDescriptor descriptor = new PartialTChargeMMFF94Descriptor();
        
		IAtomContainer mol = builder.newInstance(IAtomContainer.class);
		IAtom carbon = builder.newInstance(IAtom.class,Elements.CARBON);
		IAtom oxygen = builder.newInstance(IAtom.class,Elements.OXYGEN);
		// making sure the order matches the test results
		mol.addAtom(carbon); 
		mol.addAtom(oxygen);
		mol.addBond(builder.newInstance(IBond.class,carbon, oxygen, CDKConstants.BONDORDER_SINGLE));
		addExplicitHydrogens(mol);

		for (int i = 0 ; i < mol.getAtomCount() ; i++){
			double result= ((DoubleResult)descriptor.calculate(mol.getAtom(i),mol).getValue()).doubleValue();
			Assert.assertEquals(testResult[i],result,METHOD_ERROR);
		}
	}
	/**
	 *  A unit test for JUnit with Methylamine
	 */
	@Test
    public void testPartialTotalChargeDescriptor_Methylamine() throws ClassNotFoundException, CDKException, java.lang.Exception {
		double [] testResult={0.27,-0.99,0.0,0.0,0.0,0.36};/* from Merck Molecular Force Field. II. Thomas A. Halgren*/
		IAtomicDescriptor descriptor = new PartialTChargeMMFF94Descriptor();
        
		IAtomContainer mol = builder.newInstance(IAtomContainer.class);
		IAtom carbon = builder.newInstance(IAtom.class,Elements.CARBON);
		IAtom nitrogen = builder.newInstance(IAtom.class,Elements.NITROGEN);
		// making sure the order matches the test results
		mol.addAtom(carbon); 
		mol.addAtom(nitrogen);
		mol.addBond(builder.newInstance(IBond.class,carbon, nitrogen, CDKConstants.BONDORDER_SINGLE));
		addExplicitHydrogens(mol);

		for (int i = 0 ; i < 6 ; i++){
			double result= ((DoubleResult)descriptor.calculate(mol.getAtom(i),mol).getValue()).doubleValue();
			Assert.assertEquals(testResult[i],result,METHOD_ERROR);
		}
	}
	/**
	 *  A unit test for JUnit with ethoxyethane
	 */
	@Test
    public void testPartialTotalChargeDescriptor_Ethoxyethane() throws ClassNotFoundException, CDKException, java.lang.Exception {
		double [] testResult={0.28,-0.56,0.28,};/* from Merck Molecular Force Field. II. Thomas A. Halgren*/
		IAtomicDescriptor descriptor = new PartialTChargeMMFF94Descriptor();
        
//		IAtomContainer mol = sp.parseSmiles("COC");
		IAtomContainer mol = builder.newInstance(IAtomContainer.class);
		IAtom carbon = builder.newInstance(IAtom.class,Elements.CARBON);
		IAtom oxygen = builder.newInstance(IAtom.class,Elements.OXYGEN);
		IAtom carbon2 = builder.newInstance(IAtom.class,Elements.CARBON);
		// making sure the order matches the test results
		mol.addAtom(carbon); 
		mol.addAtom(oxygen);
		mol.addAtom(carbon2); 
		mol.addBond(builder.newInstance(IBond.class,carbon, oxygen, CDKConstants.BONDORDER_SINGLE));
		mol.addBond(builder.newInstance(IBond.class,carbon2, oxygen, CDKConstants.BONDORDER_SINGLE));
		addExplicitHydrogens(mol);
		
		for (int i = 0 ; i < 3 ; i++){
			double result= ((DoubleResult)descriptor.calculate(mol.getAtom(i),mol).getValue()).doubleValue();
			Assert.assertEquals(testResult[i],result,METHOD_ERROR);
		}
	}
	/**
	 *  A unit test for JUnit with Methanethiol
	 */
	@Test
    public void testPartialTotalChargeDescriptor_Methanethiol() throws ClassNotFoundException, CDKException, java.lang.Exception {
		double [] testResult={0.23,-0.41,0.0,};/* from Merck Molecular Force Field. II. Thomas A. Halgren*/
		IAtomicDescriptor descriptor = new PartialTChargeMMFF94Descriptor();
        
		IAtomContainer mol = builder.newInstance(IAtomContainer.class);
		IAtom carbon = builder.newInstance(IAtom.class,Elements.CARBON);
		IAtom sulfur = builder.newInstance(IAtom.class,Elements.SULFUR);
		// making sure the order matches the test results
		mol.addAtom(carbon); 
		mol.addAtom(sulfur);
		mol.addBond(builder.newInstance(IBond.class,carbon, sulfur, CDKConstants.BONDORDER_SINGLE));
		addExplicitHydrogens(mol);

		for (int i = 0 ; i < 3 ; i++){
			double result= ((DoubleResult)descriptor.calculate(mol.getAtom(i),mol).getValue()).doubleValue();
			Assert.assertEquals(testResult[i],result,METHOD_ERROR);
		}
	}
	/**
	 *  A unit test for JUnit with Chloromethane
	 */
	@Test
    public void testPartialTotalChargeDescriptor_Chloromethane() throws ClassNotFoundException, CDKException, java.lang.Exception {
		double [] testResult={0.29,-0.29,0.0};/* from Merck Molecular Force Field. II. Thomas A. Halgren*/
		IAtomicDescriptor descriptor = new PartialTChargeMMFF94Descriptor();
        
//		IAtomContainer mol = sp.parseSmiles("CCl");
		IAtomContainer mol = builder.newInstance(IAtomContainer.class);
		IAtom carbon = builder.newInstance(IAtom.class,Elements.CARBON);
		IAtom chlorine = builder.newInstance(IAtom.class,Elements.CHLORINE);
		// making sure the order matches the test results
		mol.addAtom(carbon); 
		mol.addAtom(chlorine);
		mol.addBond(builder.newInstance(IBond.class,carbon, chlorine, CDKConstants.BONDORDER_SINGLE));
		addExplicitHydrogens(mol);

		for (int i = 0 ; i < 3 ; i++){
			double result= ((DoubleResult)descriptor.calculate(mol.getAtom(i),mol).getValue()).doubleValue();
			Assert.assertEquals(testResult[i],result,METHOD_ERROR);
		}
	}
	/**
	 *  A unit test for JUnit with Benzene
	 */
	@Test
    public void testPartialTotalChargeDescriptor_Benzene() throws ClassNotFoundException, CDKException, java.lang.Exception {
		double [] testResult={-0.15,-0.15,-0.15,-0.15,-0.15,-0.15,0.15,0.15,0.15,0.15,0.15,0.15};/* from Merck Molecular Force Field. II. Thomas A. Halgren*/
		IAtomicDescriptor descriptor = new PartialTChargeMMFF94Descriptor();
        
//		IAtomContainer mol = sp.parseSmiles("c1ccccc1");
		IAtomContainer mol = builder.newInstance(IAtomContainer.class);
		for (int i=0; i<6; i++) {
			IAtom carbon = builder.newInstance(IAtom.class,Elements.CARBON);
			carbon.setFlag(CDKConstants.ISAROMATIC, true);
			// making sure the order matches the test results
			mol.addAtom(carbon);			
		}
		IBond ringBond = builder.newInstance(IBond.class,mol.getAtom(0), mol.getAtom(1), CDKConstants.BONDORDER_DOUBLE);
		ringBond.setFlag(CDKConstants.ISAROMATIC, true);
		mol.addBond(ringBond);
		ringBond = builder.newInstance(IBond.class,mol.getAtom(1), mol.getAtom(2), CDKConstants.BONDORDER_SINGLE);
		ringBond.setFlag(CDKConstants.ISAROMATIC, true);
		mol.addBond(ringBond);
		ringBond = builder.newInstance(IBond.class,mol.getAtom(2), mol.getAtom(3), CDKConstants.BONDORDER_DOUBLE);
		ringBond.setFlag(CDKConstants.ISAROMATIC, true);
		mol.addBond(ringBond);
		ringBond = builder.newInstance(IBond.class,mol.getAtom(3), mol.getAtom(4), CDKConstants.BONDORDER_SINGLE);
		ringBond.setFlag(CDKConstants.ISAROMATIC, true);
		mol.addBond(ringBond);
		ringBond = builder.newInstance(IBond.class,mol.getAtom(4), mol.getAtom(5), CDKConstants.BONDORDER_DOUBLE);
		ringBond.setFlag(CDKConstants.ISAROMATIC, true);
		mol.addBond(ringBond);
		ringBond = builder.newInstance(IBond.class,mol.getAtom(5), mol.getAtom(0), CDKConstants.BONDORDER_SINGLE);
		ringBond.setFlag(CDKConstants.ISAROMATIC, true);
		mol.addBond(ringBond);
		addExplicitHydrogens(mol);
		
		for (int i = 0 ; i < 12 ; i++){
			double result= ((DoubleResult)descriptor.calculate(mol.getAtom(i),mol).getValue()).doubleValue();
			Assert.assertEquals(testResult[i],result,METHOD_ERROR);
		}
	}
	/**
	 *  A unit test for JUnit with Water
	 */
	@Test
    public void testPartialTotalChargeDescriptor_Water() throws ClassNotFoundException, CDKException, java.lang.Exception {
		double [] testResult={-0.86,0.43,0.43};/* from Merck Molecular Force Field. II. Thomas A. Halgren*/
		IAtomicDescriptor descriptor = new PartialTChargeMMFF94Descriptor();
        
		IAtomContainer mol = builder.newInstance(IAtomContainer.class);
		IAtom oxygen = builder.newInstance(IAtom.class,Elements.OXYGEN);
		// making sure the order matches the test results
		mol.addAtom(oxygen);
		addExplicitHydrogens(mol);

		for (int i = 0 ; i < 3 ; i++){
			double result= ((DoubleResult)descriptor.calculate(mol.getAtom(i),mol).getValue()).doubleValue();
			Assert.assertEquals(testResult[i],result,METHOD_ERROR);
		}
	}
}

