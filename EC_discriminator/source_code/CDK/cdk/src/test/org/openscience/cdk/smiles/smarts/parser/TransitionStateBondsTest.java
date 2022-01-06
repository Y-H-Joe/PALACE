package org.openscience.cdk.smiles.smarts.parser;

import java.util.List;

import org.junit.Assert;
import org.junit.Test;
import org.openscience.cdk.CDKTestCase;
import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.config.IsotopeFactory;
import org.openscience.cdk.config.XMLIsotopeFactory;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IBond;
import org.openscience.cdk.interfaces.IChemObjectBuilder;
import org.openscience.cdk.isomorphism.matchers.QueryAtomContainer;
import org.openscience.cdk.isomorphism.matchers.smarts.OrderQueryBond;
import org.openscience.cdk.smiles.smarts.SMARTSQueryTool;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;


/**
 * JUnit tests for transition state bonds in SMARTS.
 *
 * @author Ruben Van de Vijver
 * @cdk.module test-smarts
 */
public class TransitionStateBondsTest extends CDKTestCase {

    @Test public void testTransitionStateSingleNoneBond() throws Exception {
        QueryAtomContainer container = SMARTSParser.parse("C{C", DefaultChemObjectBuilder.getInstance());
        Assert.assertEquals(2, container.getAtomCount());
        Assert.assertEquals(1, container.getBondCount());
        org.openscience.cdk.interfaces.IBond bond = container.getBond(0);
        Assert.assertTrue(bond instanceof OrderQueryBond);
        OrderQueryBond qBond = (OrderQueryBond) bond;
        Assert.assertEquals(IBond.Order.TRANSITION_SINGLE_NONE, qBond.getOrder());
    }
    
    @Test public void testTransitionStateBond() throws Exception {
        QueryAtomContainer container = SMARTSParser.parse("C{C", DefaultChemObjectBuilder.getInstance());
        Assert.assertEquals(2, container.getAtomCount());
        Assert.assertEquals(1, container.getBondCount());
        org.openscience.cdk.interfaces.IBond bond = container.getBond(0);
        Assert.assertTrue(bond instanceof OrderQueryBond);
        OrderQueryBond qBond = (OrderQueryBond) bond;
        Assert.assertFalse(IBond.Order.TRANSITION_SINGLE_DOUBLE.equals(qBond.getOrder()));
    }
    
    @Test public void testTransitionStateSingleDoubleBond() throws Exception {
        QueryAtomContainer container = SMARTSParser.parse("C}C", DefaultChemObjectBuilder.getInstance());
        Assert.assertEquals(2, container.getAtomCount());
        Assert.assertEquals(1, container.getBondCount());
        org.openscience.cdk.interfaces.IBond bond = container.getBond(0);
        Assert.assertTrue(bond instanceof OrderQueryBond);
        OrderQueryBond qBond = (OrderQueryBond) bond;
        Assert.assertEquals(IBond.Order.TRANSITION_SINGLE_DOUBLE, qBond.getOrder());
    }
    
    @Test public void testTransitionStateDoubleTripleBond() throws Exception {
        QueryAtomContainer container = SMARTSParser.parse("C{}C", DefaultChemObjectBuilder.getInstance());
        Assert.assertEquals(2, container.getAtomCount());
        Assert.assertEquals(1, container.getBondCount());
        org.openscience.cdk.interfaces.IBond bond = container.getBond(0);
        Assert.assertTrue(bond instanceof OrderQueryBond);
        OrderQueryBond qBond = (OrderQueryBond) bond;
        Assert.assertEquals(IBond.Order.TRANSITION_DOUBLE_TRIPLE, qBond.getOrder());
    }
    

    @Test public void testHabsPattern(){
    	IChemObjectBuilder builder = DefaultChemObjectBuilder.getInstance();
		IAtomContainer mol = builder.newInstance(IAtomContainer.class);
		mol.addAtom(builder.newInstance(IAtom.class, "C"));//0
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//1
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//2
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//3
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//4
		mol.addAtom(builder.newInstance(IAtom.class, "C"));//5
		mol.addAtom(builder.newInstance(IAtom.class, "C"));//6
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//7
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//8
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//9
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//10
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//11

		mol.addBond(0, 1, IBond.Order.SINGLE);
		mol.addBond(0, 2, IBond.Order.SINGLE);
		mol.addBond(0, 3, IBond.Order.SINGLE);
		mol.addBond(0, 4, IBond.Order.TRANSITION_SINGLE_NONE);
		mol.addBond(4, 5, IBond.Order.TRANSITION_SINGLE_NONE);
		mol.addBond(5, 6, IBond.Order.SINGLE);
		mol.addBond(5, 7, IBond.Order.SINGLE);
		mol.addBond(5, 8, IBond.Order.SINGLE);
		mol.addBond(6, 9, IBond.Order.SINGLE);
		mol.addBond(6, 10, IBond.Order.SINGLE);
		mol.addBond(6, 11, IBond.Order.SINGLE);

		try {
			AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(mol);
		} catch (CDKException e) {
			e.printStackTrace();
		}
		try
		{
			XMLIsotopeFactory.getInstance(mol.getBuilder()).configureAtoms(mol);
		}
		catch(Exception exc)
		{
		}
		
		SMARTSQueryTool querytool;
		querytool = new SMARTSQueryTool("[C]{[#1]{[C]", DefaultChemObjectBuilder.getInstance());
		boolean matches = false;
		try {
			matches = querytool.matches(mol);
		} catch (CDKException e) {
			e.printStackTrace();
		}
		
		 Assert.assertEquals(true,matches);
    }
    
    @Test public void testHabsPattern2(){
    	IChemObjectBuilder builder = DefaultChemObjectBuilder.getInstance();
		IAtomContainer mol = builder.newInstance(IAtomContainer.class);
		mol.addAtom(builder.newInstance(IAtom.class, "C"));//0
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//1
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//2
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//3
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//4
		mol.addAtom(builder.newInstance(IAtom.class, "C"));//5
		mol.addAtom(builder.newInstance(IAtom.class, "C"));//6
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//7
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//8
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//9
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//10
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//11

		mol.addBond(0, 1, IBond.Order.SINGLE);
		mol.addBond(0, 2, IBond.Order.SINGLE);
		mol.addBond(0, 3, IBond.Order.SINGLE);
		mol.addBond(0, 4, IBond.Order.TRANSITION_SINGLE_NONE);
		mol.addBond(4, 5, IBond.Order.TRANSITION_SINGLE_NONE);
		mol.addBond(5, 6, IBond.Order.SINGLE);
		mol.addBond(5, 7, IBond.Order.SINGLE);
		mol.addBond(5, 8, IBond.Order.SINGLE);
		mol.addBond(6, 9, IBond.Order.SINGLE);
		mol.addBond(6, 10, IBond.Order.SINGLE);
		mol.addBond(6, 11, IBond.Order.SINGLE);

		try{
			AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(mol);
		} catch (CDKException e) {
			e.printStackTrace();
		}
		try{
			XMLIsotopeFactory.getInstance(mol.getBuilder()).configureAtoms(mol);
		}
		catch(Exception exc){}
		
		SMARTSQueryTool querytool = new SMARTSQueryTool("C{[#1]", DefaultChemObjectBuilder.getInstance());
		boolean matches = false;
		List<List<Integer>> mappings = null;
		try {
			matches = querytool.matches(mol);
			mappings = querytool.getMatchingAtoms();
		} catch (CDKException e) {e.printStackTrace();}
		
		
		Assert.assertEquals(2,mappings.size());
		Assert.assertEquals(true,matches);
    }
    
    @Test public void testHabsPattern3(){
    	IChemObjectBuilder builder = DefaultChemObjectBuilder.getInstance();
		IAtomContainer mol = builder.newInstance(IAtomContainer.class);
		mol.addAtom(builder.newInstance(IAtom.class, "C"));//0
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//1
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//2
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//3
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//4
		mol.addAtom(builder.newInstance(IAtom.class, "C"));//5
		mol.addAtom(builder.newInstance(IAtom.class, "C"));//6
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//7
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//8
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//9
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//10
		mol.addAtom(builder.newInstance(IAtom.class, "H"));//11

		mol.addBond(0, 1, IBond.Order.SINGLE);
		mol.addBond(0, 2, IBond.Order.SINGLE);
		mol.addBond(0, 3, IBond.Order.SINGLE);
		mol.addBond(0, 4, IBond.Order.TRANSITION_SINGLE_NONE);
		mol.addBond(4, 5, IBond.Order.TRANSITION_SINGLE_NONE);
		mol.addBond(5, 6, IBond.Order.SINGLE);
		mol.addBond(5, 7, IBond.Order.SINGLE);
		mol.addBond(5, 8, IBond.Order.SINGLE);
		mol.addBond(6, 9, IBond.Order.SINGLE);
		mol.addBond(6, 10, IBond.Order.SINGLE);
		mol.addBond(6, 11, IBond.Order.SINGLE);

		try{
			AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(mol);
		} catch (CDKException e) {
			e.printStackTrace();
		}
		try{
			XMLIsotopeFactory.getInstance(mol.getBuilder()).configureAtoms(mol);
		}
		catch(Exception exc){}
		
		SMARTSQueryTool querytool = new SMARTSQueryTool("C{C", DefaultChemObjectBuilder.getInstance());
		boolean matches = true;
		try {
			matches = querytool.matches(mol);
		} catch (CDKException e) {e.printStackTrace();}
		
		 Assert.assertEquals(false,matches);
    }

}
