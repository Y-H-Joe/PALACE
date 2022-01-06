package org.openscience.cdk.inchi;

import junit.framework.Assert;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.openscience.cdk.CDKTestCase;
import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.interfaces.IAtomContainer;

/**
 * TestCase for the {@link InChIToStructure} class.
 *
 * @cdk.module test-inchi
 */
public class RadicalParserTest extends CDKTestCase{
	InChIGeneratorFactory fac;
	InChIToStructure i;
	@Before
	public void setUp() throws Exception {
		fac = InChIGeneratorFactory.getInstance();
	}

	@After
	public void tearDown() throws Exception {
	}

	@Test
	public void testOCCOJ() throws CDKException {
		String OCCOJ = "InChI=1S/C2H3O2/c3-1-2-4/h1-3H";
		
		i = fac.getInChIToStructure(OCCOJ,DefaultChemObjectBuilder.getInstance());
		IAtomContainer ac = i.getAtomContainer();
		
		Assert.assertEquals(1, ac.getSingleElectronCount());
	}

}
