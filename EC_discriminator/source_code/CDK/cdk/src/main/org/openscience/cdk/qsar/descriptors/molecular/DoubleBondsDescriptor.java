package org.openscience.cdk.qsar.descriptors.molecular;

import java.util.Iterator;

import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IBond;
import org.openscience.cdk.interfaces.IChemObjectBuilder;
import org.openscience.cdk.interfaces.IRingSet;
import org.openscience.cdk.qsar.DescriptorSpecification;
import org.openscience.cdk.qsar.DescriptorValue;
import org.openscience.cdk.qsar.IMolecularDescriptor;
import org.openscience.cdk.qsar.result.DoubleResult;
import org.openscience.cdk.qsar.result.IDescriptorResult;
import org.openscience.cdk.qsar.result.IntegerResult;
import org.openscience.cdk.ringsearch.SSSRFinder;

/**
 * IMolecularDescriptor counting the number of double bonds
 * 
 * @author nmvdewie
 * @cdk.module  qsarmolecular
 * @cdk.set     qsar-descriptors
 */
public class DoubleBondsDescriptor implements IMolecularDescriptor {

	private static final String[] names = {"doubleBonds"};
	
	@Override
	public DescriptorSpecification getSpecification() {
		 return new DescriptorSpecification(
	                "http://www.blueobelisk.org/ontologies/chemoinformatics-algorithms/#DoubleBondsCount",
	                this.getClass().getName(),
	                "$Id$",
	                "Nick Vandewiele");
	}

	@Override
	public String[] getParameterNames() {
		return names;
	}

	@Override
	public Object getParameterType(String name) {
		// no parameters for this descriptor
		return null;
	}

	@Override
	public void setParameters(Object[] params) throws CDKException {
		// no parameters for this descriptor

	}

	@Override
	public Object[] getParameters() {
		// no parameters for this descriptor
		return (null);
	}

	@Override
	public String[] getDescriptorNames() {
		return names;
	}

	@Override
	public DescriptorValue calculate(IAtomContainer container) {
		Integer count = 0;
		Iterator<IBond> iter = container.bonds().iterator();
		while(iter.hasNext()){
			if(iter.next().getOrder().equals(IBond.Order.DOUBLE)){
				count++;
			}
		}
		return new DescriptorValue(getSpecification(), getParameterNames(), getParameters(),
				new IntegerResult(count), getDescriptorNames());
	}

	@Override
	public IDescriptorResult getDescriptorResultType() {
		return new IntegerResult(0);
	}

	@Override
	public void initialise(IChemObjectBuilder builder) {
		// TODO Auto-generated method stub
		
	}

}
