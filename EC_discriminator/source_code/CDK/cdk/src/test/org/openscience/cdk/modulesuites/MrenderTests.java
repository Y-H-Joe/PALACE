/* Copyright (C) 2008-2010  Egon Willighagen <egonw@users.sf.net>
 *
 * Contact: cdk-devel@lists.sourceforge.net
 *
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation; either version 2.1 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 */
package org.openscience.cdk.modulesuites;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;
import org.openscience.cdk.coverage.RenderCoverageTest;
import org.openscience.cdk.renderer.RendererModelTest;
import org.openscience.cdk.renderer.color.CDK2DAtomColorsTest;
import org.openscience.cdk.renderer.color.CDKAtomColorsTest;
import org.openscience.cdk.renderer.color.CPKAtomColorsTest;
import org.openscience.cdk.renderer.color.PartialAtomicChargeColorsTest;
import org.openscience.cdk.renderer.color.RasmolColorsTest;
import org.openscience.cdk.renderer.elements.TextElementTest;
import org.openscience.cdk.renderer.generators.parameter.AbstractGeneratorParameterTest;

/**
 * TestSuite that runs all the tests for the CDK <code>render</code> module.
 *
 * @cdk.module  test-render
 */
@RunWith(Suite.class)
@SuiteClasses({
	RenderCoverageTest.class,
	CPKAtomColorsTest.class,
	RasmolColorsTest.class,
	CDKAtomColorsTest.class,
	CDK2DAtomColorsTest.class,
	PartialAtomicChargeColorsTest.class,
	RendererModelTest.class,
	AbstractGeneratorParameterTest.class,
	TextElementTest.class
})
public class MrenderTests {}
