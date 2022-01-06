/* $Revision: 10506 $ $Author: egonw $ $Date: 2008-03-22 16:10:12 +0100 (Sat, 22 Mar 2008) $
 *
 * Copyright (C) 1997-2007  The Chemistry Development Kit (CDK) project
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
package org.openscience.cdk.modulesuites;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;
import org.openscience.cdk.coverage.DictCoverageTest;
import org.openscience.cdk.dict.DictDBReactTest;
import org.openscience.cdk.dict.DictDBTest;
import org.openscience.cdk.dict.DictionaryTest;
import org.openscience.cdk.dict.EntryReactTest;
import org.openscience.cdk.dict.EntryTest;
import org.openscience.cdk.dict.OWLFileTest;
import org.openscience.cdk.dict.OWLReactTest;

/**
 * TestSuite that runs all the sample tests.
 *
 * @cdk.module  test-dict
 */
@RunWith(value=Suite.class)
@SuiteClasses(value={
    DictCoverageTest.class,
    DictDBTest.class,
    DictDBReactTest.class,
    EntryTest.class,
    EntryReactTest.class,
    DictionaryTest.class,
    OWLFileTest.class,
    OWLReactTest.class
})
public class MdictTests {}
