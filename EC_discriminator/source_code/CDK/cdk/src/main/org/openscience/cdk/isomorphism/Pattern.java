/*
 * Copyright (c) 2013 European Bioinformatics Institute (EMBL-EBI)
 *                    John May <jwmay@users.sf.net>
 *  
 * Contact: cdk-devel@lists.sourceforge.net
 *  
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version. All we ask is that proper credit is given
 * for our work, which includes - but is not limited to - adding the above 
 * copyright notice to the beginning of your source code files, and to any
 * copyright notice that you may distribute with programs based on this work.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 U
 */

package org.openscience.cdk.isomorphism;

import org.openscience.cdk.interfaces.IAtomContainer;

/**
 * A structural pattern for finding an exact matching in a target compound.
 *
 * @author John May
 * @cdk.module isomorphism
 */
public abstract class Pattern {

    /**
     * Find a matching of this pattern in the {@code target}. If no such order
     * exist an empty mapping is returned.
     *
     * <blockquote><pre>
     * Pattern        pattern = ...; // create pattern
     * for (IAtomContainer m : ms) {
     *     int[] mapping = pattern.match(m);
     *     if (mapping.length > 0) {
     *         // found mapping!
     *     }
     * }
     * </pre></blockquote>
     *
     * @param target the container to search for the pattern in
     * @return the mapping from the pattern to the target or an empty array
     */
    public abstract int[] match(IAtomContainer target);

    /**
     * Determine if there is a mapping of this pattern in the {@code target}.
     *
     * <blockquote><pre>
     * Pattern        pattern = ...; // create pattern
     * for (IAtomContainer m : ms) {
     *     if (pattern.matches(m)) {
     *         // found mapping!
     *     }
     * }
     * </pre></blockquote>
     *
     * @param target the container to search for the pattern in
     * @return the mapping from the pattern to the target
     */
    public final boolean matches(IAtomContainer target) {
        return match(target).length > 0;
    }

    /**
     * Find all mappings of this pattern in the {@code target}.
     *
     * <blockquote><pre>
     * Pattern        pattern = ...; // create pattern
     * for (IAtomContainer m : ms) {
     *     for (int[] mapping : pattern.matchAll(m)) {
     *         // found mapping
     *     }
     * }
     * </pre></blockquote>
     *
     * Using Guava we can easily search and manipulate the mapped states. Here's
     * an example of finding the first 5 mappings and creating an array. If the
     * mapper is lazy other states are simply not explored.
     *
     * <blockquote><pre>
     * // find only the first 5 mappings
     * Pattern pattern  = ...;
     * int[][] mappings = FluentIterable.from(pattern.matchAll(target))
     *                                  .limit(5)
     *                                  .toArray(int[].class);
     * </pre></blockquote>
     *
     * @param target the container to search for the pattern in
     * @return the mapping from the pattern to the target
     */
    public abstract Iterable<int[]> matchAll(IAtomContainer target);
}
