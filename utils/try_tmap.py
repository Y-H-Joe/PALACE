#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 14:08:34 2022

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ description ==============================####
=================================== input =====================================
=================================== output ====================================
================================= parameters ==================================
=================================== example ===================================
=================================== warning ===================================
####=======================================================================####
"""

import tmap as tm
import numpy as np
from matplotlib import pyplot as plt


""" Main function """

n = 25
edge_list = []

# Create a random graph
for i in range(n):
    for j in np.random.randint(0, high=n, size=2):
        edge_list.append([i, j, np.random.rand(1)])

# Compute the layout
x, y, s, t, _ = tm.layout_from_edge_list(
    n, edge_list, create_mst=True
)

# Plot the edges
for i in range(len(s)):
    plt.plot(
        [x[s[i]], x[t[i]]],
        [y[s[i]], y[t[i]]],
        "k-",
        linewidth=0.5,
        alpha=0.5,
        zorder=1,
    )

# Plot the vertices
plt.scatter(x, y, zorder=2)
plt.tight_layout()
plt.savefig("simple_graph.png")




##############################################################
n = 10
edge_list = []
weights = {}

# Create a random graph
for i in range(n):
    for j in np.random.randint(0, high=n, size=2):
        # Do not add parallel edges here, to be sure
        # to have the right weight later
        if i in weights and j in weights[i] or j in weights and i in weights[j]:
            continue

        weight = np.random.rand(1)
        edge_list.append([i, j, weight])

        # Store the weights in 2d map for easy access
        if i not in weights:
            weights[i] = {}
        if j not in weights:
            weights[j] = {}

        # Invert weights to make lower ones more visible in the plot
        weights[i][j] = 1.0 - weight
        weights[j][i] = 1.0 - weight

# Compute the layout
x, y, s, t, _ = tm.layout_from_edge_list(n, edge_list, create_mst=False)
x_mst, y_mst, s_mst, t_mst, _ = tm.layout_from_edge_list(
    n, edge_list, create_mst=True
)

_, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)

# Plot graph layout with spanning tree superimposed in red
for i in range(len(s)):
    ax1.plot(
        [x[s[i]], x[t[i]]],
        [y[s[i]], y[t[i]]],
        "k-",
        linewidth=weights[s[i]][t[i]],
        alpha=0.5,
        zorder=1,
    )

for i in range(len(s_mst)):
    ax1.plot(
        [x[s_mst[i]], x[t_mst[i]]],
        [y[s_mst[i]], y[t_mst[i]]],
        "r-",
        linewidth=weights[s_mst[i]][t_mst[i]],
        alpha=0.5,
        zorder=2,
    )

ax1.scatter(x, y, zorder=3)

# Plot spanning tree layout
for i in range(len(s_mst)):
    ax2.plot(
        [x_mst[s_mst[i]], x_mst[t_mst[i]]],
        [y_mst[s_mst[i]], y_mst[t_mst[i]]],
        "r-",
        linewidth=weights[s_mst[i]][t_mst[i]],
        alpha=0.5,
        zorder=1,
    )

ax2.scatter(x_mst, y_mst, zorder=2)

plt.tight_layout()
plt.savefig("spanning_tree.png")



################################################
from timeit import default_timer as timer

import numpy as np
import tmap as tm

""" Main function """

# Use 128 permutations to create the MinHash
enc = tm.Minhash(128)
lf = tm.LSHForest(128)

d = 1000
n = 10000

data = []

# Generating some random data
start = timer()
for _ in range(n):
    data.append(tm.VectorUchar(np.random.randint(0, high=2, size=d)))
print(f"Generating the data took {(timer() - start) * 1000}ms.")

# Use batch_from_binary_array to encode the data
start = timer()
data = enc.batch_from_binary_array(data)
print(f"Encoding the data took {(timer() - start) * 1000}ms.")

# Use batch_add to parallelize the insertion of the arrays
start = timer()
lf.batch_add(data)
print(f"Adding the data took {(timer() - start) * 1000}ms.")

# Index the added data
start = timer()
lf.index()
print(f"Indexing took {(timer() - start) * 1000}ms.")

# Find the 10 nearest neighbors of the first entry
start = timer()
_ = lf.query_linear_scan_by_id(0, 10)
print(f"The kNN search took {(timer() - start) * 1000}ms.")


###########################################################
# Use 128 permutations to create the MinHash
enc = tm.Minhash(128)
lf = tm.LSHForest(128)

d = 1000
n = 10000

data = []

# Generating some random data
start = timer()
for _ in range(n):
    data.append(tm.VectorUchar(np.random.randint(0, high=2, size=d)))
print(f"Generating the data took {(timer() - start) * 1000}ms.")

# Use batch_add to parallelize the insertion of the arrays
start = timer()
lf.batch_add(enc.batch_from_binary_array(data))
print(f"Adding the data took {(timer() - start) * 1000}ms.")

# Index the added data
start = timer()
lf.index()
print(f"Indexing took {(timer() - start) * 1000}ms.")

# Construct the k-nearest neighbour graph
start = timer()
knng_from = tm.VectorUint()
knng_to = tm.VectorUint()
knng_weight = tm.VectorFloat()

_ = lf.get_knn_graph(knng_from, knng_to, knng_weight, 10)
print(f"The kNN search took {(timer() - start) * 1000}ms.")



















