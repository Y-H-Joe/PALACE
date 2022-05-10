#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:31:43 2022

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
import pandas as pd
import numpy as np
import tmap
from faerun import Faerun
#from mhfp.encoder import MHFPEncoder
#from rdkit.Chem import AllChem
#from rxnfp.transformer_fingerprints import (
#    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints)

df = pd.read_csv('PALACE_train.again.again.primeEC_fingerprint.tsv',sep = '\t', header = None)

# The number of permutations used by the MinHashing algorithm
perm = 512

# Initializing the MHFP encoder with 512 permutations
enc = tmap.Minhash(perm)

# Create MHFP fingerprints from SMILES
# The fingerprint vectors have to be of the tm.VectorUint data type
fingerprints = [tmap.VectorFloat(np.asarray(eval(s))) for s in df[1]]

# Initialize the LSH Forest
lf = tmap.LSHForest(perm)

# Add the Fingerprints to the LSH Forest and index
lf.batch_add(enc.batch_from_weight_array(fingerprints))
lf.index()

# Get the coordinates
x, y, s, t, _ = tmap.layout_from_lsh_forest(lf)

# Now plot the data
color_list = [int(x) if x!= "N" else 0 for x in df[0]]
faerun = Faerun(view="front", coords=False, clear_color = '#FFFFFF')
faerun.add_scatter(
    "PALACE_dataset",
    {   "x": x,
        "y": y,
        "c": color_list # color
        }, # if add labels here, can see hover avatar
    point_scale=10,
    colormap = ['Set1'],
    has_legend=True,
    legend_title = ['EC types'],
    categorical=[True],
    shader = 'smoothCircle'
)

faerun.add_tree("PALACE_dataset_tree", {"from": s, "to": t}, point_helper="PALACE_dataset",color="#E6E4E4")

# Choose the "smiles" template to display structure on hover
faerun.plot('PALACE_dataset', notebook_height=750)

