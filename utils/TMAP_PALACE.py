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
from matplotlib.colors import ListedColormap
#from mhfp.encoder import MHFPEncoder
#from rdkit.Chem import AllChem
#from rxnfp.transformer_fingerprints import (
#    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints)

df = pd.read_csv('EC_tok.train.again.primeEC_fingerprint.tsv',sep = '\t', header = None)
output = "EC_tok_train_again" # no '.' should be in name
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
cfg = tmap.LayoutConfiguration()
cfg.node_size = 1 / 26
cfg.mmm_repeats = 2
cfg.sl_extra_scaling_steps = 5
cfg.k = 20
cfg.sl_scaling_type = tmap.RelativeToAvgLength
x, y, s, t, _ = tmap.layout_from_lsh_forest(lf,cfg)

# Now plot the data
EC_cmap = ListedColormap(
        ["#000000", "#52F9FF", "#E8C61A", "#FF52F9", "#24BD24", "#BD2424","#2424BD"],
        name="EC",
    )
color_list = [int(x) if x!= "N" else 0 for x in df[0]]
faerun = Faerun(view="front", coords=False, clear_color = '#FFFFFF')
faerun.add_scatter(
    output,
    {   "x": x,
        "y": y,
        "c": color_list # color
        }, # if add labels here, can see hover avatar
    point_scale=3,
    max_point_size=10,
    colormap = [EC_cmap],
    has_legend=True,
    legend_title = ['EC types'],
    categorical=[True],
    shader = 'smoothCircle'
)

faerun.add_tree(f"{output}_tree", {"from": s, "to": t}, point_helper=output,color="#E6E4E4")

# Choose the "smiles" template to display structure on hover
faerun.plot(output)

