# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 15:02:15 2021

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ discription ==============================####
https://xinhaoli74.github.io/posts/2020/05/TMAP/
https://rxn4chemistry.github.io/rxnfp/
#================================== input =====================================
#================================== output ====================================
#================================ parameters ==================================
#================================== example ===================================
#================================== warning ===================================
everytime the tmap plots are slightly different. which is normal.
####=======================================================================####
"""
import pandas as pd
import tmap
from faerun import Faerun
from mhfp.encoder import MHFPEncoder
from rdkit.Chem import AllChem
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)

url = 'https://raw.githubusercontent.com/XinhaoLi74/molds/master/clean_data/ESOL.csv'

df = pd.read_csv(url)
df.shape

# The number of permutations used by the MinHashing algorithm
perm = 512

# Initializing the MHFP encoder with 512 permutations
enc = MHFPEncoder(perm)
# Create MHFP fingerprints from SMILES
# The fingerprint vectors have to be of the tm.VectorUint data type
fingerprints = [tmap.VectorUint(enc.encode(s)) for s in df["smiles"]]
########################

model, tokenizer = get_default_model_and_tokenizer()

rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
example_rxn = "**[Fe+3]1(**)[S-2][Fe+3](**)(**)[S-2]1>O>C[C@H](CCC[C@@H](C)[C@H]1CC[C@H]2[C@H]3[C@H](C[C@H](O)[C@@]21C)[C@@]1(C)CC[C@@H](O)C[C@H]1C[C@H]3O)C(=O)[O-]>>**[Fe+3]1(**)[S-2][Fe+3](**)(**)[S-2]1>O>C[C@H](CCC[C@@H](C)[C@H]1CC[C@H]2[C@H]3[C@H](C[C@H](O)[C@@]21C)[C@@]1(C)CC[C@@H](O)C[C@H]1C[C@H]3O)C(=O)[O-]"

fp = rxnfp_generator.convert(example_rxn)
##########################
# Initialize the LSH Forest
lf = tmap.LSHForest(perm)

# Add the Fingerprints to the LSH Forest and index
lf.batch_add(fingerprints)
lf.index()

# Get the coordinates
x, y, s, t, _ = tmap.layout_from_lsh_forest(lf)


# Now plot the data
faerun = Faerun(view="front", coords=False)
faerun.add_scatter(
    "ESOL_Basic",
    {   "x": x,
        "y": y,
        "c": list(df.logSolubility.values),
        "labels": df["smiles"]},
    point_scale=5,
    colormap = ['rainbow'],
    has_legend=True,
    legend_title = ['ESOL (mol/L)'],
    categorical=[False],
    shader = 'smoothCircle'
)

faerun.add_tree("ESOL_Basic_tree", {"from": s, "to": t}, point_helper="ESOL_Basic")

# Choose the "smiles" template to display structure on hover
faerun.plot('ESOL_Basic', template="smiles", notebook_height=750)







