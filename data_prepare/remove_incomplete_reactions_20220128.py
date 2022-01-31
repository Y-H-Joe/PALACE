#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Created on Fri Jan 28 09:52:43 2022

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ discription ==============================####
remove incomplete reaction ('-' in substrates/products)
=================================== input =====================================    
UnitProtID	taxaID	RheaID	EC	reaction	CHEBI	protein	reformat_reaction	CHEBI_reacion	SMILES_reaction
{}.complete_rxn".format(os.path.basename(dp))
================================= parameters ==================================

=================================== example ===================================

=================================== warning ===================================

####=======================================================================####
"""
import gzip
import os
import sys
from tqdm import tqdm

dp = sys.argv[1]
#dp = 'aa.gz'
output = "{}.complete_rxn".format(os.path.basename(dp))

with gzip.open(dp,'rb') as r:
    with open(output,'w') as o:
        o.write('UnitProtID\ttaxaID\tRheaID\tEC\treaction\tCHEBI\tprotein\treformat_reaction\tCHEBI_reacion\tSMILES_reaction\r\n')
        next(r)
        for i, line in enumerate(tqdm(r,unit="b",unit_scale=True)):
            line_split = line.strip().split(b'\t')
            SMILES_reaction = line_split[9].decode("utf-8")
            SMILES_list = SMILES_reaction.split(' ')
            if '-' not in SMILES_list:
                o.write(line.decode("utf-8"))
            
