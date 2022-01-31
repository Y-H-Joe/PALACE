#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Created on Fri Jan 28 11:43:37 2022

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ discription ==============================####

=================================== input =====================================    
UnitProtID	taxaID	RheaID	EC	reaction	CHEBI	protein	reformat_reaction	CHEBI_reacion	SMILES_reaction

=================================== output ====================================
"{}.compund".format(os.path.basename(dp))
================================= parameters ==================================

=================================== example ===================================

=================================== warning ===================================

####=======================================================================####
"""
import sys
from tqdm import tqdm
import os
#import csv

dp = sys.argv[1]
#dp = 'aa'
output = "{}.compund".format(os.path.basename(dp))

set_ = set()

with open(dp,'r') as r:
    next(r)
    for i,line in enumerate(tqdm(r)):
        CHEBI_lst = line.strip().split('\t')[5].split(';')
        for CHEBI in CHEBI_lst:
            set_.add(CHEBI)
            
print("\n{} different compounds in {}".format(len(set_),dp))
with open(output,'w') as f:
    for s in (list(set_)):
        f.write(s)
        f.write('\n')
    #csv.writer(f).writerows(list(set_))

