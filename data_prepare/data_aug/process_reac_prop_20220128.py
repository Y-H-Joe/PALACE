#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Created on Fri Jan 28 15:33:16 2022

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ discription ==============================####
classifs is the EC column. For case that one row has multiple EC, will separate
to multiple rows.
also, the MNXD (compartmentalized) and coefficients will be removed, so equation
is not balanced, which makes the net more difficult to train.
=================================== input =====================================    .
['ID', 'mnx_equation', 'reference', 'classifs', 'is_balanced',
       'is_transport']
=================================== output ====================================
mnx_equation    EC
================================= parameters ==================================

=================================== example ===================================

=================================== warning ===================================

####=======================================================================####
"""
import pandas as pd
import os

dp = 'reac_prop.tsv'
output = "{}.processed".format(os.path.basename(dp))

df = pd.read_csv(dp,sep='\t')

new_lst = []

for row in df.itertuples():
    rxn = row.mnx_equation
    # one example of rxn:
    # 1 MNXM1@MNXD1 + 1 MNXM37@MNXD1 + 1 MNXM40333@MNXD1 + 1 MNXM9@MNXD1 = 1 MNXM3@MNXD1 + 1 MNXM729302@MNXD1 + 1 MNXM741173@MNXD1
    rxn_lst = [x.split('@')[0] for x in rxn.split(' ') if not x.isdigit()]
    rxn_new = ' '.join(rxn_lst)
    
    ECs = row.classifs
    EC_lst = ECs.split(';')
    
    row_new = ['\t'.join([rxn_new,x]) for x in EC_lst]
    new_lst += row_new

with open(output,'w') as o:
    o.write('mnx_equation\tEC\n')
    for row in new_lst:
        o.write(row)
        o.write('\n')
        
        