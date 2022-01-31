#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Created on Fri Jan 28 16:56:12 2022

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ discription ==============================####

=================================== input =====================================    
dp1:
ID	SMILES
MNXM01	[H+]
MNXM02	[O-][H]

dp2:
mnx_equation	EC
MNXM1 + MNXM37 + MNXM40333 + MNXM9 = MNXM3 + MNXM729302 + MNXM741173	6.3.1.2
=================================== output ====================================
SMILES_reaction EC
================================= parameters ==================================

=================================== example ===================================

=================================== warning ===================================

####=======================================================================####
"""
import pandas as pd
import numpy as np
import os

dp1 = 'chem_SMILES.tsv'
dp2 = 'reac_prop.tsv.processed'
output = "MetaNetX_EC_SMILES.tsv"

df1 = pd.read_csv(dp1,sep='\t',index_col=0)
df2 = pd.read_csv(dp2,sep='\t')

set_ = set() # to see how many unique SMILES
with open(output,'w') as o:
    o.write('EC\tSMILES_reaction\n')
    for row in df2.itertuples():
        ## rxn
        rxn = row.mnx_equation
        rxn_sub = [x for x in rxn.split(' = ')[0].split(' ') if x != '+']
        rxn_prd = [x for x in rxn.split(' = ')[1].split(' ') if x != '+']
        ## replace MNXM ID with SMILES
        tmp_lst1 = []
        continue_ = False
        for x in rxn_sub:
            try:
                smi = df1.loc[x,'SMILES']
                tmp_lst1.append(smi )
                set_.add(smi)
            except: 
                print(df1.loc[x,'SMILES'])
                continue_ = True
        if np.nan in tmp_lst1: continue
        rxn_sub_new = ' + '.join(tmp_lst1)
        #rxn_sub_new = ' + '.join([df1.loc[x,'SMILES'] for x in rxn_sub])
        tmp_lst2 = []
        for x in rxn_prd:
            try:
                smi = df1.loc[x,'SMILES'] 
                tmp_lst2.append(smi)
                set_.add(smi)
            except: 
                print(df1.loc[x,'SMILES'])
                continue_ = True
        if np.nan in tmp_lst2: continue
        rxn_prd_new = ' + '.join(tmp_lst2)
        #rxn_prd_new = ' + '.join([df1.loc[x,'SMILES'] for x in rxn_prd])
        if continue_ : continue
        rxn_new = ' = '.join([rxn_sub_new,rxn_prd_new])
        
        ## EC
        EC = row.EC
        
        new_row = '{}\t{}\n'.format(EC,rxn_new)
        
        o.write(new_row)
print("{} unique SMILES.".format(len(set_)))
