#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:41:14 2021

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ discription ==============================####

#================================== input =====================================
UnitProtID	taxaID	RheaID	EC	reaction	CHEBI	protein	reformat_reaction	CHEBI_reacion	SMILES_reaction

#================================== output ====================================

#================================ parameters ==================================

#================================== example ===================================

#================================== warning ===================================

####=======================================================================####
"""
import pandas as pd
import re

dp_train=r'D:\temp_data\uniprot_sprot_bacteria.dat.enzyme.train_v1'
dp_compound=r'uniprot_sprot_bacteria.dat.enzyme.tsv_reaction_CHEBI.compound_list'
output=str(dp_compound+"_Rhea")

df_train=pd.read_csv(dp_train,sep='\t')
df_compound=pd.read_csv(dp_compound,header=None,sep='\t')[0].to_list()

reformat_reaction_list=[re.split(' \+ | = ',x) for x in df_train['reformat_reaction'].to_list()]
Rhea_list=[]
for compound in df_compound:
    hit=0
    for index, value in enumerate(reformat_reaction_list):
        if (compound in value) and df_train.loc[index,'RheaID']!='-':
            Rhea_list.append(df_train.loc[index,'RheaID'])
            hit=1
            break
    if hit==0:
        Rhea_list.append('-')

with open(output,'w') as o:
    for m in map('\t'.join,zip(df_compound,Rhea_list)):
        o.write(m)
        o.write('\n')
    








