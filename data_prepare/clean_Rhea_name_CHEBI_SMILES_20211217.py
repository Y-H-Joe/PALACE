#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 09:59:42 2021

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ discription ==============================####
remove duplicate rows;
print out name-Smiles mismatch cases
#================================== input =====================================
25888   dihydroxyacetone phosphate      57642   C(CO)(COP([O-])(=O)[O-])=O
25888   iminosuccinate  77875   [O-]C(=O)CC(=[NH2+])C([O-])=O
25888   H+      15378   [H+]
25888   H2O     15377   [H]O[H]
#================================== output ====================================

#================================ parameters ==================================

#================================== example ===================================

#================================== warning ===================================

####=======================================================================####
"""
import pandas as pd

dp=r"Rhea_name_CHEBI_SMILES.tsv"
output=str(dp+"_filtered")

df=pd.read_csv(dp,sep='\t',header=None)

#remove duplicate rows;
df_tmp=df.drop_duplicates()
#print out name-Smiles mismatch cases
names_set=set(df_tmp[1])
mismatch_name=[]
for name in names_set:
    name2smiles=list(set(df_tmp[df_tmp[1]==name][3]))
    if len(name2smiles)!=1:
        mismatch_name.append([name,name2smiles])
mismatch_name_set=set([x[0] for x in mismatch_name])
## filtered 
filtered_names=names_set-mismatch_name_set
df_tmpp=df_tmp.loc[df_tmp[1].isin(filtered_names)]
df_tmppp=df_tmpp.drop_duplicates(subset=[1,2])
df_tmppp.index=range(df_tmppp.shape[0])
filtered_namess=list(df_tmppp[1])
these_names_should_rm=[]
for i,n in enumerate(filtered_namess):
    if filtered_namess.count(n)>1:
        these_names_should_rm.append(n)
these_names_should_rm=set(these_names_should_rm)
filtered_names=filtered_names-these_names_should_rm
df_tmpppp=df_tmppp.loc[df_tmppp[1].isin(filtered_names)]
## CHEBI or SMILES should not be -
df_tmppppp=df_tmpppp.loc[df_tmpppp[2]!='-']
df_tmpppppp=df_tmppppp.loc[df_tmppppp[3]!='-']

df_tmpppppp.to_csv(output,header=None,index=None,sep='\t')
