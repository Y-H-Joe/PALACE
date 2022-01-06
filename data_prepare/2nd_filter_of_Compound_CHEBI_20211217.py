#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 11:20:27 2021

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ discription ==============================####
compare Rhea_name_CHEBI_SMILES.tsv_filtered and 
uniprot_sprot_bacteria.dat.enzyme.tsv_reaction_CHEBI.compound_CHEBI.prepared, 
too get comprehensive compoun_CHEBI pairs
#================================== input =====================================

#================================== output ====================================

#================================ parameters ==================================

#================================== example ===================================

#================================== warning ===================================

####=======================================================================####
"""
import pandas as pd

dp1=r"Rhea_name_CHEBI_SMILES.tsv_filtered"
dp2=r"uniprot_sprot_and_trembl_bacteria.dat.enzyme.tsv_reaction_CHEBI.compound_CHEBI.prepared"
output=str(dp2+"_v2")

df1=pd.read_csv(dp1,header=None,sep='\t')
df2=pd.read_csv(dp2,header=None,sep='\t')

names_set1=set(df1[1])
names_set2=set(df2[0])

names_set12=names_set1 | names_set2
names_set2_1=names_set2 - names_set1

#check
df2_tmp=df2.loc[df2[0].isin(names_set2_1)] # basically, everything looks fine
df2_tmp.columns=[1,2]
df1_tmp=df1[[1,2]]

df_12=pd.concat([df1_tmp,df2_tmp],axis=0)
df_12.to_csv(output,sep='\t',index=None,header=None)
