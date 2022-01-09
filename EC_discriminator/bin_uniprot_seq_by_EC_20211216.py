#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 17:01:16 2021

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ discription ==============================####

#================================== input =====================================
uniprot_trembl_bacteria.enzyme.UniProtID_EC_protein.tsv
#================================== output ====================================

#================================ parameters ==================================

#================================== example ===================================

#================================== warning ===================================

####=======================================================================####
"""
import pandas as pd
import sys
dp=sys.argv[1]

df=pd.read_csv(dp,sep='\t')

EC_bin=set(df['EC'])

for EC in EC_bin:
    df_bin=df[df['EC']==EC]
    output=str(dp+"_"+str(EC))
    df_bin = df_bin.drop_duplicates()
    df_bin.to_csv(output,sep='\t',index=None)
