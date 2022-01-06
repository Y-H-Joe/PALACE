#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 17:20:28 2021

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ discription ==============================####

#================================== input =====================================

#================================== output ====================================

#================================ parameters ==================================

#================================== example ===================================

#================================== warning ===================================

####=======================================================================####
"""
import pandas as pd
import sys

#retro_dp=r"retro_processed_v2.csv"
retro_dp=sys.argv[1]

retro_df=pd.read_csv(retro_dp)
output=str(retro_dp+"_prot_bins_added")

retro_tmp=[]
for i in retro_df.index:
    tmp=list(retro_df.loc[i])
    ec=eval(tmp[-1])[0]
    
    ## not considering super enzyme, such as 1, or 1.1 for now
    #ec_bin=str("uniprot_EC_bins/auniprot_trembl_bacteria.enzyme.UniProtID_EC_protein.tsv_"+ec)
    ec_bin=str("aa_"+ec)
    try:
        ec_bin_df=pd.read_csv(ec_bin,sep='\t')
        for i in ec_bin_df.index:
            tmpp=list(ec_bin_df.loc[i])
            tmppp=tmp[:-1]+tmpp
            retro_tmp.append(tmppp)
    except:
        #tmppp=tmp[:-1]+['-','-','-']
        #retro_tmp.append(tmppp)
        pass
    
output_df=pd.DataFrame(retro_tmp,columns=['reaction_id','substrate_smiles','products_smiles','UnitProtID','EC','protein'])
output_df.to_csv(output,index=None)
    
    
    





