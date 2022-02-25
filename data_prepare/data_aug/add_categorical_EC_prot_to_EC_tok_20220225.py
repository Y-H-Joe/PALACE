#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 10:16:39 2022

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ description ==============================####

=================================== input =====================================

=================================== output ====================================

================================= parameters ==================================

=================================== example ===================================

=================================== warning ===================================

####=======================================================================####
"""
import pandas as pd

df = pd.read_csv('EC_tok.test.tsv',sep='\t')
output = 'PALACE_test.tsv'
type_ = 'test' # test, val ,train

with open(output,'w') as o:
    for index, row in df.iterrows():
        EC = row['EC']
        rxn = row['rxn']
        reagents = rxn.split(' >> ')[0]
        products = rxn.split(' >> ')[1]
        file = "uniprot_EC_bins/uniprot_trembl_sprot_bacteria.enzyme.train_v3.UniProtID_EC_protein.tsv_{}_{}"\
                .format(EC,type_)
        try:
            with open(file,'r') as r:
                for line in r:
                    o.write(line.strip() + '\t' + reagents + '\t' + products + '\n')
        except:
            print(EC)
            
    
    









