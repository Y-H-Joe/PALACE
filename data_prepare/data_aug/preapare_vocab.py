#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 10:51:46 2022

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

def split(x):
    return x.split(' >> ')
list_ = pd.read_csv('non_enzyme/EC_tok.for_non_enzyme.tsv',sep = '\t')['rxn'].apply(split).values.tolist()

with open('fake_sample_for_vocab.txt','w') as o:
    for x in list_:
        o.write('ID\tN.A\tAAA\t{}\t{}\n'.format(x[0],x[1]))
