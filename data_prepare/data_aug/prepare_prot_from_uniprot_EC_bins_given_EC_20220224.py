#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 14:49:37 2022

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ description ==============================####
With the list of EC (train, test and validation),prepare the uniprot_EC_bins.
For each EC, 90% of its protein sequences will be asigned to train, while 5% will
be asigned to test/validation. 
If some EC has not enough protein sequences to split (e.g. only 2 seqs, but train,
test, and validation all need 1, so at least 3 seqs are needed), then the seq 
as well as the corresponding rxn which has already assigned to test/val will
be asigned to train, to make full use of data.

=================================== input =====================================

=================================== output ====================================

================================= parameters ==================================

=================================== example ===================================

=================================== warning ===================================

####=======================================================================####
"""
import pandas as pd

EC_test = list(set(pd.read_csv('EC_tok.test.tsv',sep='\t')['EC'].to_list()))
EC_train = list(set(pd.read_csv('EC_tok.train.tsv',sep='\t')['EC'].to_list()))
EC_val = list(set(pd.read_csv('EC_tok.val.tsv',sep='\t')['EC'].to_list()))



