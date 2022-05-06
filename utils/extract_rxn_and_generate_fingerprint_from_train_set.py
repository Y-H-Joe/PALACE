#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 17:48:59 2022

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
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer)


train_dp = r"data/PALACE_train.tsv"
output1_dp = r"PALACE_train.primeEC_rxn.tsv"
output2_dp = r"PALACE_train.primeEC_fingerprint.tsv"

model, tokenizer = get_default_model_and_tokenizer()
rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)

with open(train_dp,'r') as r, open(output1_dp,'w') as o1, open(output2_dp,'w') as o2:
    for line in r:
        tmp = line.strip().split('\t')
        rxn = '>>'.join(tmp[-2:]).replace(" ","")
        fp = rxnfp_generator.convert(rxn)
        EC = tmp[1].split('.')[0]

        o1.write(EC + '\t' + rxn + '\n')
        o2.write(EC + '\t' + str(fp) + '\n')


# generate finger print