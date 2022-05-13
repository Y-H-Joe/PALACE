#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 17:48:59 2022

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ description ==============================####
this script works for both EC_tok.tsv and PALACE_train.tsv, as long as the rxn
reagent and product are the last two columns.
=================================== input =====================================
=================================== output ====================================
================================= parameters ==================================
=================================== example ===================================
=================================== warning ===================================
####=======================================================================####
"""
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer)


#train_dp = r"data/PALACE_train.tsv"
#output1_dp = r"PALACE_train.primeEC_rxn.tsv"
#output2_dp = r"PALACE_train.primeEC_fingerprint.tsv"

train_dp = r"D:\CurrentProjects\AI+yao\metabolism\microbiomeMetabolism\data_prepare\data_aug\EC_tok.train.again.tsv"
output1_dp = r"EC_tok.train.again.primeEC_rxn.tsv"
output2_dp = r"EC_tok.train.again.primeEC_fingerprint.tsv"

file_type = "EC_tok" # EC_tok or PALACE_train

model, tokenizer = get_default_model_and_tokenizer()
rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)

if file_type == "PALACE_train":
    with open(train_dp,'r') as r, open(output1_dp,'w') as o1, open(output2_dp,'w') as o2:
        for line in r:
            tmp = line.strip().split('\t')
            rxn = '>>'.join(tmp[-2:]).replace(" ","")
            fp = rxnfp_generator.convert(rxn)
            EC = tmp[1].split('.')[0]

            o1.write(EC + '\t' + rxn + '\n')
            o2.write(EC + '\t' + str(fp) + '\n')
if file_type == "EC_tok":
    with open(train_dp,'r') as r, open(output1_dp,'w') as o1, open(output2_dp,'w') as o2:
        for line in r:
            tmp = line.strip().split('\t')
            rxn = tmp[1].replace(" ","")
            fp = rxnfp_generator.convert(rxn)
            EC = tmp[0].split('.')[0]

            o1.write(EC + '\t' + rxn + '\n')
            o2.write(EC + '\t' + str(fp) + '\n')








