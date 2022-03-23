#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 11:21:05 2022

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
import torch
from modules import printer,prot_to_features,logging,try_gpu


printer("=======================PALACE: predicting...=======================")
# number of samples using per train
batch_size = 2
# time steps/window size,ref d2l 8.1 and 8.3
num_steps = 10
beam = 5
num_pred = 3
tgt_vocab = './saved/tgt_vocab.pkl'
trained_model_dir = './trained_models/'
device = try_gpu()

assert num_pred <= beam, "number of predictions should be no larger then beam size."
# reagents list
rgt_list = ['N c 1 n c 2 c ( c ( = O ) [nH] 1 ) N [C@@H] ( C N ( C = O ) c 1 c c c ','C ( = O ) N']
# products list
prd_list = ['N c 1 n c 2 c ( c ( = O ) [nH] 1 ) N [C@@H]', 'N c 1 n c 2 c ( c ( = O ) [nH] 1 ) N [C@@H]']
# proteins list
prot_list = ['MALAHSLGFPRIGRDR','MALAHSLGFPRIGRDR']
prot_feats = prot_to_features(prot_list, trained_model_dir, device, batch_size)
assert len(rgt_list) == len(prd_list) == len(prot_list),"reagents,products and proteins should have same amount"

for src, tgt in zip(zip(prot_feats,rgt_list), prd_list):
    translation, dec_attention_weight_seq = predict_PALACE(
        net, src, src_vocab, tgt_vocab, num_steps, device,num_pred, beam, True)


logging.shutdown()
