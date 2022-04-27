#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 14:52:19 2022

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ description ==============================####
Does USPTO dataset have the same dictionary as PALACE?
=================================== input =====================================

=================================== output ====================================

================================= parameters ==================================

=================================== example ===================================

=================================== warning ===================================

####=======================================================================####
"""
PALACE_dict_dp = r'../vocab/smi_vocab.txt'
with open(PALACE_dict_dp,'r') as r:
    PALACE_dict = [x.strip() for x in r.readlines()]

USPTO_dp = r'PALACE_USPTO_MIT_test.tsv'
with open(USPTO_dp,'r') as r:
    USPTO_dict = [y.split(' ') for x in r.readlines() for y in x.strip().split('\t') ]
    USPTO_dict = [y for x in USPTO_dict for y in x]

difference = list(set(USPTO_dict).difference(PALACE_dict))

PALACE_dict_v2 = r'../vocab/smi_vocab_v2.txt'
with open(PALACE_dict_v2,'w') as w:
    for i in PALACE_dict + difference:
        w.write(i + '\n')




