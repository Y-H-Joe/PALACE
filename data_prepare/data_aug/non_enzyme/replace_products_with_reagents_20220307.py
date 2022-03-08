#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:05:06 2022

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ description ==============================####
turn A > B >> C > D
into A > B >> A > B
because non_enzyme will not catalyze rxns.
=================================== input =====================================

=================================== output ====================================

================================= parameters ==================================

=================================== example ===================================

=================================== warning ===================================

####=======================================================================####
"""
input = ['EC_tok.for_non_enzyme.test.tsv','EC_tok.for_non_enzyme.train.tsv','EC_tok.for_non_enzyme.val.tsv']
output = ['tok.for_non_enzyme.test.tsv','tok.for_non_enzyme.train.tsv','tok.for_non_enzyme.val.tsv']
for idx,value in enumerate(input):
    with open(value,'r') as r,open(output[idx],'w') as o:
        next(r)
        for line in r:
            reagents = line.strip().split('\t')[-1].split(' >> ')[0]
            o.write('{} >> {}\n'.format(reagents,reagents))







