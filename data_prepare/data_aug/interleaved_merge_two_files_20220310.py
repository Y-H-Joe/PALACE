#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 11:18:00 2022

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ description ==============================####
f1:
    a
    b
    c
    d
f2:
    1
    2
output:
    a
    1
    b
    2
    c
    d

=================================== input =====================================

=================================== output ====================================

================================= parameters ==================================

=================================== example ===================================

=================================== warning ===================================
####=======================================================================####
"""
from itertools import zip_longest
f1 = 'PALACE_train.nonenzyme.shuf.tsv'
f2 = 'PALACE_train.enzyme.shuf.tsv'
output = 'PALACE_train.shuf.tsv'

with open(f1,'r') as r1,open(f2,'r') as r2,open(output,'w') as o:
    for x,y in zip_longest(r1,r2):
        if x: o.write(x)
        if y: o.write(y)



