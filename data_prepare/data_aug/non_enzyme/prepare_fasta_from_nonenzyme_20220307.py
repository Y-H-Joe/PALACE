#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 18:45:57 2022

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ description ==============================####

=================================== input =====================================
GFDLTVDQALLNQMETEALSYYPELAQAEVIGERVGTRAYTSDFSPFWGAVPDRAGLYVASGLGSSGLTTGPL
Q659J8	KFHIGNVVKKLGVINAKHAIRLGIELQLIRPVQS
	ATVLAQSIVNEGLKAVAAGMNPMDLKRGIDKAVIAAVEELKKLSVPCSDSKAIAQVGTISANSDETVGELIAQ
=================================== output ====================================

================================= parameters ==================================

=================================== example ===================================

=================================== warning ===================================

####=======================================================================####
"""
count = 0
with open("nonenzyme.tsv",'r') as r,open("nonenzyme.fasta",'w') as o:
    for line in r:
        prot = line.split('\t')[-1]
        title = '> nonenzyme_{}\n'.format(count)
        o.write(title)
        o.write(prot)
        count += 1

