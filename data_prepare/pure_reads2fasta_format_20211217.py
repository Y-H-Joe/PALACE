#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 15:51:55 2021

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ discription ==============================####

#================================== input =====================================
read1
read2
#================================== output ====================================
>ID1
read1
>ID2
read2
#================================ parameters ==================================

#================================== example ===================================

#================================== warning ===================================

####=======================================================================####
"""
import sys
dp=sys.argv[1]
output=str(dp+".fa")

suffix='>id_'
count=1

with open(dp,'r') as r:
    with open(output,'w') as w:
        for line in r:
            w.write(str(suffix+str(count)))
            w.write('\n')
            w.write(line)
            count+=1

