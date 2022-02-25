#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 15:02:15 2021

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ discription ==============================####

#================================== input =====================================

#================================== output ====================================

#================================ parameters ==================================

#================================== example ===================================
python3 this_script.py input output
#================================== warning ===================================

####=======================================================================####
"""
import sys
_input=sys.argv[1]
_output=sys.argv[2]
startswith='reaction_id'

with open(_input,'r') as i:
    with open(_output,'w') as o:
        while True:
            line=i.readline()
            if not line:
                break
            if not line.startswith(startswith):
                o.write(line)




