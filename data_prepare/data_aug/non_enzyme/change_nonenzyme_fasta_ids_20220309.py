# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 15:02:15 2021

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ discription ==============================####
=================================== input =====================================
nonenzyme.tsv.id:
    P80438
    P16536
nonenzyme.16to2500.nonrd90.fasta:
> nonenzyme_1
MFSFIDDIPSFEQIKARVRDDLRKHGWEKRWNDSRLVQKSRELLNDEELKIDPATWIWKR
=================================== output ====================================
================================= parameters ==================================
=================================== example ===================================
=================================== warning ===================================
####=======================================================================####
"""

with open("nonenzyme.tsv.id",'r') as r:
    ids = r.readlines()

with open('nonenzyme.16to2500.nonrd90.fasta','r') as r,open('nonenzyme.16to2500.nonrd90.fastaa','w') as o:
    seq = ''
    for line in r:
        if '>' in line:
            if len(seq) > 0:
                o.write(seq+'\n')
                seq = ''

            ID = int(line.strip().split('_')[-1])
            real_ID = ids[ID]
            if len(real_ID) == 1:
                title = line
            else: title = '>{}'.format(real_ID)
            o.write(title)
        else:
            seq = ''.join([seq,line.strip()])
    o.write(seq+'\n')







