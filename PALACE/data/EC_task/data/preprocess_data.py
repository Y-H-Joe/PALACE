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
#================================== warning ===================================
####=======================================================================####
"""

data = 'task3'

if data == 'task1':
    train = 'task1/train.tsv'
    test = 'task1/test.tsv'
    with open(train,'r') as r, open('task1/PALACE_EC_task1.train.tsv','w') as o:
        for line in r.readlines()[1:]:
            ID,seq,log = line.strip().split('\t')
            if log == 'False': log = 0
            else: log = 1
            o.write('\t'.join([ID, seq, str(log)]) + '\n')
    with open(test,'r') as r, open('task1/PALACE_EC_task1.test.tsv','w') as o:
        for line in r.readlines()[1:]:
            ID,seq,log = line.strip().split('\t')
            if log == 'False': log = 0
            else: log = 1
            o.write('\t'.join([ID, seq, str(log)]) + '\n')

# manually edited task2

if data == 'task3':
    train = 'task3/train.tsv'
    test = 'task3/test.tsv'
    with open(train,'r') as r, open('task3/PALACE_EC_task3.train.1.tsv','w') as o1\
    , open('task3/PALACE_EC_task3.train.2.tsv','w') as o2\
    , open('task3/PALACE_EC_task3.train.3.tsv','w') as o3\
    , open('task3/PALACE_EC_task3.train.4.tsv','w') as o4:
        for line in r.readlines()[1:]:
            ID,seq,ec = line.strip().split('\t')
            ec_list = ec.split(',')
            for ec in ec_list:
                sub_ec = ec.split('.')
                o1.write('\t'.join([ID, seq, str(sub_ec[0])]) + '\n')
                o2.write('\t'.join([ID, seq, str(sub_ec[1])]) + '\n')
                o3.write('\t'.join([ID, seq, str(sub_ec[2])]) + '\n')
                o4.write('\t'.join([ID, seq, str(sub_ec[3])]) + '\n')
    with open(test,'r') as r, open('task3/PALACE_EC_task3.test.1.tsv','w') as o1\
    , open('task3/PALACE_EC_task3.test.2.tsv','w') as o2\
    , open('task3/PALACE_EC_task3.test.3.tsv','w') as o3\
    , open('task3/PALACE_EC_task3.test.4.tsv','w') as o4:
        for line in r.readlines()[1:]:
            ID,seq,ec = line.strip().split('\t')
            ec_list = ec.split(',')
            for ec in ec_list:
                sub_ec = ec.split('.')
                o1.write('\t'.join([ID, seq, str(sub_ec[0])]) + '\n')
                o2.write('\t'.join([ID, seq, str(sub_ec[1])]) + '\n')
                o3.write('\t'.join([ID, seq, str(sub_ec[2])]) + '\n')
                o4.write('\t'.join([ID, seq, str(sub_ec[3])]) + '\n')
