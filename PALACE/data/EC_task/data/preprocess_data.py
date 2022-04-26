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

if data == 'task2':
    train = 'task2/train.tsv'
    test = 'task2/test.tsv'
    vocab = r'../../../vocab/EC/EC_vocab_task2.txt'
    with open(train,'r') as r, open('task2/PALACE_EC_task2.train.tsv','w') as o\
    , open(vocab,'r') as r1:
        labels = [x.strip() for x in r1.readlines()]
        for line in r.readlines()[1:]:
            ID,seq,ec_num = line.strip().split('\t')
            label = labels.index(ec_num)
            o.write('\t'.join([ID, seq, str(label)]) + '\n')
    with open(test,'r') as r, open('task2/PALACE_EC_task2.test.tsv','w') as o\
    , open(vocab,'r') as r1:
        labels = [x.strip() for x in r1.readlines()]
        for line in r.readlines()[1:]:
            ID,seq,ec_num = line.strip().split('\t')
            label = labels.index(ec_num)
            o.write('\t'.join([ID, seq, str(label)]) + '\n')

if data == 'task3':
    train = 'task3/train.tsv'
    test = 'task3/test.tsv'
    vocab = r'../../../vocab/EC/EC_vocab_task3.txt'
    with open(train,'r') as r, open('task3/PALACE_EC_task3.train.tsv','w') as o\
    , open(vocab,'r') as r1:
        labels = [x.strip() for x in r1.readlines()]
        for line in r.readlines()[1:]:
            ID,seq,ec = line.strip().split('\t')
            ec_list = ec.split(',')
            for ec in ec_list:
                o.write('\t'.join([ID, seq, str(labels.index(ec))]) + '\n')

    with open(test,'r') as r, open('task3/PALACE_EC_task3.test.tsv','w') as o\
    , open(vocab,'r') as r1:
        labels = [x.strip() for x in r1.readlines()]
        for line in r.readlines()[1:]:
            ID,seq,ec = line.strip().split('\t')
            ec_list = ec.split(',')
            for ec in ec_list:
                o.write('\t'.join([ID, seq, str(labels.index(ec))]) + '\n')

if data == 'task33':
    train = 'task3/train.tsv'
    test = 'task3/test.tsv'
    vocab = r'../../../vocab/EC/EC_vocab_task3.txt'
    with open(train,'r') as r, open('task3/PALACE_EC_task3.train.tsv','w') as o1\
    , open('task3/PALACE_EC_task3.train.2.tsv','w') as o2\
    , open('task3/PALACE_EC_task3.train.3.tsv','w') as o3\
    , open('task3/PALACE_EC_task3.train.4.tsv','w') as o4\
    , open(vocab,'r') as r1:
        labels = [x.strip() for x in r1.readlines()]
        for line in r.readlines()[1:]:
            ID,seq,ec = line.strip().split('\t')
            ec_list = ec.split(',')
            for ec in ec_list:
                sub_ec = ec.split('.')
                o1.write('\t'.join([ID, seq, str(labels.index(str(sub_ec[0])))]) + '\n')
                o2.write('\t'.join([ID, seq, str(labels.index(str(sub_ec[1])))]) + '\n')
                o3.write('\t'.join([ID, seq, str(labels.index(str(sub_ec[2])))]) + '\n')
                o4.write('\t'.join([ID, seq, str(labels.index(str(sub_ec[3])))]) + '\n')
    with open(test,'r') as r, open('task3/PALACE_EC_task3.test.1.tsv','w') as o1\
    , open('task3/PALACE_EC_task3.test.2.tsv','w') as o2\
    , open('task3/PALACE_EC_task3.test.3.tsv','w') as o3\
    , open('task3/PALACE_EC_task3.test.4.tsv','w') as o4\
    , open(vocab,'r') as r1:
        labels = [x.strip() for x in r1.readlines()]
        for line in r.readlines()[1:]:
            ID,seq,ec = line.strip().split('\t')
            ec_list = ec.split(',')
            for ec in ec_list:
                sub_ec = ec.split('.')
                o1.write('\t'.join([ID, seq, str(labels.index(str(sub_ec[0])))]) + '\n')
                o2.write('\t'.join([ID, seq, str(labels.index(str(sub_ec[1])))]) + '\n')
                o3.write('\t'.join([ID, seq, str(labels.index(str(sub_ec[2])))]) + '\n')
                o4.write('\t'.join([ID, seq, str(labels.index(str(sub_ec[3])))]) + '\n')
