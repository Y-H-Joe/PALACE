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
import pandas as pd
import exact_ec_from_uniprot as exactec
import config as cfg
import funclib
import minitools as mtool
import embedding_unirep as unirep
import numpy as np

from pandarallel import pandarallel # 导入pandaralle
pandarallel.initialize() # 初始化该这个b...并行库
# add missing '-' for ec number
def refill_ec(ec):
    if ec == '-':
        return ec
    levelArray = ec.split('.')
    if  levelArray[3]=='':
        levelArray[3] ='-'
    ec = '.'.join(levelArray)
    return ec

def specific_ecs(ecstr):
    if '-' not in ecstr or len(ecstr)<4:
        return ecstr
    ecs = ecstr.split(',')
    if len(ecs)==1:
        return ecstr

    reslist=[]

    for ec in ecs:
        recs = ecs.copy()
        recs.remove(ec)
        ecarray = np.array([x.split('.') for x in recs])

        if '-' not in ec:
            reslist +=[ec]
            continue
        linearray= ec.split('.')
        if linearray[1] == '-':
            #l1 in l1s and l2 not empty
            if (linearray[0] in  ecarray[:,0]) and (len(set(ecarray[:,0]) - set({'-'}))>0):
                continue
        if linearray[2] == '-':
            # l1, l2 in l1s l2s, l3 not empty
            if (linearray[0] in  ecarray[:,0]) and (linearray[1] in  ecarray[:,1]) and (len(set(ecarray[:,2]) - set({'-'}))>0):
                continue
        if linearray[3] == '-':
            # l1, l2, l3 in l1s l2s l3s, l4 not empty
            if (linearray[0] in  ecarray[:,0]) and (linearray[1] in  ecarray[:,1]) and (linearray[2] in  ecarray[:,2]) and (len(set(ecarray[:,3]) - set({'-'}))>0):
                continue

        reslist +=[ec]
    return ','.join(reslist)

#format ec
def format_ec(ecstr):
    ecArray= ecstr.split(',')
    ecArray=[x.strip() for x in ecArray] #strip blank
    ecArray=[refill_ec(x) for x in ecArray] #format ec to full
    ecArray = list(set(ecArray)) # remove duplicates

    return ','.join(ecArray)

exactec.run_exact_task(infile=cfg.TEMPDIR+'uniprot_sprot2018.dat.gz', outfile=cfg.DATADIR+'sprot2018.tsv')
exactec.run_exact_task(infile=cfg.TEMPDIR+'uniprot_sprot2020.dat.gz', outfile=cfg.DATADIR+'sprot2020.tsv')

#加载数据并转换时间格式
sprot2018 = pd.read_csv(cfg.DATADIR+'sprot2018.tsv', sep='\t',header=0) #读入文件
sprot2018 = mtool.convert_DF_dateTime(inputdf = sprot2018)


sprot2020 = pd.read_csv(cfg.DATADIR+'sprot2020.tsv', sep='\t',header=0) #读入文件
sprot2020 = mtool.convert_DF_dateTime(inputdf = sprot2020)

sprot2018.drop_duplicates(subset=['seq'], keep='first', inplace=True)
sprot2018.reset_index(drop=True, inplace=True)
sprot2020.drop_duplicates(subset=['seq'], keep='first', inplace=True)
sprot2020.reset_index(drop=True, inplace=True)

#sprot2018
sprot2018['ec_number'] = sprot2018.ec_number.apply(lambda x: format_ec(x))
sprot2018['ec_number'] = sprot2018.ec_number.apply(lambda x: specific_ecs(x))
sprot2018['functionCounts'] = sprot2018.ec_number.apply(lambda x: 0 if x=='-'  else len(x.split(',')))

#sprot2020
sprot2020['ec_number'] = sprot2020.ec_number.apply(lambda x: format_ec(x))
sprot2020['ec_number'] = sprot2020.ec_number.apply(lambda x: specific_ecs(x))
sprot2020['functionCounts'] = sprot2020.ec_number.apply(lambda x: 0 if x=='-'  else len(x.split(',')))


train = sprot2018.iloc[:,np.r_[0,2:8,10:12]]
test = sprot2020.iloc[:,np.r_[0,2:8,10:12]]
test =test[~test.seq.isin(train.seq)]
test.reset_index(drop=True, inplace=True)

test = test[~test.id.isin(test.merge(train, on='id', how='inner').id.values)]
test.reset_index(drop=True, inplace=True)

with pd.option_context('mode.chained_assignment', None):
    train.ec_number = train.ec_number.apply(lambda x : str(x).strip()) #ec trim
    train.seq = train.seq.apply(lambda x : str(x).strip()) #seq trim

    test.ec_number = test.ec_number.apply(lambda x : str(x).strip()) #ec trim
    test.seq = test.seq.apply(lambda x : str(x).strip()) #seq trim

train.to_csv(cfg.DATADIR + 'train.tsv',sep = '\t',index = None)
test.to_csv(cfg.DATADIR + 'test.tsv',sep = '\t',index = None)


# Task 1 isEnzyme
task1_train = train.iloc[:,np.r_[0,7,1]]
task1_test = test.iloc[:,np.r_[0,7,1]]
task1_train.to_csv(cfg.DATADIR + 'task1/train.tsv',sep = '\t',index = None)
task1_test.to_csv(cfg.DATADIR + 'task1/test.tsv',sep = '\t',index = None)

# Task2 Function Counts
task2_train = train[train.functionCounts >0]
task2_train.reset_index(drop=True, inplace=True)
task2_train = task2_train.iloc[:,np.r_[0,7,3]]

task2_test = test[test.functionCounts >0]
task2_test.reset_index(drop=True, inplace=True)
task2_test = task2_test.iloc[:,np.r_[0,7,3]]

task2_train.to_csv(cfg.DATADIR + 'task2/train.tsv',sep = '\t',index = None)
task2_test.to_csv(cfg.DATADIR + 'task2/test.tsv',sep = '\t',index = None)

# Task3 EC Number
task3_train = train[train.functionCounts >0]
task3_train.reset_index(drop=True, inplace=True)
task3_train = task3_train.iloc[:,np.r_[0,7,4]]

task3_test = test[test.functionCounts >0]
task3_test.reset_index(drop=True, inplace=True)
task3_test = task3_test.iloc[:,np.r_[0,7,4]]

task3_train.to_csv(cfg.DATADIR + 'task3/train.tsv',sep = '\t',index = None)
task3_test.to_csv(cfg.DATADIR + 'task3/test.tsv',sep = '\t',index = None)
