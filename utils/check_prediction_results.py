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
prediction_dp = r"../PALACE_predict_v1.enzyme.batch1.txt"
target_dp = r"../data/PALACE_test.enzyme.tsv_batch1"

predictions = []
targets = []
with open(prediction_dp,'r') as r1, open(target_dp,'r') as r2:
    lines1 = r1.readlines()
    lines2 = r2.readlines()
    for line1,line2 in zip(lines1,lines2):
        predictions.append(eval(line1.strip()))
        targets.append(line2.strip().split('\t')[-1])

count = 0
for target, prediction in zip(targets,predictions):
    #if target in prediction:
    if target == prediction[0]:
        count += 1

accuracy = count / len(targets)
