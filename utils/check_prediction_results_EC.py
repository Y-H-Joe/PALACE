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
# EC task 1
accuracy_list1 = []
for piece in range(1,15):
    prediction_dp = rf"../PALACE_predictions_EC/PALACE_EC_task1_piece_{piece}_prediction.txt"
    target_dp = r"../data/PALACE_EC_task1.test.tsv"

    predictions = []
    targets = []
    with open(prediction_dp,'r') as r1, open(target_dp,'r') as r2:
        lines1 = r1.readlines()
        lines2 = r2.readlines()
        for line1,line2 in zip(lines1,lines2):
            predictions.append(int(line1.strip()))
            targets.append(int(line2.strip().split('\t')[-1]))

    count = 0
    for target, prediction in zip(targets,predictions):
        #if target in prediction:
        if target == prediction:
            count += 1

    accuracy = count / len(targets)
    accuracy_list1.append(accuracy)

# EC task 2
accuracy_list2 = []
for piece in range(0,4):
    prediction_dp = rf"../PALACE_predictions_EC/PALACE_EC_task2_piece_{piece}_prediction.txt"
    target_dp = r"../data/PALACE_EC_task2.test.tsv"

    predictions = []
    targets = []
    with open(prediction_dp,'r') as r1, open(target_dp,'r') as r2:
        lines1 = r1.readlines()
        lines2 = r2.readlines()
        for line1,line2 in zip(lines1,lines2):
            predictions.append(int(line1.strip()))
            targets.append(int(line2.strip().split('\t')[-1]))

    count = 0
    for target, prediction in zip(targets,predictions):
        #if target in prediction:
        if target == prediction:
            count += 1

    accuracy = count / len(targets)
    accuracy_list2.append(accuracy)

# EC task 3
accuracy_list3 = []
for piece in range(1,49):
    try:
        prediction_dp = rf"../PALACE_predictions_EC/PALACE_EC_task3_piece_{piece}_prediction.txt"
        target_dp = r"../data/PALACE_EC_task3.test.tsv"

        predictions = []
        targets = []
        with open(prediction_dp,'r') as r1, open(target_dp,'r') as r2:
            lines1 = r1.readlines()
            lines2 = r2.readlines()
            for line1,line2 in zip(lines1,lines2):
                predictions.append(int(line1.strip()))
                targets.append(int(line2.strip().split('\t')[-1]))

        count = 0
        for target, prediction in zip(targets,predictions):
            #if target in prediction:
            if target == prediction:
                count += 1

        accuracy = count / len(targets)
        accuracy_list3.append(accuracy)
    except:pass

# EC task 3
accuracy_list3_notransfer = []
for piece in range(1,22):
    try:
        prediction_dp = rf"../PALACE_predictions_EC/PALACE_EC_task3_notransfer_piece_{piece}_prediction.txt"
        target_dp = r"../data/PALACE_EC_task3.test.tsv"

        predictions = []
        targets = []
        with open(prediction_dp,'r') as r1, open(target_dp,'r') as r2:
            lines1 = r1.readlines()
            lines2 = r2.readlines()
            for line1,line2 in zip(lines1,lines2):
                predictions.append(int(line1.strip()))
                targets.append(int(line2.strip().split('\t')[-1]))

        count = 0
        for target, prediction in zip(targets,predictions):
            #if target in prediction:
            if target == prediction:
                count += 1

        accuracy = count / len(targets)
        accuracy_list3_notransfer.append(accuracy)
    except:pass

