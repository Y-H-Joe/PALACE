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
# v1
accuracy_list_v1 = []
accuracy_EC_dict_v1 = {'N':'NA',1:'NA',2:'NA',3:'NA',4:'NA',5:'NA',6:'NA',7:'NA',8:'NA'}
for piece in range(833,866):
    prediction_dp = rf"../PALACE_predictions/PALACE_v1_piece_{piece}_prediction.txt"
    target_dp = r"../data/PALACE_test.sample.tsv"

    predictions = []
    targets = []
    ECs = []
    ECs_total = {'N':0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0}
    ECs_count = {'N':0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0}
    with open(prediction_dp,'r') as r1, open(target_dp,'r') as r2:
        lines1 = r1.readlines()
        lines2 = r2.readlines()
        prediction_num = len(lines1)
        for line1,line2 in zip(lines1,lines2[:prediction_num]):
            predictions.append(eval(line1.strip()))
            targets.append(line2.strip().split('\t')[-1])
            try:
                ECs.append(int(line2.strip().split('\t')[1].split('.')[0]))
            except:
                ECs.append(line2.strip().split('\t')[1].split('.')[0])

    count = 0
    for target, prediction,EC in zip(targets,predictions,ECs):
        ECs_total[EC] += 1
        #if target in prediction:
        if target == prediction[0]:
            count += 1
            ECs_count[EC] += 1

    accuracy = count / len(targets)
    accuracy_list_v1.append(accuracy)
    for key in ECs_count.keys():
        try:
            accuracy_EC_dict_v1[key] = ECs_count[key]/ECs_total[key]
        except:pass

# v3
accuracy_list_v3 = []
accuracy_EC_dict_v3 = {'N':'NA',1:'NA',2:'NA',3:'NA',4:'NA',5:'NA',6:'NA',7:'NA',8:'NA'}
for piece in range(1,41):
    prediction_dp = rf"../PALACE_predictions/PALACE_v3_piece_{piece}_prediction.txt"
    target_dp = r"../data/PALACE_test.sample.tsv"

    predictions = []
    targets = []
    ECs = []
    ECs_total = {'N':0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0}
    ECs_count = {'N':0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0}
    with open(prediction_dp,'r') as r1, open(target_dp,'r') as r2:
        lines1 = r1.readlines()
        lines2 = r2.readlines()
        prediction_num = len(lines1)
        for line1,line2 in zip(lines1,lines2[:prediction_num]):
            predictions.append(eval(line1.strip()))
            targets.append(line2.strip().split('\t')[-1])
            try:
                ECs.append(int(line2.strip().split('\t')[1].split('.')[0]))
            except:
                ECs.append(line2.strip().split('\t')[1].split('.')[0])

    count = 0
    for target, prediction,EC in zip(targets,predictions,ECs):
        ECs_total[EC] += 1
        #if target in prediction:
        if target == prediction[0]:
            count += 1
            ECs_count[EC] += 1

    accuracy = count / len(targets)
    accuracy_list_v3.append(accuracy)
    for key in ECs_count.keys():
        try:
            accuracy_EC_dict_v3[key] = ECs_count[key]/ECs_total[key]
        except:pass

# v4
accuracy_list_v4 = []
accuracy_EC_dict_v4 = {'N':'NA',1:'NA',2:'NA',3:'NA',4:'NA',5:'NA',6:'NA',7:'NA',8:'NA'}
for piece in range(1,49):
    prediction_dp = rf"../PALACE_predictions/PALACE_v4_piece_{piece}_prediction.txt"
    target_dp = r"../data/PALACE_test.sample.tsv"

    predictions = []
    targets = []
    ECs = []
    ECs_total = {'N':0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0}
    ECs_count = {'N':0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0}
    with open(prediction_dp,'r') as r1, open(target_dp,'r') as r2:
        lines1 = r1.readlines()
        lines2 = r2.readlines()
        prediction_num = len(lines1)
        for line1,line2 in zip(lines1,lines2[:prediction_num]):
            predictions.append(eval(line1.strip()))
            targets.append(line2.strip().split('\t')[-1])
            try:
                ECs.append(int(line2.strip().split('\t')[1].split('.')[0]))
            except:
                ECs.append(line2.strip().split('\t')[1].split('.')[0])

    count = 0
    for target, prediction,EC in zip(targets,predictions,ECs):
        ECs_total[EC] += 1
        #if target in prediction:
        if target == prediction[0]:
            count += 1
            ECs_count[EC] += 1

    accuracy = count / len(targets)
    accuracy_list_v4.append(accuracy)
    for key in ECs_count.keys():
        try:
            accuracy_EC_dict_v4[key] = ECs_count[key]/ECs_total[key]
        except:pass

# v5
accuracy_list_v5 = []
accuracy_EC_dict_v5 = {'N':'NA',1:'NA',2:'NA',3:'NA',4:'NA',5:'NA',6:'NA',7:'NA',8:'NA'}
for piece in range(1,48):
    prediction_dp = rf"../PALACE_predictions/PALACE_v5_piece_{piece}_prediction.txt"
    target_dp = r"../data/PALACE_test.sample.tsv"

    predictions = []
    targets = []
    ECs = []
    ECs_total = {'N':0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0}
    ECs_count = {'N':0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0}
    with open(prediction_dp,'r') as r1, open(target_dp,'r') as r2:
        lines1 = r1.readlines()
        lines2 = r2.readlines()
        prediction_num = len(lines1)
        for line1,line2 in zip(lines1,lines2[:prediction_num]):
            predictions.append(eval(line1.strip()))
            targets.append(line2.strip().split('\t')[-1])
            try:
                ECs.append(int(line2.strip().split('\t')[1].split('.')[0]))
            except:
                ECs.append(line2.strip().split('\t')[1].split('.')[0])

    count = 0
    for target, prediction,EC in zip(targets,predictions,ECs):
        ECs_total[EC] += 1
        #if target in prediction:
        if target == prediction[0]:
            count += 1
            ECs_count[EC] += 1

    accuracy = count / len(targets)
    accuracy_list_v5.append(accuracy)
    for key in ECs_count.keys():
        try:
            accuracy_EC_dict_v5[key] = ECs_count[key]/ECs_total[key]
        except:pass


# v4_again
accuracy_list_v4_again = []
accuracy_EC_dict_v4_again = {'N':'NA',1:'NA',2:'NA',3:'NA',4:'NA',5:'NA',6:'NA',7:'NA',8:'NA'}
for piece in range(1,9):
    prediction_dp = rf"../PALACE_predictions/PALACE_v4_again_piece_{piece}_prediction.txt"
    target_dp = r"../data/PALACE_test.sample.tsv"

    predictions = []
    targets = []
    ECs = []
    ECs_total = {'N':0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0}
    ECs_count = {'N':0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0}
    with open(prediction_dp,'r') as r1, open(target_dp,'r') as r2:
        lines1 = r1.readlines()
        lines2 = r2.readlines()
        prediction_num = len(lines1)
        for line1,line2 in zip(lines1,lines2[:prediction_num]):
            predictions.append(eval(line1.strip()))
            targets.append(line2.strip().split('\t')[-1])
            try:
                ECs.append(int(line2.strip().split('\t')[1].split('.')[0]))
            except:
                ECs.append(line2.strip().split('\t')[1].split('.')[0])

    count = 0
    for target, prediction,EC in zip(targets,predictions,ECs):
        ECs_total[EC] += 1
        #if target in prediction:
        if target == prediction[0]:
            count += 1
            ECs_count[EC] += 1

    accuracy = count / len(targets)
    accuracy_list_v4_again.append(accuracy)
    for key in ECs_count.keys():
        try:
            accuracy_EC_dict_v4_again[key] = ECs_count[key]/ECs_total[key]
        except:pass


