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
# v10
model = 291
piece = range(5)
accuracy_list = []
accuracy_EC_dict = {'N':'NA',1:'NA',2:'NA',3:'NA',4:'NA',5:'NA',6:'NA',7:'NA',8:'NA'}

for p in piece:
    prediction_dp = rf"./predictions/PALACE_v10_model_{model}_piece_{p}.txt"
    target_dp = r"./data/PALACE_test.enzyme_and_nonenzyme.shuffle.v4.tsv_{0:04}".format(p)

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

    # remove 'EC' in ECs
    targets_new, predictions_new, ECs_new = [],[],[]
    for i,v in enumerate(ECs):
        if v != 'EC':
            targets_new.append(targets[i])
            predictions_new.append(predictions[i])
            ECs_new.append(ECs[i])
    count = 0
    for target, prediction,EC in zip(targets_new, predictions_new, ECs_new):
        ECs_total[EC] += 1
        if target in prediction:
        #if target == prediction[0]:
            count += 1
            ECs_count[EC] += 1

    accuracy = count / len(targets_new)
    accuracy_list.append(accuracy)
    for key in ECs_count.keys():
        try:
            accuracy_EC_dict[key] = ECs_count[key]/ECs_total[key]
        except:pass


"""
# v1
model = 'v1_again_again'
accuracy_list_v1 = []
accuracy_EC_dict_v1 = {'N':'NA',1:'NA',2:'NA',3:'NA',4:'NA',5:'NA',6:'NA',7:'NA',8:'NA'}
for piece in range(1,27):
    prediction_dp = rf"../PALACE_predictions/PALACE_{model}_piece_{piece}_prediction.txt"
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

# v3_again_again
model = 'v3_again_again'
accuracy_list_v3 = []
accuracy_EC_dict_v3 = {'N':'NA',1:'NA',2:'NA',3:'NA',4:'NA',5:'NA',6:'NA',7:'NA',8:'NA'}
for piece in range(1,25):
    prediction_dp = rf"../PALACE_predictions/again_again_round2/PALACE_{model}_piece_{piece}_prediction.txt"
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

# v1_again_again
model = 'v1_again_again'
accuracy_list_v1 = []
accuracy_EC_dict_v1 = {'N':'NA',1:'NA',2:'NA',3:'NA',4:'NA',5:'NA',6:'NA',7:'NA',8:'NA'}
for piece in range(27,50):
    prediction_dp = rf"../PALACE_predictions/again_again_round2/PALACE_{model}_piece_{piece}_prediction.txt"
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

# v4_again_again
model = 'v4_again_again'
accuracy_list_v4 = []
accuracy_EC_dict_v4 = {'N':'NA',1:'NA',2:'NA',3:'NA',4:'NA',5:'NA',6:'NA',7:'NA',8:'NA'}
for piece in range(1,27):
    prediction_dp = rf"../PALACE_predictions/again_again_round2/PALACE_{model}_piece_{piece}_prediction.txt"
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

# v5_again_again
model = 'v5_again_again'
accuracy_list_v5 = []
accuracy_EC_dict_v5 = {'N':'NA',1:'NA',2:'NA',3:'NA',4:'NA',5:'NA',6:'NA',7:'NA',8:'NA'}
for piece in range(27,54):
    prediction_dp = rf"../PALACE_predictions/again_again_round2/PALACE_{model}_piece_{piece}_prediction.txt"
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
"""





