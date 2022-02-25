#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 14:49:37 2022

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ description ==============================####
With the list of EC (train, test and validation),prepare the uniprot_EC_bins.
For each EC, 90% of its protein sequences will be asigned to train, while 5% will
be asigned to test/validation. 
If some EC has not enough protein sequences to split (e.g. only 2 seqs, but train,
test, and validation all need 1, so at least 3 seqs are needed), then the seq 
as well as the corresponding rxn which has already assigned to test/val will
be asigned to train, to make full use of data.

=================================== input =====================================

=================================== output ====================================

================================= parameters ==================================

=================================== example ===================================

=================================== warning ===================================

####=======================================================================####
"""
import pandas as pd
import subprocess
import math

EC_test = list(set(pd.read_csv('EC_tok.test.tsv',sep='\t')['EC'].to_list()))
EC_train = list(set(pd.read_csv('EC_tok.train.tsv',sep='\t')['EC'].to_list()))
EC_val = list(set(pd.read_csv('EC_tok.val.tsv',sep='\t')['EC'].to_list()))

for EC in list(set(EC_test+EC_val+EC_train)):
    try:
        #print(EC)
        file = "uniprot_EC_bins/uniprot_trembl_sprot_bacteria.enzyme.train_v3.UniProtID_EC_protein.tsv_{}".format(EC)
        cmd = "wc -l {}".format(file)
        sub = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
        subprocess_return = sub.stdout
        
        seqs_num = int(subprocess_return.decode('utf-8').strip().split(' ')[0])
        #assert seqs_num >= 3
        
        take_out_num = math.ceil(seqs_num * 0.05)
        with open(file,'r') as f:
            if EC in EC_val and EC in EC_test:
                with open(file+"_train",'w') as o_train,\
                open(file+"_test",'w') as o_test,\
                open(file+"_val",'w') as o_val:
                    for index, line in enumerate(f):
                        if index < take_out_num:
                            o_test.write(line)
                        elif take_out_num <= index < take_out_num*2:
                            o_val.write(line)
                        else: o_train.write(line)
                    continue
            
            elif EC in EC_val:
                with open(file+"_train",'w') as o_train,\
                open(file+"_val",'w') as o_val:
                    for index, line in enumerate(f):
                        if index < take_out_num:
                            o_val.write(line)
                        else: o_train.write(line)
                    continue
        
            elif EC in EC_test:
                with open(file+"_train",'w') as o_train,\
                open(file+"_test",'w') as o_test:
                    for index, line in enumerate(f):
                        if index < take_out_num:
                            o_test.write(line)
                        else: o_train.write(line)
                    continue

            else :
                file_train = file+"_train"
                subprocess.run("cp {} {}".format(file,file_train), shell=True, stdout=subprocess.PIPE)
    except:
        print(EC)

















