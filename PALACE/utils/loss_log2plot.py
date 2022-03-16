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
import matplotlib.pyplot as plt

def split_(str_):
    return str_.split(':')[1]
loss = pd.read_csv(r'../PALACE.loss.18.log',sep='\t',header=None)[2].apply(split_).\
                    astype(float).values.reshape((-1,4)).mean(axis=1).tolist()

plt.bar(list(range(len(loss))), loss, label="training loss")
plt.xlabel("time points")
plt.ylabel("loss")
plt.ylim(5.111,5.112)
#plt.savefig("data.png")
