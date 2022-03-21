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

world_size = 1
def split_(str_):
    return str_.split(':')[1]
loss = pd.read_csv(r'../PALACE.14.loss.log',sep='\t',header=None)[2].apply(split_).\
                    astype(float).values.reshape((-1,world_size)).mean(axis=1).tolist()
accuracy = pd.read_csv(r'../PALACE.14.loss.log',sep='\t',header=None)[3].apply(split_).\
                    astype(float).values.reshape((-1,world_size)).mean(axis=1).tolist()

#折线图
x = list(range(len(loss)))#点的横坐标
y = loss #线1的纵坐标
#k2 = [0.8988,0.9334,0.9435,0.9407,0.9453,0.9453]#线2的纵坐标
plt.plot(x,y,'s-',color = 'r',label="loss")#s-:方形
#plt.plot(x,k2,'o-',color = 'g',label="CNN-RLSTM")#o-:圆形
plt.xlabel("epoch")#横坐标名字
plt.ylabel("loss")#纵坐标名字
plt.legend(loc = "best")#图例
plt.show()

#折线图
x = list(range(len(accuracy)))#点的横坐标
y = accuracy #线1的纵坐标
#k2 = [0.8988,0.9334,0.9435,0.9407,0.9453,0.9453]#线2的纵坐标
plt.plot(x,y,'s-',color = 'r',label="accuracy")#s-:方形
#plt.plot(x,k2,'o-',color = 'g',label="CNN-RLSTM")#o-:圆形
plt.xlabel("epoch")#横坐标名字
plt.ylabel("accuracy")#纵坐标名字
plt.legend(loc = "best")#图例
plt.show()

#plt.savefig("data.png")

