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

loss = pd.read_csv(r'D:\Projects\AI+yao\metabolism\microbiomeMetabolism\PALACE\PALACE_v19_large.loss_accu.log',sep='\t',header=None)[2].apply(split_).\
                    astype(float).values.reshape((-1,world_size)).mean(axis=1).tolist()
accuracy = pd.read_csv(r'D:\Projects\AI+yao\metabolism\microbiomeMetabolism\PALACE\PALACE_v19_large.loss_accu.log',sep='\t',header=None)[3].apply(split_).\
                    astype(float).values.reshape((-1,world_size)).mean(axis=1).tolist()
"""
loss = pd.read_csv(r'D:\Projects\AI+yao\metabolism\microbiomeMetabolism\PALACE\PALACE_v13.loss_accu.log',sep='\t',header=None)[2].apply(split_).\
                    astype(float).values.reshape((-1,world_size)).mean(axis=1).tolist()[600:1200]
accuracy = pd.read_csv(r'D:\Projects\AI+yao\metabolism\microbiomeMetabolism\PALACE\PALACE_v13.loss_accu.log',sep='\t',header=None)[3].apply(split_).\
                    astype(float).values.reshape((-1,world_size)).mean(axis=1).tolist()[600:1200]
"""
train_loss = loss[::2]
test_loss = loss[1:][::2]
train_accu = accuracy[::2]
test_accu = accuracy[1:][::2]

#折线图
x = list(range(len(train_loss)))#点的横坐标
y = train_loss #线1的纵坐标
"""
#k2 = [0.8988,0.9334,0.9435,0.9407,0.9453,0.9453]#线2的纵坐标
plt.plot(x,y,'s-',color = 'r',label="loss")#s-:方形
#plt.plot(x,k2,'o-',color = 'g',label="CNN-RLSTM")#o-:圆形
plt.xlabel("epoch")#横坐标名字
plt.ylabel("train_loss")#纵坐标名字
plt.legend(loc = "best")#图例
plt.show()
"""
df = pd.DataFrame({'x':x,'train_loss':y})
ax = df.plot(x='x', y='train_loss')
df = df.assign(train_loss_derivative=df.diff().eval('train_loss/x'))
df.plot(x='x', y='train_loss_derivative', ax=ax)


x = list(range(len(test_loss)))#点的横坐标
y = test_loss #线1的纵坐标
"""
#k2 = [0.8988,0.9334,0.9435,0.9407,0.9453,0.9453]#线2的纵坐标
plt.plot(x,y,'s-',color = 'r',label="loss")#s-:方形
#plt.plot(x,k2,'o-',color = 'g',label="CNN-RLSTM")#o-:圆形
plt.xlabel("epoch")#横坐标名字
plt.ylabel("test_loss")#纵坐标名字
plt.legend(loc = "best")#图例
plt.show()
"""
df = pd.DataFrame({'x':x,'test_loss':y})
ax = df.plot(x='x', y='test_loss')
df = df.assign(test_loss_derivative=df.diff().eval('test_loss/x'))
df.plot(x='x', y='test_loss_derivative', ax=ax)

#折线图
x = list(range(len(train_accu)))#点的横坐标
y = train_accu #线1的纵坐标
"""
#k2 = [0.8988,0.9334,0.9435,0.9407,0.9453,0.9453]#线2的纵坐标
plt.plot(x,y,'s-',color = 'r',label="accuracy")#s-:方形
#plt.plot(x,k2,'o-',color = 'g',label="CNN-RLSTM")#o-:圆形
plt.xlabel("epoch")#横坐标名字
plt.ylabel("train_accuracy")#纵坐标名字
plt.legend(loc = "best")#图例
plt.show()
"""
df = pd.DataFrame({'x':x,'train_accuracy':y})
ax = df.plot(x='x', y='train_accuracy')
df = df.assign(train_accuracy_derivative=df.diff().eval('train_accuracy/x'))
df.plot(x='x', y='train_accuracy_derivative', ax=ax)


x = list(range(len(test_accu)))#点的横坐标
y = test_accu #线1的纵坐标
"""
#k2 = [0.8988,0.9334,0.9435,0.9407,0.9453,0.9453]#线2的纵坐标
plt.plot(x,y,'s-',color = 'r',label="accuracy")#s-:方形
#plt.plot(x,k2,'o-',color = 'g',label="CNN-RLSTM")#o-:圆形
plt.xlabel("epoch")#横坐标名字
plt.ylabel("test_accuracy")#纵坐标名字
plt.legend(loc = "best")#图例
plt.show()
#plt.savefig("data.png")
"""
df = pd.DataFrame({'x':x,'test_accuracy':y})
ax = df.plot(x='x', y='test_accuracy')
df = df.assign(test_accuracy_derivative=df.diff().eval('test_accuracy/x'))
df.plot(x='x', y='test_accuracy_derivative', ax=ax)


##########
x = list(range(len(test_loss)))#点的横坐标
y = [(x[1] - x[0])/x[1] for x in zip(train_loss,test_loss)]
"""
#k2 = [0.8988,0.9334,0.9435,0.9407,0.9453,0.9453]#线2的纵坐标
plt.plot(x,y,'s-',color = 'r',label="accuracy")#s-:方形
#plt.plot(x,k2,'o-',color = 'g',label="CNN-RLSTM")#o-:圆形
plt.xlabel("epoch")#横坐标名字
plt.ylabel("test_accuracy")#纵坐标名字
plt.legend(loc = "best")#图例
plt.show()
#plt.savefig("data.png")
"""
df = pd.DataFrame({'x':x,'rel_loss_diff':y})
ax = df.plot(x='x', y='rel_loss_diff')
df = df.assign(rel_loss_diff_derivative=df.diff().eval('rel_loss_diff/x'))
df.plot(x='x', y='rel_loss_diff_derivative', ax=ax)


