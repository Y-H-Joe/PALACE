# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 15:02:15 2021

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ discription ==============================####
the old dictionary size is different from the new one. So need to mannually
modify the dict_state of embedding layer.
#================================== input =====================================
#================================== output ====================================
#================================ parameters ==================================
#================================== example ===================================
#================================== warning ===================================
####=======================================================================####
"""
import torch

checkpoint_path = '../PALACE_models/checkpoint_v7.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

para_list = ['smi_encoder.embedding.weight','decoder.embedding.weight',
             'decoder.dense.weight','decoder.dense.bias','prot_encoder.embedding.weight']

for para in para_list:
    # use average to expand the dimension
    average = checkpoint['net'][para].mean(dim = 0, keepdim = True)
    # old dictionary size is 166, new one is 287, want to expand
    average_repeat = average.repeat_interleave((287-166), dim = 0)
    new_para = torch.cat([checkpoint['net'][para],average_repeat])
    checkpoint['net'][para] = new_para

torch.save(checkpoint, f'../PALACE_models/checkpoint_v7.ptt')
