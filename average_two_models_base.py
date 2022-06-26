#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 10:50:52 2022

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ description ==============================####
=================================== input =====================================
=================================== output ====================================
================================= parameters ==================================
=================================== example ===================================
=================================== warning ===================================
####=======================================================================####
"""
import torch
from modules_v3 import (printer,PALACE_v2,logging,predict_PALACE,model_diagnose,
     save_on_master,PALACE_Encoder_v2,assign_gpu,PALACE_Decoder_v2,retrieve_vocab)

class Args:
    def __init__(self):
        self.seed = 1996
        # True or False
        self.print_shape = False
        # each smi_tok or prot_feat will be projected to feat_space_dim
        self.feat_space_dim = 256 #256
        # notation protein length (any length of protein will be projected to fix length)
        # self.prot_nota_len = 2 # 1024
        # number of encoder/decoder blocks
        self.prot_blks = 4 # 5
        self.smi_blks = 4 # 5
        self.cross_blks = 4 # 9
        self.dec_blks = 8 # 14
        # dropout ratio for AddNorm,PositionalEncoding,DotProductMixAttention,ProteinEncoding
        self.dropout = 0.01
        # number of samples using per train
        self.batch_size = 14 # 20 when 2 gpus, 16 when 4 gpus
        # number of protein reading when trans protein to features using pretrained BERT
        #self.prot_read_batch_size = 6
        # time steps/window size,ref d2l 8.1 and 8.3
        self.num_steps = 300
        # learning rate
        # self.lr = 0.00000009
        # number of epochs
        # self.num_epochs = 10 # 30 for 4 gpus
        # feed forward intermidiate number of hiddens
        self.ffn_num_hiddens = 64 # 64
        # number of heads
        self.num_heads = 8 # 8
        # protein encoding features feed forward
        # self.prot_MLP = [5] #128
        # multi-head attention will divide feat_space_num by num_heads
args = Args()

dp1 = r"C:\CurrentProjects\AI+yao\metabolism\microbiomeMetabolism\PALACE_models\checkpoint_v4_again_again.pt"
dp2 = r"C:\CurrentProjects\AI+yao\metabolism\microbiomeMetabolism\PALACE_models\checkpoint_v5_again_again.pt"
smi_vocab_dp = r"vocab/smi_vocab_v2.pkl"
prot_vocab_dp = r"vocab/prot_vocab.pkl"
output = r"C:\CurrentProjects\AI+yao\metabolism\microbiomeMetabolism\PALACE_models\checkpoint_v4_v5_ave_again_again.pt"

smi_vocab = retrieve_vocab(smi_vocab_dp)
prot_vocab = retrieve_vocab(prot_vocab_dp)

smi_encoder = PALACE_Encoder_v2(
    len(smi_vocab), args.feat_space_dim, args.ffn_num_hiddens, args.num_heads,
    args.smi_blks, args.dropout,args.prot_blks + args.cross_blks, args.dec_blks, device = 'cpu')

prot_encoder = PALACE_Encoder_v2(
    len(smi_vocab), args.feat_space_dim, args.ffn_num_hiddens, args.num_heads,
    args.prot_blks, args.dropout, args.prot_blks + args.cross_blks, args.dec_blks,
    is_prot = True, num_steps = args.num_steps,device = 'cpu')

cross_encoder = PALACE_Encoder_v2(
    len(smi_vocab), args.feat_space_dim, args.ffn_num_hiddens, args.num_heads,
    args.cross_blks, args.dropout, args.prot_blks + args.cross_blks, args.dec_blks,
    is_cross = True,device = 'cpu')

decoder = PALACE_Decoder_v2(
    len(smi_vocab), args.feat_space_dim, args.ffn_num_hiddens, args.num_heads,
    args.dec_blks, args.dropout, args.prot_blks + args.cross_blks, args.dec_blks)

net = PALACE_v2(smi_encoder, prot_encoder, cross_encoder, decoder,args.feat_space_dim,
                args.ffn_num_hiddens,args.dropout, args.prot_blks + args.cross_blks, args.dec_blks)


checkpoint1 = torch.load(dp1, map_location='cpu')
checkpoint2 = torch.load(dp2, map_location='cpu')

dicts1 = checkpoint1['net']
dicts2 = checkpoint2['net']
dicts3 = net.state_dict()

# Average all parameters
for key in dicts3:
    dicts3[key] = (dicts1[key] + dicts2[key]) / 2.
a3 = dicts3['cross_encoder.encoder.layers.0.self_attn.in_proj_bias']

checkpoint = {'net': dicts3}
save_on_master(checkpoint,output)

"""
# try to solve CUDA OOM with averaged model
dp3 = r"C:\CurrentProjects\AI+yao\metabolism\microbiomeMetabolism\PALACE_models\checkpoint_v4_v5_ave_again_again2.pt"
checkpoint3 = torch.load(dp3, map_location='cpu')
dicts33 = checkpoint3['net']
a33 = dicts33['cross_encoder.encoder.layers.0.self_attn.in_proj_bias']

"""

