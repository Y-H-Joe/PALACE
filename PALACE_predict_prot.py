#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 13:53:07 2022

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ description ==============================####

=================================== input =====================================
protein_seq\tsmiles_seq
=================================== output ====================================

================================= parameters ==================================

=================================== example ===================================

=================================== warning ===================================

####=======================================================================####
"""
import time
import sys

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from modules_v3 import (printer,PALACE_prot,logging,predict_PALACE_prot,model_diagnose,
                    PALACE_Encoder_v2,assign_gpu,retrieve_vocab)


def main(model_dp, data_dp, smi_vocab_dp, prot_vocab_dp, output_dp, class_num):
# ===============================Settings======================================
#%% Settings
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
            self.prot_blks = 5 # 5
            self.smi_blks = 5 # 5
            self.cross_blks = 9 # 9
            self.dec_blks = 14 # 14
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
            # number of classes
            # self.class_num = 2

    args = Args()
    assert args.feat_space_dim % args.num_heads == 0, "feat_space_dim % num_heads != 0."

    try: device = assign_gpu(0)
    except:
        try: device = assign_gpu(1)
        except:
            try: device = assign_gpu(2)
            except: device = assign_gpu(3)
#    diagnose = False

# ===============================Predicting======================================
#%% Predicting
    printer("=======================PALACE: loading data...=======================",print_=True)
    tp1 = time.time()

    prot_vocab = retrieve_vocab(prot_vocab_dp)
    smi_vocab = retrieve_vocab(smi_vocab_dp)

    samples = []
    with open(data_dp,'r') as r:
        [samples.append(x.strip().split('\t')[1]) for x in r.readlines()]

    tp2 = time.time()
    printer("=======================loading data: {}s...=======================".format(tp2 - tp1),print_=True)

    printer("=======================PALACE: building model...=======================",print_=True)
    prot_encoder = PALACE_Encoder_v2(
        len(smi_vocab), args.feat_space_dim, args.ffn_num_hiddens, args.num_heads,
        args.prot_blks, args.dropout,args.prot_blks + args.cross_blks, args.dec_blks,is_prot = True, num_steps = args.num_steps,device = device)
    # freeze prot_encoder
    for param in prot_encoder.parameters():
        param.requires_grad = False
    # feedforward layers, need training
    net = PALACE_prot(prot_encoder, args.feat_space_dim, class_num, args.dropout, args.prot_blks, 1)

    printer("=======================PALACE loading model...=======================",print_=True)
    checkpoint = torch.load(model_dp, map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    net.to(device)
    printer("=======================PALACE: running on {}...=======================".format(device),print_=True)

    printer("=======================PALACE: predicting...=======================",print_=True)
    net.eval()
    tp5 = time.time()
    with open(output_dp,'w') as o:
        for src in samples:
            prediction = predict_PALACE_prot(net, src, prot_vocab, device)
            o.write(str(prediction) + '\n')
    tp6 = time.time()
    printer("=======================predicting: {}s...=======================".format(tp6 - tp5),print_=True)

    # clean up
    logging.shutdown()
    printer("=======================PALACE: finished! : {}s=======================".format(time.time() - tp1),print_=True)

#    if diagnose:
#        model_diagnose(model_id)
    return

if __name__ == '__main__':
    model_dp = sys.argv[1]
    data_dp = sys.argv[2]
    smi_vocab_dp = sys.argv[3]
    prot_vocab_dp = sys.argv[4]
    output_dp = sys.argv[5]
    class_num = int(sys.argv[6])
    """
    model_dp = r'PALACE_models/PALACE_v3_piece_97.pt'
    data_dp = r'data/PALACE_train.shuf.batch1.sample.tsv'
    smi_vocab_dp = r'vocab/smi_vocab_v2.pkl'
    prot_vocab_dp = r'vocab/prot_vocab.pkl'
    output_dp = 'aa'
    """
    main(model_dp, data_dp, smi_vocab_dp, prot_vocab_dp, output_dp,class_num)










