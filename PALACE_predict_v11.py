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
from tqdm import tqdm

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from PALACE import (printer,PALACE_v2,logging,predict_PALACE,model_diagnose,
                    PALACE_Encoder_v2,assign_gpu,PALACE_Decoder_v2,retrieve_vocab)


def main(model_dp, smi_vocab_dp, prot_vocab_dp, output_dp, beam = 5):
# ===============================Settings======================================
#%% Settings
    class Args:
        def __init__(self):
            self.seed = 1996
            # True or False
            self.print_shape = False
            # each smi_tok or prot_feat will be projected to feat_space_dim
            self.feat_space_dim = 260 #256
            # notation protein length (any length of protein will be projected to fix length)
            # self.prot_nota_len = 2 # 1024
            # number of encoder/decoder blocks
            self.prot_blks = 9 # 5
            self.smi_blks = 9 # 5
            self.cross_blks = 9 # 9
            self.dec_blks = 11 # 14
            # dropout ratio for AddNorm,PositionalEncoding,DotProductMixAttention,ProteinEncoding
            self.dropout = 0.01
            # number of samples using per train
            self.batch_size = 30 # 20 when 2 gpus, 16 when 4 gpus
            # number of protein reading when trans protein to features using pretrained BERT
            #self.prot_read_batch_size = 6
            # time steps/window size,ref d2l 8.1 and 8.3
            self.num_steps = 300
            # learning rate
            self.lr = 0.00001
            # number of epochs
            self.num_epochs = 5 # 30 for 4 gpus
            # feed forward intermidiate number of hiddens
            self.ffn_num_hiddens = 128 # 64
            # number of heads
            self.num_heads = 10 # 8
            # protein encoding features feed forward
            # self.prot_MLP = [5] #128
            # multi-head attention will divide feat_space_num by num_heads

    args = Args()
    assert args.feat_space_dim % args.num_heads == 0, "feat_space_dim % num_heads != 0."

    try: device = assign_gpu(0, print_=False)
    except:
        try: device = assign_gpu(1, print_=False)
        except:
            try: device = assign_gpu(2, print_=False)
            except: device = assign_gpu(3, print_=False)
#    diagnose = False

# ===============================Predicting======================================
#%% Predicting
    # printer("=======================PALACE: loading data...=======================",print_=True)

    smi_vocab = retrieve_vocab(smi_vocab_dp)
    prot_vocab = retrieve_vocab(prot_vocab_dp)

# =============================================================================
#     samples = []
#     with open(data_dp,'r') as r:
#         [samples.append(x.strip().split('\t')[2:4]) for x in r.readlines()]
# =============================================================================


    # printer("=======================PALACE: building model...=======================",print_=True)
    smi_encoder = PALACE_Encoder_v2(
        len(smi_vocab), args.feat_space_dim, args.ffn_num_hiddens, args.num_heads,
        args.smi_blks, args.dropout,args.prot_blks + args.cross_blks, args.dec_blks, device = device)

    prot_encoder = PALACE_Encoder_v2(
        len(smi_vocab), args.feat_space_dim, args.ffn_num_hiddens, args.num_heads,
        args.prot_blks, args.dropout, args.prot_blks + args.cross_blks, args.dec_blks,
        is_prot = True, num_steps = args.num_steps,device = device)

    cross_encoder = PALACE_Encoder_v2(
        len(smi_vocab), args.feat_space_dim, args.ffn_num_hiddens, args.num_heads,
        args.cross_blks, args.dropout, args.prot_blks + args.cross_blks, args.dec_blks,
        is_cross = True,device = device)

    decoder = PALACE_Decoder_v2(
        len(smi_vocab), args.feat_space_dim, args.ffn_num_hiddens, args.num_heads,
        args.dec_blks, args.dropout, args.prot_blks + args.cross_blks, args.dec_blks)

    net = PALACE_v2(smi_encoder, prot_encoder, cross_encoder, decoder,args.feat_space_dim,
                    args.ffn_num_hiddens,args.dropout, args.prot_blks + args.cross_blks, args.dec_blks)

    # printer("=======================PALACE loading model...=======================",print_=True)
    checkpoint = torch.load(model_dp, map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    net.to(device)
    # printer("=======================PALACE: running on {}...=======================".format(device),print_=True)

    # printer("=======================PALACE: predicting...=======================",print_=True)
    # prediction = predict_PALACE(net, src, prot_vocab, smi_vocab, args.num_steps,device,beam,save_attention_weights=False)

#    if diagnose:
#        model_diagnose(model_id)
    # return prediction
    return net, prot_vocab, smi_vocab, args.num_steps,device

if __name__ == '__main__':
    num = sys.argv[1]
    model_dp = rf"PALACE_models/PALACE_v10_piece_{num}.pt"
    smi_vocab_dp = r'vocab/smi_vocab_v10.pkl'
    prot_vocab_dp = r'vocab/prot_vocab_v10.pkl'
    beam = 5

    for i in range(5):
        data_dp = r'data/PALACE_test.enzyme_and_nonenzyme.shuffle.v4.tsv_{0:04}'.format(i)
        output_dp = rf'predictions/PALACE_v10_model_{num}_piece_{i}.txt'
        skip_log = rf'predictions/PALACE_v10_model_{num}_piece_{i}.skipped.txt'
        predictions = []
        skipped = []
        net, prot_vocab, smi_vocab, num_steps,device = main(model_dp, smi_vocab_dp, prot_vocab_dp, output_dp, beam)
        with open(data_dp,'r') as r:
            with torch.no_grad():
                for x in tqdm(r.readlines()):
                    src = x.strip().split('\t')[2:4]
                    try:
                        prediction = predict_PALACE(net, src, prot_vocab, smi_vocab, num_steps,device,beam,save_attention_weights=False)
                        # o.write(str(prediction) + '\n')
                        predictions.append(prediction)
                    except:
                        # print("PALACE: skipped!")
                        skipped.append(x)
                        predictions.append([""] * beam)
        with open(skip_log,'w') as s, open(output_dp,'w') as o:
            for x in skipped:
                s.write(x)
            for x in predictions:
                o.write(str(x) + '\n')
        time.sleep(300)











