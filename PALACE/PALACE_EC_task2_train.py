#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:21:36 2022

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ description ==============================####
prot encoder + feedforward layer + softmax output
EC ExPAsy in total has 8060 enzyme types.
The first position has 7 types; second has 26 types; third has 32 types; fourth has 442 types
=================================== input =====================================
Q51521  1.10.3.16       MNSSVLGKPLL
Q396C5  1.10.3.16       MNTSRFESLTG
Q51793  1.10.3.16       MNGSIQGKPL
=================================== output ====================================
prot encoder will process all sequences to tensor (num_steps * feat_space_dim),
which is 300 * 256 = 76800 dimensions space, around 10 times larger then total
enzyme types.
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

from modules_v3 import (printer,PALACE_prot,logging,save_on_master,train_PALACE_prot,model_diagnose,
                    set_random_seed,load_data_EC,PALACE_Encoder_v2,
                    assign_gpu,init_weights_v2,setup_gpu)


def main(rank, world_size,piece,model_id):
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
            # number of encoder/decoder blocks
            self.prot_blks = 5 # 5
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
            self.lr = 0.0001
            # number of epochs
            self.num_epochs = 30 # 30 for 4 gpus
            # feed forward intermidiate number of hiddens
            self.ffn_num_hiddens = 64 # 64
            # number of heads
            self.num_heads = 8 # 8
            # number of classes
            self.class_num = 8

    args = Args()
    assert args.feat_space_dim % args.num_heads == 0, "feat_space_dim % num_heads != 0."

    # set the cuda backend seed
    set_random_seed(args.seed, rank>= 0)
    setup_gpu(rank, world_size,12355)
    device = assign_gpu(rank)
    diagnose = False

# ===============================Training======================================
#%% Training
    loss_log = rf'PALACE_{model_id}.loss_accu.log'
    data_dir = rf'./data/PALACE_EC_task2.train.tsv_{0:04}'.format(piece)

    if int(piece) == 0: first_train = True
    else: first_train = False

    printer("=======================PALACE: loading data...=======================",print_=True)
    tp1 = time.time()
    vocab_dir = ['./vocab/prot_vocab.pkl','./vocab/smi_vocab_v2.pkl']
    data_iter,prot_vocab, src_vocab = load_data_EC(rank, world_size, args.batch_size, data_dir, vocab_dir)
    tp2 = time.time()
    printer("=======================loading data: {}s...=======================".format(tp2 - tp1),print_=True)

    printer("=======================PALACE: building model...=======================",print_=True)
    tp3 = time.time()
    # protein encoder at evaluation mode
    prot_encoder = PALACE_Encoder_v2(
        len(src_vocab), args.feat_space_dim, args.ffn_num_hiddens, args.num_heads,
        args.prot_blks, args.dropout,args.prot_blks + args.cross_blks, args.dec_blks,is_prot = True, num_steps = args.num_steps,device = device)
    # freeze prot_encoder
    for param in prot_encoder.parameters():
        param.requires_grad = False
    # feedforward layers, need training
    prot_net = PALACE_prot(prot_encoder, args.feat_space_dim, args.class_num)
    # optimizer only optimize feedforward part
    optimizer = torch.optim.NAdam(filter(lambda p: p.requires_grad, prot_net.parameters()), args.lr)
    # schedular
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=2,threshold=1e-6,factor=0.9)
    # loss
    loss = torch.nn.CrossEntropyLoss()
    tp4 = time.time()
    printer("=======================building model: {}s...=======================".format(tp4 - tp3),print_=True)

    printer("=======================PALACE: running on {}...=======================".format(device),print_=True)
    prot_net.to(device)
    loss.to(device)
    prot_net = torch.nn.parallel.DistributedDataParallel(prot_net,device_ids=[rank],output_device=rank,find_unused_parameters=True)
    net_without_ddp = prot_net.module

    printer("=======================PALACE: training...=======================",print_=True)
    if first_train:
        # net_without_ddp.apply(init_weights_v2) # deepnorm init implemented
        loss.apply(init_weights_v2)

        init = {
        'net': net_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'loss': loss.state_dict()}
        save_on_master(init, f'./PALACE_models/PALACE_{model_id}_init.pt')
        # transfer learning
        checkpoint_path = r'./PALACE_models/PALACE_EC_task_init.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # optimizer.load_state_dict(checkpoint['optimizer'])
        net_without_ddp.load_state_dict(checkpoint['net'],strict = False)
        # scheduler.load_state_dict(checkpoint['scheduler'])
        loss.load_state_dict(checkpoint['loss'],strict = False)
    else:
        checkpoint_path = f'./PALACE_models/checkpoint_PALACE_{model_id}.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # optimizer.load_state_dict(checkpoint['optimizer'])
        net_without_ddp.load_state_dict(checkpoint['net'],strict = False)
        # scheduler.load_state_dict(checkpoint['scheduler'])
        loss.load_state_dict(checkpoint['loss'],strict = False)
    tp5 = time.time()
    train_PALACE_prot(piece,prot_net, data_iter,optimizer,scheduler, loss, args.num_epochs,
                      device,loss_log, model_id, diagnose)
    tp6 = time.time()
    printer("=======================training: {}s...=======================".format(tp6 - tp5),print_=True)

    printer("=======================PALACE: saving model...=======================",print_=True)

    checkpoint = {
        'net': net_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'loss': loss.state_dict()}

    save_on_master(checkpoint, './PALACE_models/PALACE_{}_piece_{}.pt'.format(model_id,piece))
    save_on_master(checkpoint, f'./PALACE_models/checkpoint_PALACE_{model_id}.pt')

    # clean up
    logging.shutdown()
    dist.destroy_process_group()
    printer("=======================PALACE: finished! : {}s=======================".format(time.time() - tp1),print_=True)

    if diagnose:
        model_diagnose(model_id)

if __name__ == '__main__':
    piece = int(sys.argv[1])
    # piece = 0
    # suppose we have `world_size` gpus
    world_size = int(sys.argv[2])
    # world_size = 1
    model_id = 'EC_task2'

    mp.spawn(
        main,
        args=(world_size,piece,model_id),
        nprocs=world_size
    )
    # main(0,world_size,piece,model_id)




