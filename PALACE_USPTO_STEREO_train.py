#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 13:53:07 2022

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
import time
import sys

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from modules_v3 import (printer,logging,save_on_master,train_PALACE_SMILES,model_diagnose,
                    set_random_seed,load_data,PALACE_Encoder_v2,MaskedSoftmaxCELoss,
                    assign_gpu,init_weights_v2,setup_gpu,PALACE_Decoder_v2,PALACE_SMILES)


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
            # self.prot_nota_len = 2 # 1024
            # number of encoder/decoder blocks
            self.smi_blks = 5 # 5
            self.dec_blks = 14 # 14
            # dropout ratio for AddNorm,PositionalEncoding,DotProductMixAttention,ProteinEncoding
            self.dropout = 0.01
            # number of samples using per train
            self.batch_size = 10 # 20 when 2 gpus, 16 when 4 gpus
            # number of protein reading when trans protein to features using pretrained BERT
            #self.prot_read_batch_size = 6
            # time steps/window size,ref d2l 8.1 and 8.3
            self.num_steps = 300
            # learning rate
            self.lr = 0.00001
            # number of epochs
            self.num_epochs = 100 # 30 for 4 gpus
            # feed forward intermidiate number of hiddens
            self.ffn_num_hiddens = 64 # 64
            # number of heads
            self.num_heads = 8 # 8
            # protein encoding features feed forward
            # self.prot_MLP = [5] #128
            # multi-head attention will divide feat_space_num by num_heads

    args = Args()
    assert args.feat_space_dim % args.num_heads == 0, "feat_space_dim % num_heads != 0."

    # set the cuda backend seed
    set_random_seed(args.seed, rank>= 0)
    try:
        setup_gpu(rank, world_size,12355)
    except:
        setup_gpu(rank, world_size,12356)
    device = assign_gpu(rank)
    diagnose = False

# ===============================Training======================================
#%% Training
    loss_log = rf'PALACE_{model_id}.loss_accu.log'
    data_dir = f'./data/PALACE_USPTO_STEREO_train.final.tsv_{0:04}'.format(piece)
    # data_dir = './data/fra.txt'
    #data_dir = './data/fake_sample_for_vocab.txt'
    if int(piece) == 0:
        first_train = True
        # args.num_epochs = 1000 # trick, use small batch deep epoch to init parameters
    else: first_train = False


    printer("=======================PALACE: loading data...=======================",print_=True)
    # if not first train, will use former vocab
    tp1 = time.time()
    vocab_dir = ['./vocab/smi_vocab_v2.pkl','./vocab/smi_vocab_v2.pkl','./vocab/prot_vocab.pkl']
    # vocab_dir = None
    data_iter, src_vocab, tgt_vocab, prot_vocab = load_data(rank, world_size,
            data_dir,args.batch_size, args.num_steps, device, vocab_dir)
    tp2 = time.time()
    printer("=======================loading data: {}s...=======================".format(tp2 - tp1),print_=True)


    printer("=======================PALACE: building model...=======================",print_=True)
    tp3 = time.time()

    smi_encoder = PALACE_Encoder_v2(
        len(src_vocab), args.feat_space_dim, args.ffn_num_hiddens, args.num_heads,
        args.smi_blks, args.dropout,args.smi_blks, args.dec_blks, device = device)


    decoder = PALACE_Decoder_v2(
        len(tgt_vocab), args.feat_space_dim, args.ffn_num_hiddens, args.num_heads,
        args.dec_blks, args.dropout, args.smi_blks, args.dec_blks)

    net = PALACE_SMILES(smi_encoder, decoder,args.feat_space_dim,
                    args.ffn_num_hiddens,args.dropout, args.smi_blks, args.dec_blks)

    # optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(net.parameters(), args.lr,momentum=0.9)
    optimizer = torch.optim.NAdam(net.parameters(), args.lr)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=2,threshold=1e-6,factor=0.9)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    loss = MaskedSoftmaxCELoss(len(tgt_vocab),device = device)
    tp4 = time.time()
    printer("=======================building model: {}s...=======================".format(tp4 - tp3),print_=True)


    printer("=======================PALACE: running on {}...=======================".format(device),print_=True)

    net.to(device)
    loss.to(device)
    net = torch.nn.parallel.DistributedDataParallel(net,device_ids=[rank],output_device=rank,find_unused_parameters=True)
    net_without_ddp = net.module


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
    else:
        checkpoint_path = f'./PALACE_models/checkpoint_PALACE_{model_id}.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # optimizer.load_state_dict(checkpoint['optimizer'])
        net_without_ddp.load_state_dict(checkpoint['net'],strict = False)
        # scheduler.load_state_dict(checkpoint['scheduler'])
        loss.load_state_dict(checkpoint['loss'],strict = False)
    tp5 = time.time()
    train_PALACE_SMILES(piece, net, data_iter, optimizer,scheduler,loss, args.num_epochs, tgt_vocab, device, loss_log, model_id, diagnose)
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
    model_id = 'USPTO_STEREO'

    mp.spawn(
        main,
        args=(world_size,piece,model_id),
        nprocs=world_size
    )

    # main(0,world_size,piece,model_id)










