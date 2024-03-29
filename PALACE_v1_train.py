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
seperate big dataset into pieces
for i in range(128):
    shutil.copy('PALACE_train.enzyme_and_nonenzyme.shuffle.v4.tsv_{0:04}'.format(i),'PALACE_train.enzyme_and_nonenzyme.shuffle.v4.tsv_{0:04}'.format(i+256))

####=======================================================================####
"""
import time
import sys

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from modules_v3 import (printer,PALACE_v2,logging,save_on_master,train_PALACE,model_diagnose,
                    set_random_seed,load_data,PALACE_Encoder_v2,MaskedSoftmaxCELoss,
                    assign_gpu,init_weights_v2,setup_gpu,PALACE_Decoder_v2)


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
            self.lr = 0.00000009
            # number of epochs
            self.num_epochs = 10 # 30 for 4 gpus
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
    data_dir = './data/PALACE_train.shuf.batch1.tsv_{0:04}'.format(piece)
    # data_dir = './data/fra.txt'
    #data_dir = './data/fake_sample_for_vocab.txt'

    if int(piece) == 0:
        first_train = True
        args.num_epochs = 1000 # trick, use small batch deep epoch to init parameters
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
        args.smi_blks, args.dropout,args.prot_blks + args.cross_blks, args.dec_blks, device = device)

    prot_encoder = PALACE_Encoder_v2(
        len(src_vocab), args.feat_space_dim, args.ffn_num_hiddens, args.num_heads,
        args.prot_blks, args.dropout, args.prot_blks + args.cross_blks, args.dec_blks,
        is_prot = True, num_steps = args.num_steps,device = device)

    cross_encoder = PALACE_Encoder_v2(
        len(src_vocab), args.feat_space_dim, args.ffn_num_hiddens, args.num_heads,
        args.cross_blks, args.dropout, args.prot_blks + args.cross_blks, args.dec_blks,
        is_cross = True,device = device)

    decoder = PALACE_Decoder_v2(
        len(tgt_vocab), args.feat_space_dim, args.ffn_num_hiddens, args.num_heads,
        args.dec_blks, args.dropout, args.prot_blks + args.cross_blks, args.dec_blks)

    net = PALACE_v2(smi_encoder, prot_encoder, cross_encoder, decoder,args.feat_space_dim,
                    args.ffn_num_hiddens,args.dropout, args.prot_blks + args.cross_blks, args.dec_blks)

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

        save_on_master(init, f'./PALACE_models/init_{model_id}.pt')
    else:
        checkpoint_path = './PALACE_models/checkpoint_{}.pt'.format(model_id)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # optimizer.load_state_dict(checkpoint['optimizer'])
        net_without_ddp.load_state_dict(checkpoint['net'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        loss.load_state_dict(checkpoint['loss'])
    tp5 = time.time()
    train_PALACE(piece, net, data_iter, optimizer,scheduler,loss, args.num_epochs, tgt_vocab, device, loss_log, model_id, diagnose)
    tp6 = time.time()
    printer("=======================training: {}s...=======================".format(tp6 - tp5),print_=True)

    printer("=======================PALACE: saving model...=======================",print_=True)

    checkpoint = {
        'net': net_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'loss': loss.state_dict()}

    save_on_master(checkpoint, './PALACE_models/PALACE_{}_piece_{}.pt'.format(model_id,piece))
    save_on_master(checkpoint, f'./PALACE_models/checkpoint_{model_id}.pt')

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
    model_id = 'v1'

    mp.spawn(
        main,
        args=(world_size,piece,model_id),
        nprocs=world_size
    )

    # main(0,world_size,piece,model_id)










