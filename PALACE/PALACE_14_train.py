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

from modules import (printer,PALACE,logging,save_on_master,train_PALACE,
                    set_random_seed,load_data,PALACE_Encoder,PALACE_Decoder,
                    assign_gpu,xavier_init_weights,setup_gpu,PALACE_prot_net )


def main(rank, world_size,piece,model_id):
# ===============================Settings======================================
#%% Settings
    class Args:
        def __init__(self):
            self.seed = 1996
            # True or False
            self.print_shape = False
            # each smi_tok or prot_feat will be projected to feat_space_dim
            self.feat_space_dim = 256
            # notation protein length (any length of protein will be projected to fix length)
            self.prot_nota_len = 1024
            # number of encoder/decoder blocks
            self.num_blks = 14
            # dropout ratio for AddNorm,PositionalEncoding,DotProductMixAttention,ProteinEncoding
            self.dropout = 0.2
            # number of samples using per train
            self.batch_size = 4
            # number of protein reading when trans protein to features using pretrained BERT
            #self.prot_read_batch_size = 6
            # time steps/window size,ref d2l 8.1 and 8.3
            self.num_steps = 250
            # learning rate
            self.lr = 0.05
            # number of epochs
            self.num_epochs = 1
            # feed forward intermidiate number of hiddens
            self.ffn_num_hiddens = 64
            # number of heads
            self.num_heads = 8
            # protein encoding features feed forward
            self.prot_MLP = [128]
            # multi-head attention will divide feat_space_num by num_heads

    args = Args()
    assert args.feat_space_dim % args.num_heads == 0, "feat_space_dim % num_heads != 0."

    # set the cuda backend seed
    set_random_seed(args.seed, rank>= 0)
    setup_gpu(rank, world_size)
    device = assign_gpu(rank)

# ===============================Training======================================
#%% Training
    loss_log = 'PALACE.loss.log'
    data_dir = './data/PALACE_train.shuf.batch1.tsv_{0:04}'.format(piece)
    #data_dir = './data/fake_sample_for_vocab.txt'

    if int(piece) == 0: first_train = True
    else: first_train = False


    printer("=======================PALACE: loading data...=======================",print_=True)
    # if not first train, will use former vocab
    tp1 = time.time()
    vocab_dir = ['./vocab/merge_vocab.pkl','./vocab/merge_vocab.pkl','./vocab/prot_vocab.pkl']
    data_iter, src_vocab, tgt_vocab, prot_vocab = load_data(rank, world_size,
            data_dir,args.batch_size, args.num_steps, device, vocab_dir)
    tp2 = time.time()
    printer("=======================loading data: {}s...=======================".format(tp2 - tp1),print_=True)


    printer("=======================PALACE: building model...=======================",print_=True)
    tp3 = time.time()
    encoder = PALACE_Encoder(
        len(src_vocab), args.prot_nota_len, args.feat_space_dim, args.ffn_num_hiddens, args.num_heads, args.num_blks, args.dropout, args.prot_MLP)
    decoder = PALACE_Decoder(
        len(tgt_vocab), args.prot_nota_len, args.feat_space_dim, args.ffn_num_hiddens, args.num_heads, args.num_blks, args.dropout, args.prot_MLP)
    prot_net = PALACE_prot_net(len(prot_vocab), args.feat_space_dim, args.prot_nota_len, args.dropout)
    net = PALACE(encoder,decoder,prot_net)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    tp4 = time.time()
    printer("=======================building model: {}s...=======================".format(tp4 - tp3),print_=True)


    printer("=======================PALACE: running on {}...=======================".format(device),print_=True)

    net.to(device)
    net = torch.nn.parallel.DistributedDataParallel(net,device_ids=[rank],output_device=rank,find_unused_parameters=True)
    net_without_ddp = net.module


    printer("=======================PALACE: training...=======================",print_=True)
    if first_train:
        net.apply(xavier_init_weights)
    else:
        checkpoint_path = './PALACE_models/checkpoint_{}.pt'.format(model_id)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        net_without_ddp.load_state_dict(checkpoint['net'])
    tp5 = time.time()
    train_PALACE(piece, net, data_iter, optimizer, args.num_epochs, tgt_vocab, device, loss_log)
    tp6 = time.time()
    printer("=======================training: {}s...=======================".format(tp6 - tp5),print_=True)

    printer("=======================PALACE: saving model...=======================",print_=True)
    checkpoint = {
        'net': net_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict()}

    save_on_master(checkpoint, './PALACE_models/PALACE_{}_piece_{}.pt'.format(model_id,piece))
    save_on_master(checkpoint, f'./PALACE_models/checkpoint_{model_id}.pt')

    # clean up
    logging.shutdown()
    dist.destroy_process_group()
    printer("=======================PALACE: finished! : {}s=======================".format(time.time() - tp1),print_=True)

if __name__ == '__main__':
    piece = int(sys.argv[1])
    # piece = 0
    # suppose we have `world_size` gpus
    #world_size = int(sys.argv[2])
    world_size = 1
    model_id = 14

# =============================================================================
#     mp.spawn(
#         main,
#         args=(world_size,piece,model_id),
#         nprocs=world_size
#     )
# =============================================================================
    main(0,world_size,piece,model_id)










