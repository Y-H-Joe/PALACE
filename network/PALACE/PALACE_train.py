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
import os
import sys
import torch
from torch import nn
from modules import MaskedSoftmaxCELoss,Timer,Accumulator,printer,grad_clipping,\
                    set_random_seed,load_data_nmt,PALACE_Encoder,PALACE_Decoder,\
                    PALACE,logging,try_gpu

# ===============================Settings======================================
#%% Settings
seed = 3434
# True or False
print_shape = False
# each smi_tok or prot_feat will be projected to feat_space_dim
feat_space_dim = 128
# notation protein length (any length of protein will be projected to fix length)
prot_nota_len = 1000
# number of encoder/decoder blocks
num_blks = 2
# dropout ratio for AddNorm,PositionalEncoding,DotProductMixAttention,ProteinEncoding
dropout = 0.2
# number of samples using per train
batch_size = 256
# time steps/window size,ref d2l 8.1 and 8.3
num_steps = 10
# learning rate
lr = 0.005
# number of epochs
num_epochs = 1
# feed forward intermidiate number of hiddens
ffn_num_hiddens = 64
# number of heads
num_heads = 8
# protein encoding features feed forward
prot_MLP = [128]
# multi-head attention will divide feat_space_num by num_heads
assert feat_space_dim % num_heads == 0, "feat_space_dim % num_heads != 0."
# ===============================Training======================================
#%% Training

def train_PALACE(piece,global_epoch,net, data_iter, lr, num_epochs, tgt_vocab,
                 device,loss_log, trained_model_dir=None):
    """训练序列到序列模型
    """
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2)  # 训练损失总和，词元数量
        assert trained_model_dir is not None, "trained model is not available."
        for batch in data_iter:
            optimizer.zero_grad()
            X_prot, X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # bos: torch.Size([batch_size, 1])
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],device=device).reshape(-1, 1)

            # dec_input: torch.Size([batch_size, num_steps])
            # removed the last tok in each sample of Y: (Y: [batch_size, num_steps-1])
            # add bos tok in begining of each sample of Y: (dec_input[batch_size, num_steps])
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # force teaching
            printer("train_PALACE:","X_prot",X_prot.shape)
            printer("train_PALACE:","X",X.shape)

            Y_hat, _ = net((X_prot, X), (X_prot,dec_input), X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward() # 损失函数的标量进行“反向传播”
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        with open(loss_log,'a') as o:
            o.write(f'global_epoch:{global_epoch}\tpiece:{piece}\tepoch:{epoch}\tloss:{metric[0] / metric[1]:.3f}\tsec:{metric[1] / timer.stop():.1f}\tdevice: {str(device)}\n')

# data 分成300份训练。但是要保证vocab一样。能存储vocab，能加载vocab
piece = 1
global_epoch = 1
loss_log = 'PALACE.loss.log'
data_dir = './data/PALACE_train.shuf.tsv_000'
trained_model_dir = './trained_models/'
print_shape = False
first_train = True
printer("=======================PALACE: assigning GPU...=======================",print_=True)
# multi-gpu
# gpu id/cpu
gpu_rank = int(os.environ['LOCAL_RANK'])
device = try_gpu(gpu_rank)
# set the cuda backend seed
set_random_seed(seed, gpu_rank>= 0)

printer("=======================PALACE: loading data...=======================",print_=True)
if first_train:
    vocab_dir = None
    data_iter, src_vocab, tgt_vocab = load_data_nmt(
            data_dir,batch_size,num_steps, device, vocab_dir, trained_model_dir)
else:
    # if not first train, will use former vocab
    vocab_dir = ['./saved/src_vocab.pkl','./saved/tgt_vocab.pkl']
    data_iter, src_vocab, tgt_vocab = load_data_nmt(
            data_dir,batch_size,num_steps, device, vocab_dir, trained_model_dir)

printer("=======================PALACE: building encoder...=======================",print_=True)
vocab_size = len(src_vocab)
encoder = PALACE_Encoder(
    vocab_size, prot_nota_len, feat_space_dim, ffn_num_hiddens, num_heads, num_blks, dropout, prot_MLP)

printer("=======================PALACE: building decoder...=======================",print_=True)
vocab_size = len(tgt_vocab)
decoder = PALACE_Decoder(
    vocab_size, prot_nota_len, feat_space_dim, ffn_num_hiddens, num_heads, num_blks, dropout, prot_MLP)

printer("=======================PALACE: training...=======================",print_=True)
net = PALACE(encoder,decoder).to(device)
net = net.cuda()
net = torch.nn.parallel.DistributedDataParallel(net, find_unused_parameters=True)
train_PALACE(piece,global_epoch, net, data_iter, lr, num_epochs, tgt_vocab, device, loss_log, trained_model_dir)

printer("=======================PALACE: saving model...=======================",print_=True)
torch.save(net.state_dict(), './trained_models/PALACE_{}_{}.pt'.format(global_epoch,piece))

logging.shutdown()













