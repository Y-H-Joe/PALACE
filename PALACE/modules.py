#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 18:08:11 2022

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

import math
import os
import time
import numpy as np
from collections import Counter
import pickle
import re
import logging
import random
# from tqdm import tqdm
from matplotlib_inline import backend_inline
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils import data
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


from transformers import BertForMaskedLM, BertTokenizer
from transformers import logging as tr_log
# will not print warning
tr_log.set_verbosity_error()

# ==============================Pytorch=================================
#%% Pytorch
class PALACE_Encoder_v2(nn.Module):
    def __init__(self, vocab_size, feat_space_dim, ffn_num_hiddens, num_heads,
                 num_blks, dropout, is_cross = False, is_prot = False, use_bias=False, num_steps = 0,device = 'cpu',**kwargs):
        super(PALACE_Encoder_v2, self).__init__()
        self.is_cross = is_cross
        self.is_prot = is_prot
        self.feat_space_dim = feat_space_dim
        if not is_cross:
            self.embedding = nn.Embedding(vocab_size, feat_space_dim)
            if not is_prot:
                self.pos_encoding = PositionalEncoding(feat_space_dim, dropout)
            else:
                self.num_steps = num_steps
                self.dense = nn.Linear(2500,num_steps)
                self.relu = nn.ReLU()
                # self.conv1d = nn.Conv1d(2500,num_steps,1,stride=1)
                self.ffn = PositionWiseFFN(feat_space_dim, ffn_num_hiddens)
                self.addnorm = AddNorm(feat_space_dim,dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model = feat_space_dim,
                                                   dropout = dropout,
                                                   nhead = num_heads,
                                                   batch_first = True,
                                                   dim_feedforward = ffn_num_hiddens,
                                                   activation = 'relu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blks)
    def forward(self, X, valid_lens):
        if not self.is_cross:
            X = self.embedding(X)
            if not self.is_prot:
                X = self.pos_encoding(X * math.sqrt(self.feat_space_dim))
            else:
                X = X.permute(0, 2, 1)
                X = self.relu(self.dense(X))
                X = X.permute(0, 2, 1)
                # X = self.conv1d(X)
                X = self.addnorm(self.ffn(X),X)
        if not self.is_prot:
            key_padding_mask = sequence_mask_v2(X, valid_lens)
            X = self.encoder(X, src_key_padding_mask = key_padding_mask)
        else:
            X = self.encoder(X)

        return X


class DecoderBlock_v2(nn.Module):
    """解码器中第i个块"""
    def __init__(self, feat_space_dim, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock_v2, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(feat_space_dim, num_heads, dropout)
        self.addnorm1 = AddNorm(feat_space_dim, dropout)
        self.attention2 = MultiHeadAttention(feat_space_dim, num_heads, dropout)
        self.addnorm2 = AddNorm(feat_space_dim, dropout)
        self.ffn = PositionWiseFFN(feat_space_dim, ffn_num_hiddens)
        self.addnorm3 = AddNorm(feat_space_dim, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size,num_steps),
            # 其中每一行是[1,2,...,num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # 自注意力
        q,k,v = X, key_values, key_values
        X2 = self.attention1(k,q,v, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器－解码器注意力。
        # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
        q,k,v = Y, enc_outputs, enc_outputs
        Y2 = self.attention2(k,q,v, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


class Decoder(nn.Module):
    """The base decoder interface for the encoder-decoder architecture.
    """
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError

    @property
    def attention_weights(self):
        raise NotImplementedError

class PALACE_Decoder_v2(Decoder):
    def __init__(self, vocab_size, feat_space_dim, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(PALACE_Decoder_v2, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.feat_space_dim = feat_space_dim
        self.embedding = nn.Embedding(vocab_size, feat_space_dim)
        self.pos_encoding = PositionalEncoding(feat_space_dim, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                DecoderBlock_v2(feat_space_dim, ffn_num_hiddens,
                             num_heads, dropout, i))
        self.dense = nn.Linear(feat_space_dim, vocab_size)
        # self.softmax = nn.Softmax()
        self.softmax = nn.LogSoftmax(dim=2)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.feat_space_dim))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][i] = blk.attention1.dot.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][i] = blk.attention2.dot.attention_weights
        return self.softmax(self.dense(X)), state

    @property
    def attention_weights(self):
        return self._attention_weights

# =============================================================================
#
# class MultiHeadAttention_v2(nn.MultiheadAttention):
#     """多头注意力
#     """
#     def __init__(self,feat_space_dim, num_heads, dropout, bias=False, **kwargs):
#         super(MultiHeadAttention_v2, self).__init__(**kwargs)
#         self.embed_dim = feat_space_dim
#         self.num_heads = num_heads
#         self.batch_first = True
#         self.bias = False
#         self.dropout = dropout
#
#         self.dot = DotProductMixAttention(dropout)
#         # nn.Linear can accept N-D tensor
#         # ref https://stackoverflow.com/questions/58587057/multi-dimensional-inputs-in-pytorch-linear-method
#         self.W_q = nn.Linear(feat_space_dim, feat_space_dim, bias=bias)
#         self.W_k = nn.Linear(feat_space_dim, feat_space_dim, bias=bias)
#         self.W_v = nn.Linear(feat_space_dim, feat_space_dim, bias=bias)
#         self.W_o = nn.Linear(feat_space_dim, feat_space_dim, bias=bias)
#
#     def forward(self, keys, queries, values, valid_lens):
#         # when is_cross: k,q,v = X_prot, X_smi, X_prot
#         # Shape of `queries`, `keys`, or `values`:
#         # (`batch_size`, `num_steps`, `num_hiddens`)
#         # Shape of `valid_lens`:
#         # (`batch_size`,) or (`batch_size`, no. of queries)
#         # After transposing, shape of output `queries`, `keys`, or `values`:
#         # (`batch_size` * `num_heads`, no. of queries or key-value pairs,`num_hiddens` / `num_heads`)
#         queries = transpose_qkv(self.W_q(queries), self.num_heads)
#         printer("MultiHeadAttention","keys",keys.shape)
#         keys = transpose_qkv(self.W_k(keys), self.num_heads)
#         values = transpose_qkv(self.W_v(values), self.num_heads)
#
#         if valid_lens is not None:
#             # 在轴0，将第一项（标量或者矢量）复制num_heads次，
#             # 然后如此复制第二项，然后诸如此类。
#             valid_lens = torch.repeat_interleave(
#                 valid_lens, repeats=self.num_heads, dim=0)
#
#         # output:(batch_size*num_heads,num_steps, num_hiddens/num_heads)
#         output = self.dot(keys, queries, values, valid_lens)
#
#         # output_concat的形状:(batch_size，查询的个数，num_hiddens)
#         output_concat = transpose_output(output, self.num_heads)
#         return self.W_o(output_concat)
#
# =============================================================================

# ==============================Model building=================================
#%% Model building

def grad_clipping(net, theta):
    """裁剪梯度
    """
    if isinstance(net, nn.Module):
        params,names = [],[]
        for name,p in net.named_parameters():
            if p.requires_grad:
                params.append(p)
                names.append(name)
    # params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    for p,n in zip(params,names):
        try:
            a = torch.sum((p.grad ** 2))
        except Exception as e:
            import sys
            sys.exit(f"{e}\nerror param:{n}\n{p}")

    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))

    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def grad_diagnose(net):
    if isinstance(net, nn.Module):
        params,names = [],[]
        for name,p in net.named_parameters():
            if p.requires_grad:
                params.append(p)
                names.append(name)
    else:
        params = net.params
    with open(r"PALACE_model_diagnose.grad.txt","w") as o:
        for p,n in zip(params,names):
            o.write(f"#####################{n}:#####################\ngrad:{p.grad}\n")

def transpose_qkv(X, num_heads):
    """
    change the input shape for multi-head attention
    """
    # X :(batch_size，num_steps，feat_space_dim)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # X :(batch_size，num_steps，num_heads，feat_space_dim/num_heads)

    # X:(batch_size，num_heads，num_steps,feat_space_dim/num_heads)
    X = X.permute(0, 2, 1, 3)

    # X:(batch_size*num_heads,num_steps,feat_space_dim/num_heads)
    X = X.reshape(-1, X.shape[2], X.shape[3])
    return X

def transpose_output(X, num_heads):
    """reverse transpose_qkv
    """
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项
    """
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def sequence_mask_v2(X, valid_len):
    """pytorch mask
    """
    mask = torch.zeros((X.shape[0],X.shape[1]),device = X.device)
    for i in torch.arange(mask.shape[0]):
        mask[i][valid_len[i]:] = 1
    #print(X)
    #print(valid_len)
    #print(mask)
    return mask

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数
    """
    def __init__(self, vocab_size,device):
        super(MaskedSoftmaxCELoss, self).__init__()
        # train loss weights, generated using prepare_CE_loss_weight.py
        self.weight = None
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len, epoch, diagnose):
        """
        print(f"label:{label}")
        print(f"label shape:{label.shape}")
        print(f"pred:{pred}")
        print(f"pred shape:{pred.shape}")
        print(f"valid_len: {valid_len}")
        """
        # weights 1 of (batch_size, num_steps)
        weights = torch.ones_like(label)
        """
        print(f"weights:{weights}")
        print(f"weights shape:{weights.shape}")
        """
        # valid_len include <eos>
        # set weights of padding locations to 0
        weights = sequence_mask(weights, valid_len)
        """
        print(f"weights after:{weights}")
        print(f"weights after shape:{weights.shape}")
        """
        # reduction: 'none' | 'mean' | 'sum'.
        self.reduction = 'none'
        # permute here to let class in middle (asked by pytorch)
        # unweighted_loss: (batch_size, num_steps)
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
                pred.permute(0, 2, 1), label)
        """
        print(f"unweighted_loss :{unweighted_loss}")
        print(f"unweighted_loss shape:{unweighted_loss.shape}")
        """
        # weighted_loss: batch_size
        # original d2l was wrong, you cannot just mean over num_steps, which will
        # make net learn to predict as long tensor as possible
        # weighted_loss = (unweighted_loss * weights).mean(dim=1)
        weighted_loss = (unweighted_loss * weights).sum(dim=1)
        """
        weighted_loss = (unweighted_loss * weights)
        weighted_loss_valid = (weighted_loss[0].sum() / valid_len[0]).unsqueeze(0)
        for i,(seq,val) in enumerate(zip(weighted_loss[1:],valid_len[1:])):
            loss_ = (seq.sum() / val).unsqueeze(0)
            weighted_loss_valid = torch.cat((weighted_loss_valid, loss_),0)
        """
        # weighted_loss_valid = torch.tensor(weighted_loss_valid, device = pred.device,requires_grad=True)
        # print(f"weighted_loss_valid :{weighted_loss_valid}")
        # print(f"weighted_loss_valid shape:{weighted_loss_valid.shape}")

        """
        print(f"weighted_loss after:{weighted_loss}")
        print(f"weighted_loss after shape:{weighted_loss.shape}")
        """
        if diagnose == True:
            if epoch == 500 :
                with open(r'PALACE_model_diagnose.loss.txt','w') as w:
                    w.write("########################### pred ###########################\n")
                    w.write(str(pred))
                    w.write(f"\npred shape:{pred.shape}")
                    w.write("\n########################### label ###########################\n")
                    w.write(str(label))
                    w.write(f"\nlabel shape:{label.shape}")
                    w.write("\n########################### valid_len ###########################\n")
                    w.write(str(valid_len))
                    w.write(f"\nvalid_len shape:{valid_len.shape}")
                    w.write("\n########################### weights ###########################\n")
                    w.write(str(weights))
                    w.write("\n########################### unweighted_loss ###########################\n")
                    w.write(str(unweighted_loss))
                    w.write("\n########################### weighted_loss ###########################\n")
                    w.write(str(weighted_loss))
                    try:
                        w.write("\n########################### weighted_loss_valid ###########################\n")
                        w.write(str(weighted_loss_valid))
                    except:
                        pass

        # return weighted_loss_valid
        return weighted_loss

class PositionalEncoding(nn.Module):
    """位置编码
    """
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        printer("PositionalEncoding","output",X.shape)
        return self.dropout(X)

class DotProductMixAttention(nn.Module):
    """Actually is the same as DotProductAttention
    """
    def __init__(self, dropout, **kwargs):
        super(DotProductMixAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k, q, v, valid_lens=None):
        # when is_cross: k,q,v = X_prot, X_smi, X_prot
        # k, v:tensor([batch_size*num_heads, num_steps, num_hiddens/num_heads])
        # q: during training: (batch_size*num_heads, num_steps, num_hiddens/num_heads)
        # q: during predicting: (batch_size*num_heads, varies, num_hiddens/num_heads)
        printer("DotProductMixAttention","k",k.shape)
        printer("DotProductMixAttention","v",v.shape)
        assert k.shape == v.shape, "k,v should have same dimensions."
        d = q.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(q, k.transpose(1,2)) / math.sqrt(d)

        def masked_softmax(X, valid_lens):
            """Perform softmax operation by masking elements on the last axis.
            """
            # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
            if valid_lens is None:
                return nn.functional.softmax(X, dim=-1)
            else:
                shape = X.shape
                if valid_lens.dim() == 1:
                    valid_lens = torch.repeat_interleave(valid_lens, shape[1])
                else: valid_lens = valid_lens.reshape(-1)
                # On the last axis, replace masked elements with a very large negative
                # value, whose exponentiation outputs 0
                X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,value=-1e6)
                return nn.functional.softmax(X.reshape(shape), dim=-1)

        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), v)

class DotProductAttention(nn.Module):
    """Scaled dot product attention.

    Defined in :numref:`subsec_additive-attention`"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of `keys` with `keys.transpose(1,2)`
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        def masked_softmax(X, valid_lens):
            """通过在最后一个轴上掩蔽元素来执行softmax操作
            """
            # X:3D张量，valid_lens:1D或2D张量
            if valid_lens is None:
                return nn.functional.softmax(X, dim=-1)
            else:
                shape = X.shape
                if valid_lens.dim() == 1:
                    valid_lens = torch.repeat_interleave(valid_lens, shape[1])
                else: valid_lens = valid_lens.reshape(-1)
                # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
                X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,value=-1e6)
                return nn.functional.softmax(X.reshape(shape), dim=-1)

        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

class MultiHeadAttention(nn.Module):
    """多头注意力
    """
    def __init__(self,feat_space_dim, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.dot = DotProductMixAttention(dropout)
        # nn.Linear can accept N-D tensor
        # ref https://stackoverflow.com/questions/58587057/multi-dimensional-inputs-in-pytorch-linear-method
        self.W_q = nn.Linear(feat_space_dim, feat_space_dim, bias=bias)
        self.W_k = nn.Linear(feat_space_dim, feat_space_dim, bias=bias)
        self.W_v = nn.Linear(feat_space_dim, feat_space_dim, bias=bias)
        self.W_o = nn.Linear(feat_space_dim, feat_space_dim, bias=bias)

    def forward(self, keys, queries, values, valid_lens):
        # when is_cross: k,q,v = X_prot, X_smi, X_prot
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, `num_steps`, `num_hiddens`)
        # Shape of `valid_lens`:
        # (`batch_size`,) or (`batch_size`, no. of queries)
        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,`num_hiddens` / `num_heads`)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        printer("MultiHeadAttention","keys",keys.shape)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # output:(batch_size*num_heads,num_steps, num_hiddens/num_heads)
        output = self.dot(keys, queries, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

class MultiHeadExternalMixAttention(nn.Module):
    """
    Multi-head attention with external input and internal input.
    the feature dimension of k,q,v are all set to feat_space_dim
    """
    def __init__(self, feat_space_dim, num_heads, prot_nota_len, dropout, bias=False, **kwargs):
        super(MultiHeadExternalMixAttention, self).__init__(**kwargs)

        # nn.Linear can accept N-D tensor
        # ref https://stackoverflow.com/questions/58587057/multi-dimensional-inputs-in-pytorch-linear-method
        self.W_mix_k = nn.Linear(prot_nota_len,feat_space_dim, bias=bias)
        self.W_mix_q = nn.Linear(prot_nota_len,feat_space_dim, bias=bias)
        self.W_k = nn.Linear(feat_space_dim, feat_space_dim, bias=bias)
        self.W_q = nn.Linear(feat_space_dim, feat_space_dim, bias=bias)
        self.W_v = nn.Linear(feat_space_dim, feat_space_dim, bias=bias)
        self.W_o = nn.Linear(feat_space_dim, feat_space_dim, bias=bias)
        self.dot = DotProductMixAttention(dropout)

        self.num_heads = num_heads

    def forward(self, keys, queries, values, valid_lens):
        # k, q, v = (X_prot,X_smi), (X_prot,X_smi), X_smi
        # X_prot: tensor([batch_size,prot_nota_len,feat_space_dim])
        # X_smi: tensor([batch_size,num_steps,feat_space_dim])
        # valid_lens is src_valid_len: ([N]): tensor([valid_len_for_each_sample])
        # output k,q,v: (batch_size*num_heads，num_steps, feat_space_dim/num_heads)

        key_prot, key_smi = keys
        query_prot, query_smi = queries
        printer("MultiHeadExternalMixAttention",'key_prot',key_prot.shape)
        printer("MultiHeadExternalMixAttention",'key_smi',key_smi.shape)

        # key_mix: tensor([batch_size,num_steps,prot_nota_len])
        # query_mix is similar
        key_mix = torch.bmm(key_smi,key_prot.transpose(1,2))
        query_mix = torch.bmm(query_smi,query_prot.transpose(1,2))

        # key_mix: from tensor([batch_size,num_steps,prot_nota_len]) ->
        # tensor([batch_size,num_steps,feat_space_dim])
        # query_mix is similar
        key_mix = self.W_mix_k(key_mix)
        query_mix = self.W_mix_q(query_mix)

        # dimension remain tensor([batch_size,num_steps,feat_space_dim])
        # just to fit the attention
        key = self.W_k(key_mix)
        query = self.W_q(query_mix)
        #transpose_qkv change X: from (batch_size，num_steps，feat_space_dim)
        # -> (batch_size*num_heads,num_steps,feat_space_dim/num_heads)
        key = transpose_qkv(key, self.num_heads)
        query = transpose_qkv(query, self.num_heads)
        printer("MultiHeadExternalMixAttention",'key',key.shape,"transpose_qkv")
        printer("MultiHeadExternalMixAttention",'query',query.shape,"transpose_qkv")

        # values: from tensor([batch_size,num_steps,feat_space_dim]) ->
        # (batch_size*num_heads,num_steps,feat_space_dim/num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # output:tensor([batch_size*num_heads, num_steps, feat_space_dim/num_heads])
        output = self.dot(key, query, values, valid_lens)
        printer("MultiHeadExternalMixAttention",'output',output.shape,'DotProductMixAttention')
        # output_concat:tensor([batch_size, num_steps, feat_space_dim])
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络, change the last dimension"""
    def __init__(self, feat_space_dim, ffn_num_hiddens, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(feat_space_dim, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, feat_space_dim)

    def forward(self, X):
        """ simply propagate forward """
        return self.dense2(self.relu(self.dense1(X)))

class PositionWiseFFN_v2(nn.Module):
    """基于位置的前馈网络, change the last dimension"""
    def __init__(self, feat_space_dim, ffn_num_hiddens, **kwargs):
        super(PositionWiseFFN_v2, self).__init__(**kwargs)
        self.dense1 = nn.Linear(feat_space_dim, feat_space_dim)
        self.relu = nn.ReLU()

    def forward(self, X):
        """ simply propagate forward """
        return self.relu(self.dense1(X))

class AddNorm(nn.Module):
    """残差连接后进行层规范化,
    drop out, add and layer normalization"""
    def __init__(self, feat_space_dim, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        # LayerNorm doesn't change the shape of input
        self.ln = nn.LayerNorm(feat_space_dim)

    def forward(self, X, Y):
        """ add """
        return self.ln(self.dropout(Y) + X)

class ProteinEncoding_large(nn.Module):
    """
    protein encoding block.
    """
    def __init__(self,feat_space_dim, prot_MLP, dropout, **kwargs):
        super(ProteinEncoding_large, self).__init__(**kwargs)
        printer("ProteinEncoding","prot_MLP",prot_MLP,"__init__")
        self.dense1 = nn.Linear(45, prot_MLP[0])
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(prot_MLP[0], feat_space_dim)
        self.relu2 = nn.ReLU()
        self.ln = nn.LayerNorm(feat_space_dim)
        self.dropout = nn.Dropout(dropout)

class ProteinEncoding(nn.Module):
    """
    protein encoding block.
    """
    def __init__(self,feat_space_dim, prot_MLP, dropout, **kwargs):
        super(ProteinEncoding, self).__init__(**kwargs)
        printer("ProteinEncoding","prot_MLP",prot_MLP,"__init__")
        self.dense = nn.Linear(feat_space_dim, feat_space_dim)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(feat_space_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        """
        X: torch.tensor([batch_size,45,feat_space_dim])
        output: torch.tensor([batch_size,45,feat_space_dim])
        """
        assert X is not None, "ProteinEncoding input cannot be None."

        return self.ln(self.dropout(self.relu(self.dense(X))))


class PALACE_prot_net(nn.Module):
    """
    prot_net
    """
    def __init__(self, prot_vocab_size, feat_space_dim, prot_nota_len, dropout, **kwargs):
        super(PALACE_prot_net, self).__init__(**kwargs)

        self.embedding = nn.Embedding(prot_vocab_size, feat_space_dim)
        self.dense1 = nn.Linear(feat_space_dim, feat_space_dim)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(feat_space_dim,feat_space_dim)
        self.relu2 = nn.ReLU()

    def forward(self, X, valid_lens, *args):
        # X: tensor([batch_size,2500])
        # valid_lens: tensor([N])
        X = self.embedding(X)
        # X: tensor([batch_size,2500,feat_space_dim])
        valid_lens = valid_lens.cpu().numpy()
        # paddings set to zero
        for i,v in enumerate(valid_lens):
            X[i][v:] = 0
        # sum to feat_space_dim, eliminate protein lens varies
        X = X.sum(dim=1)
        # X output: [batch_size,prot_nota_len]
        return self.relu2(self.dense2(self.dropout(self.relu1(self.dense1(X)))))

class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError

class EncoderBlock(nn.Module):
    """transformer编码器块"""
    def __init__(self, feat_space_dim, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)

        self.attention = MultiHeadAttention(
                feat_space_dim, num_heads,dropout,use_bias)
        self.addnorm1 = AddNorm(feat_space_dim, dropout)
        self.ffn = PositionWiseFFN(feat_space_dim, ffn_num_hiddens)
        self.addnorm2 = AddNorm(feat_space_dim, dropout)
        #self.addnorm3 = AddNorm(feat_space_dim, dropout)

    def forward(self, X, valid_lens,is_cross = False):
        # X : tensor([batch_size,num_steps,feat_space_dim])
        # k and q will be the mix of X_prot and X_smi
        # v is just X_smi
# =============================================================================
#         if is_cross:
#             batch_size = int(X.shape[0] / 2)
#             X_prot, X_smi = X[:batch_size], X[batch_size:]
#             k,q,v = X_prot, X_smi, X_prot
#         else:
#             k, q, v = X, X, X
# =============================================================================
        k, q, v = X, X, X
        # X: tensor([batch_size,num_steps,feat_space_dim])
        X = self.attention(k, q, v, valid_lens)
        printer("EncoderBlock","X",X.shape,"MultiHeadAttention")

        # X: tensor([batch_size,num_steps,feat_space_dim])
        X = self.addnorm1(X, X)
        printer("EncoderBlock","X",X.shape,"AddNorm")

        # return self.addnorm3(self.addnorm2(X, X),X_smi)
        return self.addnorm2(self.ffn(X), X)

class PALACE_Encoder(Encoder):
    """
    Encoder
    """
    def __init__(self,vocab_size, feat_space_dim, ffn_num_hiddens, num_heads,
                 num_blks, dropout, is_cross = False, is_prot = False, use_bias=False, num_steps = 0,**kwargs):
        super(PALACE_Encoder, self).__init__(**kwargs)
        self.feat_space_dim = feat_space_dim
        self.is_cross = is_cross
        self.is_prot = is_prot
        printer("PALACE_Encoder",'vocab_size',vocab_size)
        printer("PALACE_Encoder",'feat_space_dim',feat_space_dim)

        if not is_cross:
            self.embedding = nn.Embedding(vocab_size, feat_space_dim)
            if not is_prot:
                self.pos_encoding = PositionalEncoding(feat_space_dim, dropout)
            else:
                self.num_steps = num_steps
                self.conv1d = nn.Conv1d(2500,num_steps,1,stride=1)
                self.ffn = PositionWiseFFN(feat_space_dim, ffn_num_hiddens)
                self.addnorm = AddNorm(feat_space_dim,dropout)

        # encoder block
        if not is_prot:
            self.blks = nn.Sequential()
            for i in range(num_blks):
                self.blks.add_module("block"+str(i),
                    EncoderBlock(feat_space_dim, ffn_num_hiddens, num_heads,
                                 dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        # X: tensor([batch_size,num_steps])
        is_cross = self.is_cross
        # print(is_cross)
        if not is_cross:
            X = self.embedding(X)
            if not self.is_prot:
                X = self.pos_encoding(X * math.sqrt(self.feat_space_dim))
            else:
                X = self.conv1d(X)

        # X: from tensor([batch_size,num_steps]) -> tensor([batch_size,num_steps,feat_space_dim])
        if not self.is_prot:
            self.attention_weights = [None] * len(self.blks)
            for i, blk in enumerate(self.blks):
                # return of blk = X: tensor([batch_size,num_steps,feat_space_dim])
                X = blk(X, valid_lens, is_cross)
                # after the first cross attention blk, the k,q,v became the same
                is_cross = False
                self.attention_weights[i] = blk.attention.dot.attention_weights
        else:
            X = self.addnorm(self.ffn(X),X)
            # print(X)

        printer("PALACE_Encoder",'output',X.shape)
        # the X_smi is not the original X_smi anymore
        return X


class DecoderBlock(nn.Module):
    """解码器中第i个块"""
    def __init__(self,feat_space_dim, ffn_num_hiddens, num_heads,
                 dropout, dec_rank,use_bias, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)

        self.dec_rank = dec_rank
        self.attention1 = MultiHeadAttention(feat_space_dim, num_heads, dropout,use_bias)
        self.addnorm1 = AddNorm(feat_space_dim, dropout)

        self.attention2 = MultiHeadAttention(feat_space_dim, num_heads, dropout)
        self.addnorm2 = AddNorm(feat_space_dim, dropout)
        self.ffn = PositionWiseFFN(feat_space_dim, ffn_num_hiddens)
        self.addnorm3 = AddNorm(feat_space_dim, dropout)
        #self.addnorm4 = AddNorm(feat_space_dim, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.dec_rank]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.dec_rank]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.dec_rank] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.dec_rank], X), axis=1)
        state[2][self.dec_rank] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens: (batch_size,num_steps),
            # each row is: [1,2,...,num_steps]
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # 自注意力
        k,q,v = key_values, X, key_values
        X2 = self.attention1(k,q,v, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器－解码器注意力。
        # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
        k,q,v = enc_outputs,Y,enc_outputs
        Y2 = self.attention2(k,q,v, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

class PALACE_Decoder(Decoder):
    def __init__(self, vocab_size, feat_space_dim, ffn_num_hiddens, num_heads,
                 num_blks, dropout,use_bias=False, **kwargs):
        super(PALACE_Decoder, self).__init__(**kwargs)
        self.feat_space_dim = feat_space_dim
        printer("PALACE_Decoder",'vocab_size',vocab_size)
        printer("PALACE_Decoder",'feat_space_dim',feat_space_dim)

        self.embedding = nn.Embedding(vocab_size, feat_space_dim)
        self.pos_encoding = PositionalEncoding(feat_space_dim, dropout)

        self.blks = nn.Sequential()
        for dec_rank in range(num_blks):
            self.blks.add_module("block"+str(dec_rank),
                DecoderBlock(feat_space_dim, ffn_num_hiddens, num_heads,
                 dropout, dec_rank,use_bias))

        self.dense = nn.Linear(feat_space_dim, vocab_size)
        self.softmax = nn.Softmax(dim=2)

        self.num_blks = num_blks
        self.feat_space_dim = feat_space_dim

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def forward(self, X, state):
        # dec_input is the right shifted X_smi with force teaching
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。

        # X: tensor([batch_size,num_steps])
        X = self.embedding(X)
        # X: from tensor([batch_size,num_steps]) -> tensor([batch_size,num_steps,feat_space_dim])
        X = self.pos_encoding(X * math.sqrt(self.feat_space_dim))

        # _attention_weights: (2,num_blks)
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]

        # init state: [enc_outputs, enc_valid_lens, [None] * self.num_blks]
        # each decoder block, will take in same enc_outputs and enc_valid_lens
        for i, blk in enumerate(self.blks):
            #printer("PALACE_Decoder","state",state,i)
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][i] = blk.attention1.dot.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][i] = blk.attention2.dot.attention_weights
        return self.softmax(self.dense(X)), state

    @property
    def attention_weights(self):
        return self._attention_weights



class PALACE(nn.Module):
    """编码器-解码器架构的基类
    """
    def __init__(self, smi_encoder, prot_encoder, cross_encoder, decoder,feat_space_dim,ffn_num_hiddens,dropout, **kwargs):
        super(PALACE, self).__init__(**kwargs)
        self.smi_encoder = smi_encoder
        self.prot_encoder = prot_encoder
        self.cross_encoder = cross_encoder
        self.ffn = PositionWiseFFN(feat_space_dim, ffn_num_hiddens)
        self.addnorm = AddNorm(feat_space_dim, dropout)
        self.decoder = decoder

    def forward(self, X, dec_X, valid_lens, *args):
        # Y_hat, _ = net((X_prot, X), dec_input, (prot_valid_len,X_valid_len))
        # X_prot: tensor([batch_size,2500])
        # X_smi: tensor([batch_size,num_steps])
        X_prot, X_smi = X[0],X[1]
        prot_valid_lens,enc_valid_lens = valid_lens[0],valid_lens[1]
        # encoder outputs: tensor([batch_size,num_steps,feat_space_dim])
        X_prot = self.prot_encoder(X_prot, prot_valid_lens, *args)
        X_smi = self.smi_encoder(X_smi, enc_valid_lens, *args)
        #enc_outputs = self.cross_encoder(torch.cat((X_prot,X_smi),0), enc_valid_lens, *args)
        # X_mix = X_prot * X_smi
        X_mix = X_prot + X_smi
        # X_mix = self.addnorm(self.ffn(X_mix),(X_mix+X_prot+X_smi))
        X_mix = self.addnorm(self.ffn(X_mix),(X_mix+X_prot+X_smi) / torch.tensor(3))
        enc_outputs = self.cross_encoder(X_mix, enc_valid_lens, *args)
        printer('PALACE','enc_outputs',enc_outputs.shape)

        # dec_state: [enc_outputs, enc_valid_lens, [None] * self.num_blks]
        # enc_valid_lens = src_valid_len: ([N]): tensor([valid_len_for_each_sample])
        dec_state = self.decoder.init_state(enc_outputs,enc_valid_lens, *args)
        #printer("PALACE","dec_state",dec_state)

        return self.decoder(dec_X, dec_state)

# ===============================Preprocess====================================
#%% Preprocess


class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = self.count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        # 回复调用自身，使得即使输入是list/tuple也能带着[]返回0
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

    def count_corpus(self,tokens):
        """统计词元的频率
        """
        # 这里的tokens是1D列表或2D列表
        if len(tokens) == 0 or isinstance(tokens[0], list):
            # 将词元列表展平成一个列表
            tokens = [token for line in tokens for token in line]
        return Counter(tokens)


def save_vocab(vocab,vocab_dir):
    """
    save vocab class using pickle
    """
    with open(vocab_dir,'wb') as o:
        pickle.dump(vocab,o,pickle.HIGHEST_PROTOCOL)

def retrieve_vocab(vocab_dir):
    """
    retrieve vocab class using pickle
    """
    with open(vocab_dir,'rb') as r:
        return pickle.load(r)

def creat_vocab(file_path = './prot_models/embedding/vocab.txt'):
    """
    file_path = './prot_models/embedding/vocab.txt'
    """
# =============================================================================
#     import io
#     from torchtext.vocab import build_vocab_from_iterator
#
#     def yield_tokens(file_path):
#         with io.open(file_path, encoding = 'utf-8') as f:
#             for line in f:
#                 yield line.strip().split()
#     vocab = build_vocab_from_iterator(yield_tokens(file_path), specials=["<unk>",'<pad>'])
#     vocab.set_default_index(vocab["<unk>"])
#     return vocab
# =============================================================================
    with open(file_path,'r') as r:
        source = [[x.strip()] for x in r.readlines()]
    vocab = Vocab(source, min_freq=0,reserved_tokens=['<pad>'])
    vocab
    save_vocab(vocab,r'./vocab/prot_vocab.pkl')
    return vocab

def prot_to_features_bert(prot : list, trained_model_dir: str, device: str, prot_read_batch_size: int):
    """
    call pre-trained BERT to generate prot features
    input: prot: [prot_seq_sample1,prot_seq_sample2,...,prot_seq_sampleN]
    output: prot_feature: torch.Size([N, 45])
    """
    assert os.path.exists(trained_model_dir) == True
    # get model
    model = BertForMaskedLM.from_pretrained(trained_model_dir)
    tokenizer = BertTokenizer.from_pretrained(trained_model_dir)

    # model to GPU
    model = model.to(device)
    model.eval()

    # tokenize prot
    # aa is amino acid
    # replace unknown aa to X and inter-fill with space
    prot_tok = [' '.join([re.sub(r"[UZOB]", "X", aa) for aa in list(seq)]) for seq in prot]
    prot_tok_list = [prot_tok[i:i+prot_read_batch_size] for i in range(0, len(prot_tok), prot_read_batch_size)]
    prot_features = []
    with torch.no_grad():
        for index,batch in enumerate(prot_tok_list):
            # Tokenize, encode sequences and load it into the GPU if possibile
            ids = tokenizer.batch_encode_plus(batch, add_special_tokens=True, padding=True)
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)
            # reshape into batch
            seq_num,seq_len = input_ids.shape
            try:
                embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]
            except Exception as e:
                #import sys
                #sys.exit("{}.\nprot_to_features failed at {} batch.".format(e,index))
                print("=====================prot_to_features failed at {} batch with batch_size {}.=================".format(index,prot_read_batch_size))
                print("====================={}=================".format(e))
                #prot_num = index * batch_size
                break # memory restriction, can only get these protein features
            # Remove padding ([PAD]) and special tokens ([CLS],[SEP])
            # that is added by ProtBert-BFD model
            for seq_num in range(len(embedding)):
                # remove padding
                seq_len = (attention_mask[seq_num] == 1).sum()
                # remove special tokens
                seq_emd = embedding[seq_num][1:seq_len-1]
                prot_features.append(seq_emd)

    #prot_features = torch.stack([torch.tensor(x).sum(dim=0) for x in prot_features ])
    del model,tokenizer
    return torch.stack([torch.tensor(x).sum(dim=0) for x in prot_features ])

def prot_to_features_prose(prot : list, device: str):
    """
    call pre-trained BERT to generate prot features
    input: prot: [prot_seq_sample1,prot_seq_sample2,...,prot_seq_sampleN]
    output: prot_feature: torch.Size([N, 45])
    """
    from prot_models.prose.multitask import ProSEMT

    class Uniprot21:
        def __init__(self):
            missing = 20
            self.chars = np.frombuffer(b'ARNDCQEGHILKMFPSTWYVXOUBZ', dtype=np.uint8)

            self.encoding = np.zeros(256, dtype=np.uint8) + missing

            encoding = np.arange(len(self.chars))
            encoding[21:] = [11,4,20,20] # encode 'OUBZ' as synonyms

            self.encoding[self.chars] = encoding
            self.size = encoding.max() + 1
            self.mask = False

        def __len__(self):
            return self.size

        def __getitem__(self, i):
            return chr(self.chars[i])

        def encode(self, x):
            """ encode a byte string into alphabet indices """
            x = np.frombuffer(x, dtype=np.uint8)
            return self.encoding[x]

        def decode(self, x):
            """ decode index array, x, to byte string of this alphabet """
            string = self.chars[x]
            return string.tobytes()

        def unpack(self, h, k):
            """ unpack integer h into array of this alphabet with length k """
            n = self.size
            kmer = np.zeros(k, dtype=np.uint8)
            for i in reversed(range(k)):
                c = h % n
                kmer[i] = c
                h = h // n
            return kmer

        def get_kmer(self, h, k):
            """ retrieve byte string of length k decoded from integer h """
            kmer = self.unpack(h, k)
            return self.decode(kmer)

    model = ProSEMT.load_pretrained()

    # model to GPU
    model.to(device)
    model.eval()

    prot_features = []
    with torch.no_grad():
        alphabet = Uniprot21()
        #for seq in tqdm(prot):
        for seq in prot:
            #print('featuring: {}'.format(seq))
            seq = seq.encode() # to bytes
            if len(seq) == 0:
                n = model.embedding.proj.weight.size(1)
                z = np.zeros((1,n), dtype=np.float32)
            else:
                seq = seq.upper()
                # convert to alphabet index
                seq = alphabet.encode(seq)
                seq = torch.from_numpy(seq)
                seq = seq.to(device)
                seq = seq.long().unsqueeze(0)
                z = model.transform(seq)
                z = z.squeeze(0)
                z = z.sum(dim=0) # size (6165)
                prot_features.append(z)
    # reshape to be torch.Size([N, 45])
    return torch.stack(prot_features).reshape((len(prot_features),-1,45)).mean(dim = 1)

# 2 3 4
def read_data(data_dir, device, prot_col = 2, src_col = 3, tgt_col = 4, sep = '\t'):
    """
    read text file and return prot_features, source, target
    let's say in total N samples in text file,
    prot_feature: torch.Size([N, 45])
    prot: [prot_seq_sample1,prot_seq_sample2,...,prot_seq_sampleN]
    source: [[tok1,tok2...],[tok3,tok2,...],...until_N]
    """
    prot = []
    source = []
    target = []
    give_up_pre_trained = True
    with open(data_dir,'r',encoding='utf-8') as r:
        for line in r:
            line_split = line.strip().split(sep)
            if not give_up_pre_trained:
                prot.append(line_split[prot_col])
            else: prot.append(list(line_split[prot_col].upper()))
            source.append(line_split[src_col].split(' '))
            target.append(line_split[tgt_col].split(' '))
            # prot.append(line_split[prot_col].split(' '))
    # let's say in total N samples
    # prot: [prot_seq_sample1,prot_seq_sample2,...,prot_seq_sampleN]
    # source: [[tok1,tok2...],[tok3,tok2,...],...until_N]
    # target: simliar as source

    # due to speed, give using pre-trained protein model
    if not give_up_pre_trained:
        prot_features = prot_to_features_prose(prot, device)
        printer('read_data','prot_features',prot_features.shape,'prot_to_features')

        # prot_feature: torch.Size([N, 45])
        # 45 is the feature dimension for each amino acid
        # prot_features compress all amino acids to feature direction

        # truncate
        if len(source) > len(prot_features):
            source = source[:len(prot_features)]
            target = target[:len(prot_features)]

        return prot_features, source, target
    return prot,source,target

def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列
    """
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

def build_array_nmt(lines, vocab, num_steps, add_eos = True):
    """将机器翻译的文本序列转换成小批量
    """
    lines = [vocab[l] for l in lines] # tokens to idx
    if add_eos:
        lines = [l + [vocab['<eos>']] for l in lines] # add idx of <eos> to each line
    # truncate or padding using idx of <pad> to be num_steps length
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    # figure out the actual length of each line (remove the padding ones)
    valid_len = reduce_sum(
        astype(array != vocab['<pad>'], torch.int32), 1)
    return array, valid_len


def load_data(rank, world_size,data_dir,batch_size, num_steps, device, vocab_dir):
    """返回翻译数据集的迭代器和词表
        vocab_dir: [src_vocab,tgt_vocab]
    """
    # prot_feature: torch.Size([N, 45])
    # prot: [[aa1,aa2,...],...N]
    # source: [[tok1,tok2...],[tok3,tok2,...],...until_N]
    # prot_features, source, target = read_data(data_dir, device)
    prot, source, target = read_data(data_dir, device)

    if vocab_dir:
        assert os.path.exists(vocab_dir[0]) and os.path.exists(vocab_dir[1]) and os.path.exists(vocab_dir[2]),"cannot find available vocab"
        src_vocab = retrieve_vocab(vocab_dir[0])
        tgt_vocab = retrieve_vocab(vocab_dir[1])
        prot_vocab = retrieve_vocab(vocab_dir[2])
    else:
        src_vocab = Vocab(source, min_freq=0,reserved_tokens=['<pad>', '<bos>', '<eos>'])
        tgt_vocab = Vocab(target, min_freq=0,reserved_tokens=['<pad>', '<bos>', '<eos>'])
        prot_vocab = Vocab(prot, min_freq=0,reserved_tokens=['<pad>', '<bos>', '<eos>'])
        merge_vocab = Vocab(source + target,min_freq=0,reserved_tokens=['<pad>', '<bos>', '<eos>'])
        save_vocab(src_vocab,'./vocab/src_vocab.pkl')
        save_vocab(tgt_vocab,'./vocab/tgt_vocab.pkl')
        save_vocab(prot_vocab,'./vocab/prot_vocab.pkl')
        save_vocab(merge_vocab,'./vocab/merge_vocab.pkl')

    # src_array: ([N,num_steps]): tensor([[tok1_to_idx,tok2_to_idx,...,tok_num_steps_to_idx],[...],...])
    # src_valid_len: ([N]): tensor([valid_len_for_each_sample])
    # tgt_array and tgt_valid_len are similar to src
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    prot_array, prot_valid_len = build_array_nmt(prot, prot_vocab, 2500, add_eos = False) # maxium prot is 2500

    # the first dim of the following are all N (total number of samples)
    # data_arrays = (prot_features, src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_arrays = (prot_array, prot_valid_len, src_array, src_valid_len, tgt_array, tgt_valid_len)

    def load_array(rank, world_size,data_arrays, batch_size,pin_memory=False, num_workers=0):
        """构造一个PyTorch数据迭代器
        """
        dataset = data.TensorDataset(*data_arrays)
        sampler = DistributedSampler(dataset, num_replicas=world_size,\
                                     rank=rank, shuffle=False, drop_last=False)
        dataloader = data.DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, \
                                num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
        return dataloader

    data_iter = load_array(rank, world_size,data_arrays, batch_size)

    return data_iter, src_vocab, tgt_vocab, prot_vocab


# =================================GPUs=======================================
#%% GPUs
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def set_random_seed(seed, is_cuda):
    """Sets the random seed.
    为什么使用相同的网络结构，跑出来的效果完全不同，用的学习率，迭代次数，batch size 都是一样？
    固定随机数种子是非常重要的。但是如果你使用的是PyTorch等框架，还要看一下框架的种子是否固定了。
    还有，如果你用了cuda，别忘了cuda的随机数种子。这里还需要用到torch.backends.cudnn.deterministic.
    """
    if seed > 0:
        torch.manual_seed(seed)
        # this one is needed for torchtext random call (shuffled iterator)
        # in multi gpu it ensures datasets are read in the same order
        random.seed(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    if is_cuda and seed > 0:
        # These ensure same initialization in multi gpu mode
        torch.cuda.manual_seed(seed)

def assign_gpu(rank):
    printer(f"=======================PALACE: assigning GPU{rank}...=======================",print_=True)
    #local_rank = torch.distributed.get_rank()
    #torch.distributed.init_process_group(backend="nccl", rank=local_rank)
    if rank >= 0:
        return torch.device(f'cuda:{rank}')
    return torch.device('cpu')

# setup the process groups
def setup_gpu(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # nccl for linux, gloo for windows
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
# =================================Utils=======================================
#%% Utils
class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Timer:
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

def printer(function,instance = None,content = None ,after = None, print_=False,print_shape=False):
    """
    'function': 'instance' after 'after' is: 'content'
    """
    logging.basicConfig(filename='PALACE.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s')
    try:
        if print_shape:
            if after:
                logging.warning("{}: {} after {} is: {}".format(function,instance,after,content))
            elif instance:
                logging.warning("{}: {} is: {}".format(function,instance,content))
            else:
                logging.warning("{}".format(function))
            logging.warning(time.localtime())
        if print_:
            if after:
                print("{}: {} after {} is: {}\n".format(function,instance,after,content))
            elif instance:
                print("{}: {} is: {}\n".format(function,instance,content))
            else:
                print("{}\n".format(function))
            #print(time.localtime())
    except Exception as e: print(e)

def model_compare(dict1,dict2):
    for k,v in dict1.items():
        if torch.equal(dict1[k], dict2[k]):
            print(k)



def use_svg_display():
    """Use the svg format to display a plot in Jupyter.
    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """Show heatmaps of matrices.
    Defined in :numref:`sec_attention-cues`"""
    use_svg_display()
    num_rows, num_cols = len(matrices), len(matrices[0])
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(numpy(matrix), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)

def model_diagnose(model_id):
    from modules import model_compare
    dict1 = torch.load(f'./PALACE_models/init_{model_id}.pt', map_location='cpu')['net']
    dict2 = torch.load('./PALACE_models/checkpoint_{}.pt'.format(model_id), map_location='cpu')['net']
    model_compare(dict1,dict2)
    dist_log = {}
    with open(r'./PALACE_model_diagnose.model.txt','w') as w:
        for k in dict1.keys():
            w.write(f'{k} shape: {dict1[k].shape}\n')
            distance = (dict1[k]-dict2[k]).abs().mean()
            dist_log[k] = distance
            w.write(str(dict1[k]) + '\n')
            w.write(str(dict2[k]) + '\n')
            w.write('##########################################################\n')
        w.write(str({k: v for k, v in sorted(dist_log.items(), key=lambda item: item[1])}))

numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
transpose = lambda x, *args, **kwargs: x.t(*args, **kwargs)
reduce_mean = lambda x, *args, **kwargs: x.mean(*args, **kwargs)


# ===============================Training======================================
#%% Training
def init_weights(m,use = 'xavier_uniform_'):
    if use == 'xavier_normal_':
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
    if use == 'xavier_uniform_':
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    if use == 'kaiming_uniform_':
        if type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight,mode='fan_in', nonlinearity='relu')
    if use == 'orthogonal_':
        if type(m) == nn.Linear:
            nn.init.orthogonal_(m.weight)

def train_PALACE(piece,net, data_iter,optimizer,scheduler, loss, num_epochs, tgt_vocab,
                 device,loss_log, diagnose = False):
    """训练序列到序列模型
    """
    net.train()
    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2)  # 训练损失总和，词元数量
        #assert trained_model_dir is not None, "trained model is not available."
        correct = 0
        total_seq = 0
        data_iter.sampler.set_epoch(epoch)
        for i,batch in enumerate(data_iter):
            # print(f"batch: {batch}\n")
            optimizer.zero_grad()
            X_prot,prot_valid_len, X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # bos: torch.Size([batch_size, 1])
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],device=device).reshape(-1, 1)

            # dec_input: torch.Size([batch_size, num_steps])
            # removed the last tok in each sample of Y: (Y: [batch_size, num_steps-1])
            # add bos tok in begining of each sample of Y: (dec_input[batch_size, num_steps])
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # force teaching
            printer("train_PALACE:","X_prot",X_prot.shape)
            printer("train_PALACE:","X",X.shape)
            # Y_hat: (batch_size,num_steps,vocab_size)
            # Y: (batch_size,num_steps)
            Y_hat, _ = net([X_prot, X], dec_input, [prot_valid_len,X_valid_len])

            # loss and backward
            l = loss(Y_hat, Y, Y_valid_len, epoch, diagnose)
            l.sum().backward() # 损失函数的标量进行“反向传播”
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                loss_sum = l.sum()
                metric.add(loss_sum, num_tokens)

            # accuracy
            batch_size = Y_hat.shape[0]
            total_seq += batch_size
            num_steps = Y_hat.shape[1]
            preds = Y_hat.argmax(dim=2).type(torch.int32)
            for i in range(batch_size):
                p, y = preds[i], Y[i]
                if epoch % 90 == 1 and i == 1: # 8
                    print(f"p:{p}")
                    print(f"p shape:{p.shape}")
                    print(f"y:{y}")
                    print(f"y shape:{y.shape}")
                try: y_eos = (y == tgt_vocab['<eos>']).nonzero()[0].item()
                except: y_eos = num_steps
                try: p_eos = (p == tgt_vocab['<eos>']).nonzero()[0].item()
                except: p_eos = num_steps
                p_valid = p[:p_eos]
                y_valid = y[:y_eos]
                if epoch % 90 == 1 and i == 1:
                    print(f"p_valid:{p_valid}")
                    print(f"p_valid shape:{p_valid.shape}")
                    print(f"y_valid:{y_valid}")
                    print(f"y_valid shape:{y_valid.shape}")
                    print(f"lossCE weight:{loss.weight}")
                if torch.equal(p_valid, y_valid): correct += 1

        if diagnose and epoch % 30 == 0: grad_diagnose(net)

        scheduler.step(loss_sum)
        # scheduler.step()

        loss_ = metric[0] / metric[1]
        accuracy = correct / total_seq
        with open(loss_log,'a') as o:
            o.write(f"piece:{piece}\tepoch:{epoch}\tloss:{loss_:.8f}\taccuracy:{accuracy:.8f}\tlr:{optimizer.param_groups[0]['lr']}\ttokens/sec:{metric[1] / timer.stop():.1f}\tdevice: {str(device)}\n")
    dist.barrier()
# ==============================Prediction=====================================
#%% Prediction
def predict_PALACE(net, src, prot_vocab, src_vocab, tgt_vocab, num_steps,
                    device,beam,save_attention_weights=False):
    """Predict for sequence to sequence.
    """
    # Set `net` to eval mode for inference
    net.eval()

    # X_prot: seq1
    # X_smi: smi1
    # source tokens to ids and truncate/pad source length
    X_prot, X_smi = src
    X_smi, X_smi_valid_len = build_array_nmt(X_smi.split(' '), src_vocab, num_steps)
    X_prot, X_prot_valid_len = build_array_nmt(list(X_prot), prot_vocab, 2500, add_eos = False)
    # X_prot, X_smi became 2d tensor of idx
    X_smi.to(device)
    X_smi_valid_len.to(device)
    X_prot.to(device)
    X_prot_valid_len.to(device)
    # Add the batch axis
    X_prot = torch.unsqueeze(torch.tensor(X_prot, dtype=torch.float, device=device), dim=0)
    X_smi = torch.unsqueeze(torch.tensor(X_smi, dtype=torch.long, device=device), dim=0)
    printer("predict_PALACE","X_prot",X_prot.shape)
    printer("predict_PALACE","X_smi",X_smi.shape)

    # enc_X: (1,num_steps)
    enc_X = [X_prot,X_smi]
    X_prot_out = net.prot_encoder(X_prot,X_prot_valid_len)
    X_smi_out = net.smi_encoder(X_smi,X_smi_valid_len)

    X_mix = X_prot_out * X_smi_out
    X_mix = net.addnorm(net.ffn(X_mix),(X_mix+X_prot_out))

    enc_outputs = net.cross_encoder(enc_X, X_smi_valid_len)
    dec_init_state = net.decoder.init_state(enc_outputs, X_smi_valid_len)

    # Add the batch axis
    # dec_X: [[X]]
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)

    # initial beam search
    Y, dec_state = net.decoder(dec_X, dec_init_state)
    # Y_top: (1,beam)
    Y_top = torch.topk(Y,beam,dim=2,sorted = True).values.squeeze(dim=0)
    # idx_top: (1,beam)
    idx_top = torch.topk(Y,beam,dim=2,sorted = True).indices.squeeze(dim=0)
    # beam_Y_top: (1,beam)
    # beam_idx_top: (1,beam)
    beam_Y_top = Y_top.clone().detach()
    beam_idx_top = idx_top.clone().detach()
    attention_weight_seq = []
    eos_hit = [False for _ in range(beam)]
    for _ in range(num_steps):
        # for each top `beam` prediction likelihood, We use its token as input
        # of the decoder at the next time step
        Y_top_tmp = []
        idx_top_tmp = []
        for idx_loc,idx in enumerate(idx_top.squeeze(0)):
            dec_X = idx.unsqueeze(0).unsqueeze(0)
            # Y: (1,1,tgt_vocab_size) # batch_size, num_steps = 1
            Y, dec_state = net.decoder(dec_X, dec_state)
            Y_top = torch.topk(Y,1,dim=2,sorted = True).values.item()
            idx_top = torch.topk(Y,1,dim=2,sorted = True).indices.item()
            if idx_top == tgt_vocab['<eos>']:
                eos_hit[idx_loc] = True
            if sum(eos_hit) == beam: break # break from beam loop
            Y_top_tmp.append(Y_top)
            idx_top_tmp.append(idx_top)

        if sum(eos_hit) == beam: break # break from num_steps loop

        beam_Y_top = torch.cat([beam_Y_top,torch.tensor([Y_top_tmp]).to(device)],dim=0)
        beam_idx_top = torch.cat([beam_idx_top,torch.tensor([idx_top_tmp]).to(device)],dim=0)
        idx_top = torch.tensor([idx_top_tmp]).to(device)

        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)

    beam_Y_top = beam_Y_top[1:]
    beam_idx_top = beam_idx_top[1:] # remove the <bos>

    # translation
    beam_Y = beam_Y_top.sum(dim=0)
    beam_rank = torch.topk(beam_Y,beam,sorted = True).indices
    # rank of rank
    beam_rank_rank = torch.topk(beam_rank,beam,sorted = True,largest = False).indices
    beam_idx_top = beam_idx_top.transpose(0,1)[beam_rank_rank[:beam]].to('cpu').numpy()
    # truncate
    eos = tgt_vocab['<eos>']
    output_seq = []
    for idx in beam_idx_top:
        idx = list(idx)
        try:
            trunc_point = idx.index(eos)
            output_seq.append(idx[:trunc_point])
        except:
            output_seq.append(idx)

    translation = [tgt_vocab.to_tokens(list(seq)) for seq in output_seq]

    return translation
    #return ' '.join(translation[0]), attention_weight_seq
