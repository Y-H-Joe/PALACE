#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 11:21:05 2022

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
from modules import printer,prot_to_features,logging,try_gpu
# ==============================Prediction=====================================
#%% Prediction
def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences.
    """
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad

def predict_PALACE(net, src, src_vocab, tgt_vocab, num_steps,
                    device,num_pred,beam,save_attention_weights=False):
    """Predict for sequence to sequence.
    """
    # Set `net` to eval mode for inference
    net.eval()

    # source tokens to ids and truncate/pad source length
    X_prot, X_smi = src
    X_smi = src_vocab[X_smi.split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(X_smi)], device=device)
    X_smi = truncate_pad(X_smi, num_steps, src_vocab['<pad>'])

    # Add the batch axis
    X_prot = torch.unsqueeze(torch.tensor(X_prot, dtype=torch.float, device=device), dim=0)
    X_smi = torch.unsqueeze(torch.tensor(X_smi, dtype=torch.long, device=device), dim=0)
    printer("predict_PALACE","X_prot",X_prot.shape)
    printer("predict_PALACE","X_smi",X_smi.shape)

    # enc_X: (1,num_steps)
    enc_X = (X_prot,X_smi)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    #printer("predict_PALACE","dec_state",dec_state,"init_state")

    # Add the batch axis
    # dec_X: [[X]]
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)

    # initial beam search
    Y, dec_state = net.decoder((X_prot,dec_X), dec_state)
    # Y_top: (1,beam)
    Y_top = torch.topk(Y,beam,dim=2,sorted = True).values.squeeze(dim=0)
    # idx_top: (1,beam)
    idx_top = torch.topk(Y,beam,dim=2,sorted = True).indices.squeeze(dim=0)
    # beam_Y_top: (1,beam)
    # beam_idx_top: (1,beam)
    beam_Y_top = Y_top.clone().detach()
    beam_idx_top = idx_top.clone().detach()
    output_seq, attention_weight_seq = [[] for _ in range(num_pred)], []
    eos_hit = [False for _ in range(beam)]
    for _ in range(num_steps):
        # for each top `beam` prediction likelihood, We use its token as input
        # of the decoder at the next time step
        Y_top_tmp = []
        idx_top_tmp = []
        for idx_loc,idx in enumerate(idx_top.squeeze(0)):
            dec_X = idx.unsqueeze(0).unsqueeze(0)
            # Y: (1,1,tgt_vocab_size) # batch_size, num_steps = 1
            Y, dec_state = net.decoder((X_prot,dec_X), dec_state)
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
    beam_idx_top = beam_idx_top.transpose(0,1)[beam_rank_rank[:num_pred]].to('cpu').numpy()
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

    return ' '.join(translation[0]), attention_weight_seq

printer("=======================PALACE: predicting...=======================")
# number of samples using per train
batch_size = 2
# time steps/window size,ref d2l 8.1 and 8.3
num_steps = 10
beam = 5
num_pred = 3
tgt_vocab = './saved/tgt_vocab.pkl'
trained_model_dir = './trained_models/'
device = try_gpu()

assert num_pred <= beam, "number of predictions should be no larger then beam size."
# reagents list
rgt_list = ['N c 1 n c 2 c ( c ( = O ) [nH] 1 ) N [C@@H] ( C N ( C = O ) c 1 c c c ','C ( = O ) N']
# products list
prd_list = ['N c 1 n c 2 c ( c ( = O ) [nH] 1 ) N [C@@H]', 'N c 1 n c 2 c ( c ( = O ) [nH] 1 ) N [C@@H]']
# proteins list
prot_list = ['MALAHSLGFPRIGRDR','MALAHSLGFPRIGRDR']
prot_feats = prot_to_features(prot_list, trained_model_dir, device, batch_size)
assert len(rgt_list) == len(prd_list) == len(prot_list),"reagents,products and proteins should have same amount"

for src, tgt in zip(zip(prot_feats,rgt_list), prd_list):
    translation, dec_attention_weight_seq = predict_PALACE(
        net, src, src_vocab, tgt_vocab, num_steps, device,num_pred, beam, True)


logging.shutdown()
