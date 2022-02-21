#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 09:26:37 2022

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
"""
from transformers import BertForMaskedLM, BertTokenizer, pipeline
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert_bfd")
unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)
unmasker('D L I P T S S K L V V [MASK] D T S L Q V K K A F F A L V T')
"""

import torch
from torch import nn
import re
from transformers import BertForMaskedLM, BertTokenizer
from transformers import pipeline

## get model
model = BertForMaskedLM.from_pretrained('./trained_models/')
tokenizer = BertTokenizer.from_pretrained('./trained_models')
## model to GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

## try
seq = ' '.join([re.sub(r"[UZOB]", "X", sequence) for sequence in list('MIFDGKVAIITGGGKAKSIGYGIAVAYAK')])
indexed_tokens = tokenizer.encode(seq)
tokens_tensor = torch.tensor([indexed_tokens])
## get prot sequences
sequences_Example = ["A E T C Z A O "*500,"S K T Z P"] ## after testing, the input prot seq can more than 2000
sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

## Tokenize, encode sequences and load it into the GPU if possibile
ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, padding=True)
input_ids = torch.tensor(ids['input_ids']).to(device)
attention_mask = torch.tensor(ids['attention_mask']).to(device)

##  Extracting sequences' features and load it into the CPU if needed
with torch.no_grad():
    embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]

## Remove padding ([PAD]) and special tokens ([CLS],[SEP]) that is added by ProtBert-BFD model
features = [] 
for seq_num in range(len(embedding)):
    # remove padding
    seq_len = (attention_mask[seq_num] == 1).sum()
    # remove special tokens
    seq_emd = embedding[seq_num][1:seq_len-1]
    features.append(seq_emd)







## trash bin
if False:
    ## embeddings
    fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer,device=0 )
    embedding = fe([seq])
    embedding = torch.tensor(embedding)
    
    ## see model parameters in dict
    model_dict = model.state_dict()
    ## get encoder part of model
    """
    for name,_ in model.named_children():
        print(name)
    
    bert
    cls
    """
    model_bert = list(model.children())[0]
    model_bert_dict = model_bert.state_dict()




