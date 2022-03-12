#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 15:36:11 2022

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ description ==============================####
the Vocab in opennmt was trained based on pytorch 0.4.1 (torchtext 0.6.0). Now I want to use the trained
model in pytorch 1.10.2 (torchtext_0.11.2). After researching the original code 
from two versions, this script is to finish the job.
=================================== input =====================================

=================================== output ====================================

================================= parameters ==================================

=================================== example ===================================

=================================== warning ===================================

####=======================================================================####
"""

import torch ## 1.10.2 

## the checkpoint model was trained in 0.4.1
pretrained_smi_dp = 'trained_models/STEREO_separated_augm_model_average_20.pt'

pretrained_smi_state_dict = torch.load(pretrained_smi_dp)
"""
extract from 0.6.0 torchtext

class Vocab(object):
    Defines a vocabulary object that will be used to numericalize a field.

    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.

"""
v_old = pretrained_smi_state_dict['vocab'][0][1].itos ## ajustify based on your model 

"""
how to create a vocab class from 0.11.2 torchtext:

>>> from torchtext.vocab import vocab
>>> from collections import Counter, OrderedDict
>>> counter = Counter(["a", "a", "b", "b", "b"])
>>> sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
>>> ordered_dict = OrderedDict(sorted_by_freq_tuples)
>>> v1 = vocab(ordered_dict)

and:
in [96]: ordered_dict
Out[96]: OrderedDict([('b', 3), ('a', 2)])

so just need to construct a new ordered_dict from v_old
"""
from torchtext.vocab import vocab # 0.11.2
from collections import OrderedDict
#new_ordered_dict = [(value, len(v_old)-int(index)) for index,value in enumerate(v_old)]

new_sorted_by_freq_tuples = [(value, len(v_old)-int(index)) for index,value in enumerate(v_old)]
new_ordered_dict = OrderedDict(new_sorted_by_freq_tuples)
v_new = vocab(new_ordered_dict)

"""
check:
In [104]: v_new.get_itos() == v_old
Out[104]: True
"""

