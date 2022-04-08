#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 09:20:03 2022

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ description ==============================####
in USPTO_STEREO,reaction A>B>C means the product is C;
reaction A>B>>C also means the product is C.
=================================== input =====================================

=================================== output ====================================
subtance\tproduct
================================= parameters ==================================

=================================== example ===================================

=================================== warning ===================================

####=======================================================================####
"""
from rdkit import Chem
import re
import sys
import pandas as pd

dp = r'USPTO_STEREO\US_patents_1976-Sep2016_1product_reactions_train.tsv'
df = pd.read_csv(dp,skiprows = 2,sep = '\t')
output = r'USPTO_STEREO\PALCE_USPTO_STEREO_train.tsv'

def tokenize_smiles(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens),"{} failed tokenization.".format(smi)
    return ' '.join(tokens)

def canonicalize_smiles(smi):
    """
    use RDkit Chem.MolToSmiles to canonicalize smiles
    Parameters
    ----------
    smi : TYPE
        DESCRIPTION.

    Returns
    -------
    can_smi : TYPE
        DESCRIPTION.

    """
    try:
        can_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi),isomericSmiles = True,kekuleSmiles=False,canonical=True)
    except:
        sys.exit("error SMILES: {}".format(smi))
    return  can_smi


rxn_list = list(df['CanonicalizedReaction'])
sub_list = [] # substance
pro_list = [] # product
for rxn in rxn_list:
    tmp = [x for x in rxn.split('>') if x != '']
    sub = tmp[:-1]
    pro = tmp[-1]
    if type(sub) is str:
        sub = [sub]
    sub_can = [canonicalize_smiles(smi) for smi in sub]
    pro_can = canonicalize_smiles(pro)
    sub_tok = [tokenize_smiles(smi) for smi in sub_can]
    pro_tok = tokenize_smiles(pro_can)
    sub_list.append(' > '.join(sub_tok))
    pro_list.append(pro_tok)

with open(output,'w') as o:
    for sub,pro in zip(sub_list,pro_list):
        o.write('\t'.join([sub,pro])+'\n')




