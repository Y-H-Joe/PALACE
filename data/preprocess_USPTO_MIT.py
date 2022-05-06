#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 10:31:01 2022

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ description ==============================####
A>>B
A is substance, B is product
all reactions are one substance, one product
=================================== input =====================================

=================================== output ====================================

================================= parameters ==================================

=================================== example ===================================

=================================== warning ===================================

####=======================================================================####
"""
from rdkit import Chem
import re
import sys

dp = r'USPTO_MIT\train.txt'
output = r'USPTO_MIT\PALACE_USPTO_MIT_train.tsv'
with open(dp,'r') as r:
    rxn_list = [x.split(' ')[0].strip() for x in r.readlines()]

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
    # remove atom mapping
    mol = Chem.MolFromSmiles(smi)
    [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    smi = Chem.MolToSmiles(mol)
    try:
        can_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi),isomericSmiles = True,kekuleSmiles=False,canonical=True)
    except:
        sys.exit("error SMILES: {}".format(smi))
    return  can_smi

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





