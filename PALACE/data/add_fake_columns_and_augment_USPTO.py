#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 11:28:37 2022

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
import random
from rdkit import Chem
import sys
import re

dp = r'USPTO_MIT/PALACE_USPTO_MIT_train.tsv'
output = r'USPTO_MIT/PALACE_USPTO_MIT_train.final.tsv'

def tokenize_smiles(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens),"{} failed tokenization.".format(smi)
    return ' '.join(tokens)

def generate_equivalent_smiles(smi):
    """
    data augmentation, use randome generated equivalent SMILES with duplicate same input
    """
    def smi_to_random_smi(smi):
        mol = Chem.MolFromSmiles(smi)
        Chem.SanitizeMol(mol)
        try: random_equivalent_smiles = Chem.MolToSmiles(mol, doRandom=True)
        except: sys.exit("SMILES has format error: {}".format(smi))
        return random_equivalent_smiles

    def smi_with_dot(smi):
        if '.' in smi: ## MolToSmiles cannot handle '.' within SMILES https://github.com/rdkit/rdkit/issues/3225
            smi_lst = smi.split('.')
            rand_smi_lst = [ smi_to_random_smi(smi) for smi in smi_lst]
            random_smi = '.'.join(rand_smi_lst)
        else:
            random_smi = smi_to_random_smi(smi)
        return(random_smi)

    random_smi = smi_with_dot(smi)
    if smi == random_smi:
        if len(smi) > 20:
            random_smi = smi_with_dot(smi) ## second chance
            if smi == random_smi:
                random_smi = smi_with_dot(smi) ## thrid chance
                if smi == random_smi:
                    print("{} failed randome augmentaion. skip.".format(smi))
    return random_smi

with open(dp,'r') as r:
    sub_pro_list = r.readlines()
    """
    for sub_pro in sub_pro_list:
        w.write('ID\tEC\tSeq\t'+sub_pro)
    """

sub_pro_aug_list = []
for rxn in sub_pro_list:
    sub_list = rxn.strip().split('\t')[0].split(' > ')
    pro = rxn.strip().split('\t')[1]
    sub_aug_list = [tokenize_smiles(generate_equivalent_smiles(smi.replace(' ',''))) for smi in sub_list]
    pro_aug = tokenize_smiles(generate_equivalent_smiles(pro.replace(' ','')))
    if len(sub_aug_list) == 0:
        print(rxn)
    sub_pro_aug_list.append(' > '.join(sub_aug_list) + '\t' + pro_aug + '\n')

sub_pro_list = sub_pro_list + sub_pro_aug_list
random.shuffle(sub_pro_list)
with open(output,'w') as w:
    for sub_pro in sub_pro_list:
        w.write('ID\tEC\tSeq\t'+sub_pro)




