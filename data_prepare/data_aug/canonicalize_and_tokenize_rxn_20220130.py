#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 11:00:46 2022

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
import pandas as pd
import re
from rdkit import Chem
import sys
import random

dp = 'uniprot_MetaNetX_EC_SMILES.v2.tsv'
df = pd.read_csv(dp,sep='\t',header=None)


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
                print("{} failed randome augmentaion.".format(smi))
    return random_smi

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

    
rxn_lst = list(df[1])
rxn_can_lst = []
rxn_aug_lst = []
rxn_tok_lst = []
rxn_aug_tok_lst = []
rxn_aug_shu_lst = []
rxn_tok_shu_lst = []
rxn_aug_tok_shu_lst = []
for rxn in rxn_lst:
    rxn_spl = rxn.strip().split(' = ')
    rxn_sub = rxn_spl[0]
    rxn_pro = rxn_spl[1]
    
    sub_lst = rxn_sub.split(' + ')
    pro_lst = rxn_pro.split(' + ')
    
    ## if the exact same molecule appears on the product, as well as on the reactant side, we remove it from the product side
    ## which is same as MolecularTransformer
    pro_lst = [pro for pro in pro_lst if pro_lst not in sub_lst]
    
    sub_can_lst = [canonicalize_smiles(smi) for smi in sub_lst]
    pro_can_lst = [canonicalize_smiles(smi) for smi in pro_lst]
    ## shuffle the substrates for augmentatoin
    sub_can_lst_shu = sub_can_lst.copy()
    random.shuffle(sub_can_lst_shu)
    
    sub_lst_aug = [generate_equivalent_smiles(smi) for smi in sub_can_lst]
    pro_lst_aug = [generate_equivalent_smiles(smi) for smi in pro_can_lst]
    ## shuffle the substrates for augmentatoin
    sub_lst_aug_shu = sub_lst_aug.copy()
    random.shuffle(sub_lst_aug_shu)
    
    sub_lst_tok = [tokenize_smiles(smi) for smi in sub_can_lst]
    pro_lst_tok = [tokenize_smiles(smi) for smi in pro_can_lst]
    sub_lst_tok_shu = [tokenize_smiles(smi) for smi in sub_can_lst_shu]
    
    sub_lst_aug_tok = [tokenize_smiles(smi) for smi in sub_lst_aug]
    pro_lst_aug_tok = [tokenize_smiles(smi) for smi in pro_lst_aug]
    sub_lst_aug_shu_tok = [tokenize_smiles(smi) for smi in sub_lst_aug_shu]
    
    rxn_sub_can = ' + '.join(sub_can_lst)
    rxn_pro_can = ' + '.join(pro_can_lst)
    rxn_sub_tok = ' + '.join(sub_lst_tok)
    rxn_pro_tok = ' + '.join(pro_lst_tok)
    rxn_sub_tok_shu = ' + '.join(sub_lst_tok_shu)
    
    rxn_sub_aug = ' + '.join(sub_lst_aug)
    rxn_pro_aug = ' + '.join(pro_lst_aug)
    rxn_sub_aug_shu = ' + '.join(sub_lst_aug_shu)
    
    rxn_sub_aug_tok = ' + '.join(sub_lst_aug_tok)
    rxn_pro_aug_tok = ' + '.join(pro_lst_aug_tok)
    rxn_sub_aug_shu_tok = ' + '.join(sub_lst_aug_shu_tok)
    
    ## not using ' = ' as the separator, because it's same as tokenzied double bond
    rxn_can = ' >> '.join([rxn_sub_can,rxn_pro_can])
    rxn_aug = ' >> '.join([rxn_sub_aug,rxn_pro_aug])
    rxn_aug_shu = ' >> '.join([rxn_sub_aug_shu,rxn_pro_aug])
    rxn_tok = ' >> '.join([rxn_sub_tok,rxn_pro_tok])
    rxn_tok_shu = ' >> '.join([rxn_sub_tok_shu,rxn_pro_tok])
    rxn_aug_tok = ' >> '.join([rxn_sub_aug_tok,rxn_pro_aug_tok])
    rxn_aug_tok_shu = ' >> '.join([rxn_sub_aug_shu_tok,rxn_pro_aug_tok])
    
    rxn_can_lst.append(rxn_can)
    rxn_aug_lst.append(rxn_aug)
    rxn_tok_lst.append(rxn_tok)
    rxn_aug_tok_lst.append(rxn_aug_tok)
    
    rxn_aug_shu_lst.append(rxn_aug_shu)
    rxn_tok_shu_lst.append(rxn_tok_shu)
    rxn_aug_tok_shu_lst.append(rxn_aug_tok_shu)

df[2] = rxn_aug_lst
df[3] = rxn_aug_shu_lst
df[4] = rxn_tok_lst
df[5] = rxn_aug_tok_lst
df[6] = rxn_tok_shu_lst
df[7] = rxn_aug_tok_shu_lst
df.columns = ['EC','rxn','rxn_aug','rxn_aug_shu','rxn_tok','rxn_aug_tok','rxn_tok_shu','rxn_aug_tok_shu']

