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
uniprot_MetaNetX_EC_SMILES.v2.aug_tok.tsv:
    'EC','rxn','rxn_can','rxn_aug','rxn_aug_shu','rxn_tok','rxn_aug_tok','rxn_tok_shu','rxn_aug_tok_shu'

EC_tok.tsv: the tokenized and augmented rxn SMILES pair with EC
    EC, rxn
## split test and train of SMILES, so the network never see test SMILES during training
## the protein sequence will also be split to train and test
EC_tok.train.tsv
EC_tok.test.tsv

#deprecated:only 6300 (1575*4) rxn has one-product, other 60000+ rxn are
#multiproduct rxn, so let's predict multi-product
# EC_tok.one_product.tsv: in this study, we mainly focused on one-product rxn:
#     EC, rxn
# EC_tok.one_product.train.tsv
# EC_tok.one_product.test.tsv

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

dp = 'uniprot_MetaNetX_EC_SMILES.tsv'
df = pd.read_csv(dp,sep='\t',header=None)
output1 = 'uniprot_MetaNetX_EC_SMILES.for_non_enzyme.aug_tok.tsv'
output2 = 'EC_tok.for_non_enzyme.tsv'
output3 = 'EC_tok.for_non_enzyme.train.tsv' # 90%
output4 = 'EC_tok.for_non_enzyme.test.tsv' # 5%
output5 = 'EC_tok.for_non_enzyme.val.tsv' # 5%

#output3 = 'EC_tok.one_product.tsv'
#output4 = 'EC_tok.one_product.train.tsv' # 90%
#output5 = 'EC_tok.one_product.split.tsv' # 10%


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

    ## using '>' rather than '+' to fit with pretrained MolecularTransformer network
    rxn_sub_can = ' > '.join(sub_can_lst)
    rxn_pro_can = ' > '.join(pro_can_lst)
    rxn_sub_tok = ' > '.join(sub_lst_tok)
    rxn_pro_tok = ' > '.join(pro_lst_tok)
    rxn_sub_tok_shu = ' > '.join(sub_lst_tok_shu)

    rxn_sub_aug = ' > '.join(sub_lst_aug)
    rxn_pro_aug = ' > '.join(pro_lst_aug)
    rxn_sub_aug_shu = ' > '.join(sub_lst_aug_shu)

    rxn_sub_aug_tok = ' > '.join(sub_lst_aug_tok)
    rxn_pro_aug_tok = ' > '.join(pro_lst_aug_tok)
    rxn_sub_aug_shu_tok = ' > '.join(sub_lst_aug_shu_tok)

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

df[2] = rxn_can_lst
df[3] = rxn_aug_lst
df[4] = rxn_aug_shu_lst
df[5] = rxn_tok_lst
df[6] = rxn_aug_tok_lst
df[7] = rxn_tok_shu_lst
df[8] = rxn_aug_tok_shu_lst
df.columns = ['EC','rxn','rxn_can','rxn_aug','rxn_aug_shu','rxn_tok','rxn_aug_tok','rxn_tok_shu','rxn_aug_tok_shu']

EC_count_df = pd.read_csv('EC_and_num_of_protein.tsv',sep='\t')
EC_enough = EC_count_df[EC_count_df['count']>=3]['EC'].to_list()

df_enough_prot = df[df['EC'].isin(EC_enough)]

df_test_val = df_enough_prot.sample(n = int(df.shape[0]*0.1), axis = 0)
df_test = df_test_val.sample(frac = 0.5, axis = 0)
df_val_idx = [x for x in df_test_val.index if x not in df_test.index]
df_val = df_test_val.loc[df_val_idx]
df_train_idx = [x for x in df.index if x not in df_test_val.index]
df_train = df.loc[df_train_idx]

df_EC_tok = pd.concat([df[['EC','rxn_tok']].rename(columns = {'rxn_tok':'rxn'}),
                       df[['EC','rxn_aug_tok']].rename(columns = {'rxn_aug_tok':'rxn'}),
                       df[['EC','rxn_tok_shu']].rename(columns = {'rxn_tok_shu':'rxn'}),
                       df[['EC','rxn_aug_tok_shu']].rename(columns = {'rxn_aug_tok_shu':'rxn'})],
                      axis = 0, ignore_index=True)
df_train_EC_tok = pd.concat([df_train[['EC','rxn_tok']].rename(columns = {'rxn_tok':'rxn'}),
                       df_train[['EC','rxn_aug_tok']].rename(columns = {'rxn_aug_tok':'rxn'}),
                       df_train[['EC','rxn_tok_shu']].rename(columns = {'rxn_tok_shu':'rxn'}),
                       df_train[['EC','rxn_aug_tok_shu']].rename(columns = {'rxn_aug_tok_shu':'rxn'})],
                      axis = 0, ignore_index=True)
df_test_EC_tok = pd.concat([df_test[['EC','rxn_tok']].rename(columns = {'rxn_tok':'rxn'}),
                       df_test[['EC','rxn_aug_tok']].rename(columns = {'rxn_aug_tok':'rxn'}),
                       df_test[['EC','rxn_tok_shu']].rename(columns = {'rxn_tok_shu':'rxn'}),
                       df_test[['EC','rxn_aug_tok_shu']].rename(columns = {'rxn_aug_tok_shu':'rxn'})],
                      axis = 0, ignore_index=True)
df_val_EC_tok = pd.concat([df_val[['EC','rxn_tok']].rename(columns = {'rxn_tok':'rxn'}),
                       df_val[['EC','rxn_aug_tok']].rename(columns = {'rxn_aug_tok':'rxn'}),
                       df_val[['EC','rxn_tok_shu']].rename(columns = {'rxn_tok_shu':'rxn'}),
                       df_val[['EC','rxn_aug_tok_shu']].rename(columns = {'rxn_aug_tok_shu':'rxn'})],
                      axis = 0, ignore_index=True)
## one product
"""
df_EC_tok_1p = []
for row in df_EC_tok.itertuples():
    if len(row.rxn.split(' >> ')[1].split(' + ')) == 1:
        df_EC_tok_1p.append(row)
df_EC_tok_1p = pd.DataFrame(df_EC_tok_1p)
"""

#df.to_csv(output1, sep = '\t',index=None)
df_EC_tok.to_csv(output2,sep='\t',index=None)
df_train_EC_tok.to_csv(output3,sep='\t',index=None)
df_test_EC_tok.to_csv(output4,sep='\t',index=None)
df_val_EC_tok.to_csv(output5,sep='\t',index=None)
