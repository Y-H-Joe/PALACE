#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 15:27:53 2022

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ discription ==============================####

#================================== input =====================================

#================================== output ====================================

#================================ parameters ==================================

#================================== example ===================================

#================================== warning ===================================

####=======================================================================####
"""
def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def generate_equivalent_smiles(smi):
    """
    data augmentation, use randome generated equivalent SMILES with duplicate same input
    """
    from rdkit import Chem
    random_equivalent_smiles = Chem.MolFromSmiles(Chem.MolToSmiles(smi, doRandom=True))

def canonicalize_smiles(smi):
    from rdkit import Chem

    smis = ["CN2C(=O)N(C)C(=O)C1=C2N=CN1C",
          "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"]
    
    cans = [Chem.MolToSmiles(Chem.MolFromSmiles(smi),True) for smi in smis]
    assert cans[0] == cans[1]
    
    