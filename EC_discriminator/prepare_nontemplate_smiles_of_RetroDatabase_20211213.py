#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 21:24:17 2021

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
from rdkit.Chem import rdFMCS
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions
from rdkit.Chem import DataStructs
import itertools
import re
import pandas as pd
from input_output2EC_20211208 import rxn_smarts2mol_list
from input_output2EC_20211208 import two_lists_combination
from input_output2EC_20211208 import Sanitize_mol_list
import sys

#rules_dp=r"retrorules_rr01_rp2_flat_all.csv"
rules_dp=sys.argv[1]
output=str(rules_dp+".noncommon_common")
rules_df=pd.read_csv(rules_dp)

nontemplate_smiles=[]
common_smarts=[]
for index,rule in enumerate(rules_df['Rule']):
    template_reagents_mol_list,template_products_mol_list=rxn_smarts2mol_list(rule)
    template_reagent_product_mol_pair_list=two_lists_combination(template_reagents_mol_list,template_products_mol_list)
    template_FMCS_list=[rdFMCS.FindMCS(x) for x in template_reagent_product_mol_pair_list]
    template_FMCS_mol_list=[Chem.MolFromSmarts(x.smartsString) for x in template_FMCS_list]
    template_products_del_common_mol_list=[AllChem.DeleteSubstructs(x[0],x[1]) for x in zip(template_products_mol_list,template_FMCS_mol_list)]
    #Sanitize_mol_list(template_products_del_common_mol_list)
    
    temp=[]
    for m in template_products_del_common_mol_list:
        m_s=Chem.rdmolfiles.MolToSmiles(m)
        if m_s != '':
            m_s_list=m_s.split('.')
            for l in m_s_list:
                num_letters = sum(c.isalpha() for c in l)
                num_Hs=l.count('H')
                if num_letters != num_Hs: ## remove only-H noncommon part
                    temp.append(l)
                
    #nontemplate_smiles.append([Chem.rdmolfiles.MolToSmiles(m) for m in template_products_del_common_mol_list if Chem.rdmolfiles.MolToSmiles(m)!=''])
    nontemplate_smiles.append(temp)
    common_smarts.append([x.smartsString for x in template_FMCS_list if x.smartsString!=''])
rules_df['noncommon_smiles']=nontemplate_smiles
rules_df['common_smarts']=common_smarts

rules_df.to_csv(output,index=None)



