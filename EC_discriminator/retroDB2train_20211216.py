#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 10:44:31 2021

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ discription ==============================####
rules:
    reaction_id substrate_id smiles_id direction
rule_products:
    reaction_id substrate_id product_id
smiles:
    id smiles_string
ec_reactions:
    reaction_id ec_number

#================================== input =====================================
retro database, which includes tables of:
    rules
    rule_products
    smiles
    ec_reactions
    
#================================== output ====================================

reaction_id substrate_smiles products_smiles ec
output1: ec can be empty or multiple
output2: ec can only be one (melt)
#================================ parameters ==================================

#================================== example ===================================

#================================== warning ===================================
care about the direction of reactions
####=======================================================================####
"""
import pandas as pd

rules_dp=r"rules.zip"
rule_products_dp="rule_products.zip"
smiles_dp="smiles.zip"
ec_reactions_dp="ec_reactions.zip"

output1='retro_processed_v1.csv'
output2='retro_processed_v2.csv'


rules_df=pd.read_csv(rules_dp)
rule_products_df=pd.read_csv(rule_products_dp)
smiles_df=pd.read_csv(smiles_dp)
ec_reactions_df=pd.read_csv(ec_reactions_dp)

## only takes forward reaction
rules_df_tmp=rules_df[["reaction_id","substrate_id","smiles_id","direction"]]
rule_products_df_tmp=rule_products_df[["reaction_id","substrate_id","product_id"]]

rules_df_tmp_forward=rules_df_tmp[rules_df_tmp['direction']==1]
reaction_substrate_pair=set(zip(rules_df_tmp_forward["reaction_id"],rules_df_tmp_forward["substrate_id"]))

pair_responding_products=[]
for pair in reaction_substrate_pair:
    reaction_id=pair[0]
    substrate_id=pair[1]
    product_set=set(rule_products_df_tmp.loc[(rule_products_df_tmp["reaction_id"]==reaction_id) & (rule_products_df_tmp["substrate_id"]==substrate_id)]['product_id'])
    pair_responding_products.append(product_set)


## annotate with smiles
reaction_list=[x[0] for x in reaction_substrate_pair]
substrate_list=[x[1] for x in reaction_substrate_pair]
product_list=[list(x) for x in pair_responding_products]

substrate_smiles_list=[]
product_smiles_list=[]
for s in substrate_list:
    substrate_smiles_list.append(smiles_df.loc[s,'smiles_string'])
# one substrate will respond to multiple products
for p in product_list:
    p_smiles=[smiles_df.loc[x,'smiles_string'] for x in p]
    product_smiles_list.append(p_smiles)

## reaction to ec
ec_list=[]
for r in reaction_list:
    ec_list.append(list(ec_reactions_df[ec_reactions_df['reaction_id']==r]['ec_number']))

## concate
output={'reaction_id':reaction_list,\
            'substrate_smiles':substrate_smiles_list,\
            'products_smiles':product_smiles_list,\
            'ec':ec_list}
output_df1=pd.DataFrame(output)
# prepare output2
index_list=[]
for i in output_df1.index:
    if output_df1.loc[i,'ec']!=[]:
        index_list.append(i)
output_df2_tmp=output_df1.loc[index_list]

series_list=[]
for i in output_df2_tmp.index:
    if len(output_df2_tmp.loc[i,'ec'])==1:
        series_list.append(list(output_df2_tmp.loc[i]))
    else:
        tmp=list(output_df2_tmp.loc[i])[:-1]
        for e in output_df2_tmp.loc[i,'ec']:
            tmp1=tmp+[[e]]
            series_list.append(tmp1)
output_df2=pd.DataFrame(series_list,columns=['reaction_id','substrate_smiles','products_smiles','ec'])

output_df1.to_csv(output1,index=None)
output_df2.to_csv(output2,index=None)

