#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 22:18:16 2021

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ description ==============================####

=================================== input =====================================
uniprot_sprot_bacteria.dat.enzyme.tsv_reaction_CHEBI.compound_CHEBI:
ferulate	17620
Feruloyl-polysaccharide	-
gentamicin	17833

uniprot_sprot_bacteria.dat.enzyme.tsv_reaction_CHEBI.reformat_reaction:
cholate + NADP(+) = 3alpha,7alpha-dihydroxy-12-oxo-5beta-cholanate + H(+) + NADPH
[(1->2)-beta-D-glucosyl](n) + phosphate = [(1->2)-beta-D-glucosyl](n-1) + alpha-D-glucose 1-phosphate
=================================== output ====================================
CHEBI1 + CHEBI2 = CHEBI3 + CHEBI4\tSMILES1 + SMILES2 = SMILES3 + SMILES4
================================= parameters ==================================

=================================== example ===================================

=================================== warning ===================================

####=======================================================================####
"""
import pandas as pd

dp_compound_CHEBI=r'uniprot_sprot_and_trembl_bacteria.enzyme.tsv_reaction_CHEBI.compound_CHEBI.prepared_v2'
dp_reformat_reaction=r'uniprot_sprot_bacteria.dat.enzyme.tsv_reaction_CHEBI.reformat_reaction'
dp_CHEBI_SMILES=r'chebi_core.obo.CHEBI_SMILES'

output=str(dp_reformat_reaction+".CHEBI_annotated_v2")

df_compound_CHEBI=pd.read_csv(dp_compound_CHEBI,sep='\t',header=None,index_col=0)
df_reformat_reaction=pd.read_csv(dp_reformat_reaction,sep='\t',header=None)
df_CHEBI_SMILES=pd.read_csv(dp_CHEBI_SMILES,sep='\t',header=None,index_col=0)

reagent_part=[]
product_part=[]
CHEBI_reagent_part=[]
CHEBI_product_part=[]
SMILES_reagent_part=[]
SMILES_product_part=[]

for i in df_reformat_reaction[0]:
    reaction_split=i.split(' = ')
    try:
        reagent_part.append(reaction_split[0].split(' + '))
    except:
        reagent_part.append(reaction_split)
    try:
        product_part.append(reaction_split[1].split(' + '))
    except:
        product_part.append(reaction_split)

## annotate compound with CHEBI
for r in reagent_part:
    try:
        CHEBI_tmp=[]
        for x in r:
            try:
                CHEBI_tmp.append(df_compound_CHEBI.loc[x,1])
            except:
                CHEBI_tmp.append('-')
        #CHEBI_reagent_part.append(' + '.join([df_compound_CHEBI.loc[x,1] for x in r]))
        CHEBI_reagent_part.append(' + '.join(CHEBI_tmp))
    except:
        CHEBI_reagent_part.append('-')

for p in product_part:
    try:
        CHEBI_tmp=[]
        for x in p:
            try:
                CHEBI_tmp.append(df_compound_CHEBI.loc[x,1])
            except:
                CHEBI_tmp.append('-')
        #CHEBI_product_part.append(' + '.join([df_compound_CHEBI.loc[x,1] for x in p]))
        CHEBI_product_part.append(' + '.join(CHEBI_tmp))
    except:
        CHEBI_product_part.append('-')

## annotate CHEBI with SMILES
for Cr in CHEBI_reagent_part:
    try:
        SMILES_tmp=[]
        for x in Cr.split(' + '):
            try:
                SMILES_tmp.append(df_CHEBI_SMILES.loc[int(x),1])
            except:
                SMILES_tmp.append('-')
        #SMILES_reagent_part.append(' + '.join(df_CHEBI_SMILES.loc[int(x),1] for x in Cr.split(' + ')))
        SMILES_reagent_part.append(' + '.join(SMILES_tmp))
    except:
        SMILES_reagent_part.append('-')
for Cp in CHEBI_product_part:
    try:
        SMILES_tmp=[]
        for x in Cp.split(' + '):
            try:
                SMILES_tmp.append(df_CHEBI_SMILES.loc[int(x),1])
            except:
                SMILES_tmp.append('-')
        #SMILES_product_part.append(' + '.join(df_CHEBI_SMILES.loc[int(x),1] for x in Cp.split(' + ')))
        SMILES_product_part.append(' + '.join(SMILES_tmp))
    except:
        SMILES_product_part.append('-')
        

CHEBI_reformat_reaction_part1=map(' = '.join,zip(CHEBI_reagent_part,CHEBI_product_part))
CHEBI_reformat_reaction_part2=map(' = '.join,zip(SMILES_reagent_part,SMILES_product_part))
CHEBI_reformat_reaction=map('\t'.join,zip(CHEBI_reformat_reaction_part1,CHEBI_reformat_reaction_part2))

with open(output,'w') as o:
    o.write('reformat_reaction\tCHEBI_reacion\tSMILES_reaction\n')
    for index,Crr in enumerate(CHEBI_reformat_reaction):
        o.write(df_reformat_reaction[0][index])
        o.write('\t')
        o.write(Crr)
        o.write('\n')






    