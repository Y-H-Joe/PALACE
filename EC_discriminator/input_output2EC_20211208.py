#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 11:12:53 2021

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
from rdkit.Chem import rdFMCS
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions
from rdkit.Chem import DataStructs
import itertools
import re
import pandas as pd

def rxn_smarts2mol_list(rxn_smarts):
    rxn_list=[x for x in rxn_smarts.split('>') if x !='']
    ## remove the wrapping ()
    if rxn_list[0].startswith('(') and rxn_list[0].endswith(')'):
        reagents_list=rxn_list[0][1:-1].split('.')
    else:
        reagents_list=rxn_list[0].split('.')
    if rxn_list[1].startswith('(') and rxn_list[1].endswith(')'):
        products_list=rxn_list[1][1:-1].split('.')
    else:
        products_list=rxn_list[1].split('.')
    
    reagents_mol_list=[Chem.MolFromSmarts(r) for r in reagents_list]
    products_mol_list=[Chem.MolFromSmarts(p) for p in products_list]

    return reagents_mol_list,products_mol_list

def mol_list2rxn_smarts(reagents_mol_list,products_mol_list,addHs=False):
    if addHs:
        ## add Hs
        reagents_mol_list_addH=[Chem.AddHs(m) for m in reagents_mol_list]
        products_mol_list_addH=[Chem.AddHs(m) for m in products_mol_list]
        
        reagents_smarts_list=[Chem.rdmolfiles.MolToSmarts(r) for r in reagents_mol_list_addH]
        products_smarts_list=[Chem.rdmolfiles.MolToSmarts(p) for p in products_mol_list_addH]
    else:
        reagents_smarts_list=[Chem.rdmolfiles.MolToSmarts(r) for r in reagents_mol_list]
        products_smarts_list=[Chem.rdmolfiles.MolToSmarts(p) for p in products_mol_list]
    rxn_smarts=str('.'.join(reagents_smarts_list)+'>>'+'.'.join(products_smarts_list))
    #rxn = rdChemReactions.ReactionFromSmarts(rxn_smarts)
    return rxn_smarts

def Sanitize_mol_list(mol_list):
    for item in mol_list:
        Chem.SanitizeMol(item)

def rxn_similarity(rxn,template_rxn):
    # construct difference fingerprint (subtracts reactant fingerprint from product)
    finger_print_rxn = rdChemReactions.CreateDifferenceFingerprintForReaction(rxn)
    finger_print_template_rxn = rdChemReactions.CreateDifferenceFingerprintForReaction(template_rxn)
    similarity=DataStructs.TanimotoSimilarity(finger_print_rxn,finger_print_template_rxn)
    return similarity

def mol_list_smilarity(list1,list2,method='Tanimoto'):
    
    if method == 'Tanimoto':
        fp_list1=[Chem.RDKFingerprint(x) for x in list1]
        fp_list2=[Chem.RDKFingerprint(x) for x in list2]
        pair_list=two_lists_combination(fp_list1,fp_list2)
        smilarity_list=[DataStructs.FingerprintSimilarity(p[0],p[1]) for p in pair_list]
        
    if method == 'MACCS':
        from rdkit.Chem import MACCSkeys
        fp_list1=[MACCSkeys.GenMACCSKeys(x) for x in list1]
        fp_list2=[MACCSkeys.GenMACCSKeys(x) for x in list2]
        pair_list=two_lists_combination(fp_list1,fp_list2)
        smilarity_list=[DataStructs.FingerprintSimilarity(p[0],p[1],\
                      metric=DataStructs.DiceSimilarity) for p in pair_list]
    
    if method == 'Morgan':
        fp_list1=[AllChem.GetMorganFingerprint(x,2,useFeatures=True) for x in list1]
        fp_list2=[AllChem.GetMorganFingerprint(x,2,useFeatures=True) for x in list2]
        pair_list=two_lists_combination(fp_list1,fp_list2)
        smilarity_list=[DataStructs.DiceSimilarity(p[0],p[1]) for p in pair_list]
    
    if method not in ['Tanimoto','MACCS','Morgan']:
        print('Wrong method.')
        return 
    
    return smilarity_list


def smiles_list2mol_list(smiles_list):
    return [Chem.MolFromSmiles(s,sanitize=True) for s in smiles_list]
    
def smarts_list2mol_list(smarts_list):
    return [Chem.MolFromSmarts(s) for s in smarts_list]
    
def two_lists_combination(list1,list2):
    lists=[]
    for i in list1:
        for j in list2:
            lists.append([i,j])
    return lists

def smiles_list2common_noncommon(reagents_smiles_list,products_smiles_list):
    """
    Parameters
    ----------
    reagents_smiles_list : TYPE
        DESCRIPTION.
    products_smiles_list : TYPE
        DESCRIPTION.

    Returns
    -------
    common_smarts_list : TYPE
        DESCRIPTION.
    noncommon_smiles_list : TYPE
        DESCRIPTION.

    """
    reagents_mol_list=smiles_list2mol_list(reagents_smiles_list)
    products_mol_list=smiles_list2mol_list(products_smiles_list)
    
    reagent_product_mol_pair_list=two_lists_combination(reagents_mol_list,products_mol_list)
    
    FMCS_list=[rdFMCS.FindMCS(x) for x in reagent_product_mol_pair_list]
    FMCS_mol_list=[Chem.MolFromSmarts(x.smartsString) for x in FMCS_list]
    
    common_smarts_list=[x.smartsString for x in FMCS_list]
    products_del_common_mol_list=[AllChem.DeleteSubstructs(x[0],x[1]) for x in zip(products_mol_list,FMCS_mol_list)]
    #noncommon_mol_list=[Chem.rdmolfiles.MolToSmiles(m) for m in products_del_common_mol_list]

    return common_smarts_list,products_del_common_mol_list


if __name__=='__main__':
    #reagents_smiles_list=["[NH3+][C@@H](CCC(=O)[O-])C(=O)[O-]","O=C([O-])CC(=O)C(=O)[O-]"]
    reagents_smiles_list=["O=C([O-])CC(=O)C(=O)[O-]"]
    products_smiles_list=["[NH3+][C@@H](CC(=O)[O-])C(=O)[O-]","O=C([O-])CCC(=O)C(=O)[O-]"]
    ## convert SMILES to SMARTS
    reagents_mol_list=smiles_list2mol_list(reagents_smiles_list)
    products_mol_list=smiles_list2mol_list(products_smiles_list)
    
    Sanitize_mol_list(reagents_mol_list)
    Sanitize_mol_list(products_mol_list)
    
    rxn_smarts=mol_list2rxn_smarts(reagents_mol_list,products_mol_list)
    rxn=rdChemReactions.ReactionFromSmarts(rxn_smarts)
    
    common_smarts_list,noncommon_mol_list=smiles_list2common_noncommon(\
                                      reagents_smiles_list,products_smiles_list)
    common_mol_list=smarts_list2mol_list(common_smarts_list)
    #Sanitize_mol_list(common_mol_list)
    for m in common_mol_list:
        m.UpdatePropertyCache()
    
    rules_dp=r"retrorules_rr01_rp2_flat_all.csv_28.noncommon_common"
    rules_df=pd.read_csv(rules_dp)
    
    rules_noncommon_smiles=rules_df['noncommon_smiles']
    rules_common_smarts=rules_df['common_smarts']
    # str to list
    rules_noncommon_smiles_col=[eval(x) for x in rules_noncommon_smiles]
    rules_common_smarts_col=[eval(x) for x in rules_common_smarts]
    
    rules_noncommon_mol_col=[smiles_list2mol_list(x) for x in rules_noncommon_smiles_col]
    rules_common_mol_col=[smarts_list2mol_list(x) for x in rules_common_smarts_col]
    
    # score the noncommon part
    noncommon_score_list=[]
    for rules_mol_list in rules_noncommon_mol_col:
        rules_mol_list=[x for x in rules_mol_list if x] ## remove None
        similarity_list=mol_list_smilarity(noncommon_mol_list,rules_mol_list)
        noncommon_score_list.append(similarity_list)

    # score the common part
    common_score_list=[]
    for rules_mol_list in rules_common_mol_col:
        rules_mol_list=[x for x in rules_mol_list if x] ## remove None
        """
        Sanitize_mol_list(rules_mol_list)
        for m in rules_mol_list:
            m.UpdatePropertyCache()
        """
        similarity_list=mol_list_smilarity(common_mol_list,rules_mol_list)
        common_score_list.append(similarity_list)
    
    # process score lists to pick out the rule
    common_similarity_cutoff=0.25
    noncommon_similarity_cutoff=0.9
    ## first filter: similarity score
    ## if two reactions are similar, then the common part of query must be very
    ## similar to at least one of target common part
    ## let's say we have query common mol1 and mol2, and the target common Mol1,Mol2,Mol3
    ## we will have 6 similarity scores, which are [mol1_vs_Mol1, mol1_vs_Mol2, mol1_vs_Mol3,\
    ##  mol2_vs_Mol1, mol2_vs_Mpl2, mol3_vs_Mol3]. So we can split the scores to
    ## two parts, in [mol1_vs_Mol1, mol1_vs_Mol2, mol1_vs_Mol3] part, there must be at least
    ## one high score, same as the left part. Also, the highest score locus in two
    ## parts should be different.
    
    ## second filter: the atom type, the common atom number
    ## both passed common_filter and noncommon_filter
    common_mol_num=len(common_mol_list)
    passed_common_mol={}
    for index,c in enumerate(common_score_list):
        c_binary=[bool(x > common_similarity_cutoff) for x in c ]
        c_sublist=[c_binary[x:x+len(c_binary)] for x in range(0, len(c_binary), common_mol_num)]
        final_score=sum([bool(True in x) for x in c_sublist ])
        if final_score == len(c_sublist):
            passed_common_mol[index]=c
        
    noncommon_mol_num=len(noncommon_mol_list)
    passed_noncommon_mol={}
    for index,c in enumerate(noncommon_score_list):
        c_binary=[bool(x > noncommon_similarity_cutoff) for x in c ]
        c_sublist=[c_binary[x:x+len(c_binary)] for x in range(0, len(c_binary), common_mol_num)]
        final_score=sum([bool(True in x) for x in c_sublist ])
        if final_score == len(c_sublist):
            passed_noncommon_mol[index]=c
    
    inter_keys=sorted(list(set( passed_common_mol.keys() ) & set( passed_noncommon_mol.keys())))
    
    ID=2
    print(noncommon_score_list[inter_keys[ID]])
    print(common_score_list[inter_keys[ID]])
    template_rxn_smarts=rules_df.loc[inter_keys[ID],'Rule']
    template_rxn=rdChemReactions.ReactionFromSmarts(template_rxn_smarts)
    Draw.ReactionToImage(template_rxn)
    #Draw.ReactionToImage(rxn)
    #template_rxn_smarts="([#8&v2:1]=[#6&v4:2](-[#6&v4:3](-[#6&v4:4](-[#8&v2:5]-[#1&v1:6])=[#8&v2:7])(-[#1&v1:8])-[#1&v1:9])-[#6&v4:10](=[#8&v2:11])-[#8&v2:12]-[#1&v1:13])>>([#7&v3](-[#6&v4:2](-[#6&v4:3](-[#6&v4:4](-[#8&v2:5]-[#1&v1:6])=[#8&v2:7])(-[#1&v1:8])-[#1&v1:9])(-[#6&v4:10](=[#8&v2:11])-[#8&v2:12]-[#1&v1:13])-[#1&v1])(-[#1&v1])-[#1&v1].[#8&v2](-[#6&v4](=[#8&v2])-[#6&v4](=[#8&v2:1])-[#6&v4](-[#6&v4](-[#6&v4](=[#8&v2])-[#8&v2]-[#1&v1])(-[#1&v1])-[#1&v1])(-[#1&v1])-[#1&v1])-[#1&v1])"
    #template_rxn_smarts='([#8]=[#6](-[#6](-[#6](-[#8]-[#1])=[#8])(-[#1])-[#1])-[#6](=[#8])-[#8]-[#1])>>([#7](-[#6](-[#6](-[#6](-[#8]-[#1])=[#8])(-[#1])-[#1])(-[#6](=[#8])-[#8]-[#1])-[#1])(-[#1])-[#1].[#8](-[#6](=[#8])-[#6](=[#8])-[#6](-[#6](-[#6](=[#8])-[#8]-[#1])(-[#1])-[#1])(-[#1])-[#1])-[#1])'
    ## replace reagent -> 0.06 to 0.1
    #template_rxn_smarts="[#8]=[#6](-[#8-])-[#6]-[#6](=[#8])-[#6](=[#8])-[#8-]>>([#7](-[#6](-[#6](-[#6](-[#8]-[#1])=[#8])(-[#1])-[#1])(-[#6](=[#8])-[#8]-[#1])-[#1])(-[#1])-[#1].[#8](-[#6](=[#8])-[#6](=[#8])-[#6](-[#6](-[#6](=[#8])-[#8]-[#1])(-[#1])-[#1])(-[#1])-[#1])-[#1])"
    ## replace product -> 0.06 to 0.39
    #template_rxn_smarts="([#8]=[#6](-[#6](-[#6](-[#8]-[#1])=[#8])(-[#1])-[#1])-[#6](=[#8])-[#8]-[#1])>>[#7H3+]-[#6@@H](-[#6]-[#6](=[#8])-[#8-])-[#6](=[#8])-[#8-].[#8]=[#6](-[#8-])-[#6]-[#6]-[#6](=[#8])-[#6](=[#8])-[#8-]"
    ## remove Hs in template
    #template_reagents_mol_list,template_products_mol_list=rxn_smarts2mol_list(template_rxn_smarts)
    
    """
    if rmH==True:
        template_reagents_mol_list_rmH=[Chem.RemoveHs(m) for m in template_reagents_mol_list]
        template_product_mol_list_rmH=[Chem.RemoveHs(m) for m in template_products_mol_list]
        Sanitize_mol_list(template_reagents_mol_list_rmH)
        Sanitize_mol_list(template_product_mol_list_rmH)
        template_rxn_smarts=mol_list2rxn_smarts(template_reagents_mol_list_rmH,template_product_mol_list_rmH)
    else:
        template_reagents_mol_list_rmH=[Chem.RemoveHs(m) for m in template_reagents_mol_list]
        template_product_mol_list_rmH=[Chem.RemoveHs(m) for m in template_products_mol_list]
        Sanitize_mol_list(template_reagents_mol_list_rmH)
        Sanitize_mol_list(template_product_mol_list_rmH)
        template_rxn_smarts=mol_list2rxn_smarts(template_reagents_mol_list,template_products_mol_list)
    """
    #template_rxn=rdChemReactions.ReactionFromSmarts(template_rxn_smarts)
    
    #Draw.ReactionToImage(rxn)
    #Draw.ReactionToImage(template_rxn)
    ## calculate similarity
    ## forward or reverse reaction will not change similarity
    ## higher diameter will increase similarity
    ## split reagents will increase similarity
    """
    0.06是所有template里最高的吗？ 不是，中上都算不上
    similarity_dict={}
    for index,rule in enumerate(rules_df['Rule']):
        template_rxn=rdChemReactions.ReactionFromSmarts(rule)
        similarity_dict[index]=similarity(rxn,template_rxn)
    sorted_similarity_dict=dict(sorted(similarity_dict.items(), key=lambda item: item[1],reverse=True))
    sorted_similarity_dict_keys=list(sorted_similarity_dict.keys())
    highest_similarity=max(similarity_dict.values())
    highest_rxn_smarts='([#8&v2:1]-[#6&v4:2]1-[#6&v4:3]-[#6&v4:4]-[#6&v4:5](=[#6&v4:6](-[#7&v3:7]=1)-[#1&v1:8])-[#6&v4:9])>>([#7&v3:7](-[#1&v1])(-[#1&v1])-[#1&v1].[#8&v2:1]-[#6&v4:2](=[#8&v2])-[#6&v4:3]-[#6&v4:4]-[#6&v4:5](-[#6&v4:6](=[#8&v2])-[#1&v1:8])(-[#6&v4:9])-[#1&v1])'
    highest_rxn=rdChemReactions.ReactionFromSmarts(highest_rxn_smarts)
    Draw.ReactionToImage(highest_rxn)
    """
    """
    输入利用rules生成输出，看哪个输出与我的输出最相似？不行，输入必须与模板的反应物有某种匹配，否则无输出
    product_mol_list=rxn.RunReactants(reagents_mol_list)
    Draw.MolToImage(product_mol_list[0][1])
    
    tmp_product_mol_list=template_rxn.RunReactants(reagents_mol_list)
    Draw.MolToImage(tmp_product_mol_list[0][1])  
    """
    """
    #根据output-input的差异部分，与database里所有的差异部分做比较?
    AllChem.GenerateDepictionMatching2DStructure(reagents_mol_list[0], template_reagents_mol_list[0])
    res=rdFMCS.FindMCS([reagents_mol_list[0], template_reagents_mol_list[0]])
    common=Chem.MolFromSmarts(res.smartsString)
    Draw.MolToImage(common)
    
    reagent_product_mol_pair_list=two_lists_combination(reagents_mol_list,products_mol_list)
    FMCS_list=[rdFMCS.FindMCS(x) for x in reagent_product_mol_pair_list]
    FMCS_mol_list=[Chem.MolFromSmarts(x.smartsString) for x in FMCS_list]
    products_del_common_mol_list=[AllChem.DeleteSubstructs(x[0],x[1]) for x in zip(products_mol_list,FMCS_mol_list)]
    
    template_reagent_product_mol_pair_list=two_lists_combination(template_reagents_mol_list,template_products_mol_list)
    template_FMCS_list=[rdFMCS.FindMCS(x) for x in template_reagent_product_mol_pair_list]
    template_FMCS_mol_list=[Chem.MolFromSmarts(x.smartsString) for x in template_FMCS_list]
    template_products_del_common_mol_list=[AllChem.DeleteSubstructs(x[0],x[1]) for x in zip(template_products_mol_list,template_FMCS_mol_list)]
    
    
    fp1=Chem.RDKFingerprint(products_del_common_mol_list[0])
    fp2=Chem.RDKFingerprint(template_products_del_common_mol_list[0])
    DataStructs.FingerprintSimilarity(fp1,fp2) # 0.28
    
    smiles1=Chem.rdmolfiles.MolToSmiles(products_del_common_mol_list[0]) # '[NH3+]'
    smiles2=Chem.rdmolfiles.MolToSmiles(template_products_del_common_mol_list[0]) # '[HH].[H]N[H]'
    m1=Chem.MolFromSmiles(smiles1)
    m2=Chem.MolFromSmiles(smiles2)
    fp1=Chem.RDKFingerprint(m1)
    fp2=Chem.RDKFingerprint(m2)
    DataStructs.FingerprintSimilarity(fp1,fp2) # 1
    
    smiles3=Chem.rdmolfiles.MolToSmiles(products_del_common_mol_list[1]) # 'O=C[O-]'
    smiles4=Chem.rdmolfiles.MolToSmiles(template_products_del_common_mol_list[1]) # '[HH].[HH].[H]OC=O'
    m3=Chem.MolFromSmiles(smiles3)
    m4=Chem.MolFromSmiles(smiles4)
    fp3=Chem.RDKFingerprint(m3)
    fp4=Chem.RDKFingerprint(m4)
    DataStructs.FingerprintSimilarity(fp3,fp4) # 0
    
    smarts1=Chem.rdmolfiles.MolToSmarts(products_del_common_mol_list[0]) # '[#7H3+]'
    smarts2=Chem.rdmolfiles.MolToSmarts(template_products_del_common_mol_list[0]) # '[#7](-[#1])-[#1].[#1]'
    s1=Chem.MolFromSmarts(smarts1)
    s2=Chem.MolFromSmarts(smarts2)
    fp1=Chem.RDKFingerprint(s1)
    fp2=Chem.RDKFingerprint(s2)
    DataStructs.FingerprintSimilarity(fp1,fp2) # 0
    """
    #为什么similarity是0？ 用smarts会成0，用smiles会成1
    
    """
    取common会增加相似性吗？不会
    cmfp=Chem.RDKFingerprint(common)
    trm0fp=Chem.RDKFingerprint(template_reagents_mol_list[0])
    DataStructs.FingerprintSimilarity(cmfp,trm0fp) # 0.28
    """
    
    """
    每个分子都相似吗？ 不是
    rm0fp=Chem.RDKFingerprint(reagents_mol_list[0])
    trm0fp=Chem.RDKFingerprint(template_reagents_mol_list_rmH[0])
    DataStructs.FingerprintSimilarity(rm0fp,trm0fp) # 0.28
    
    rm0fp=Chem.RDKFingerprint(products_mol_list[0])
    trm0fp=Chem.RDKFingerprint(template_product_mol_list_rmH[0])
    DataStructs.FingerprintSimilarity(rm0fp,trm0fp) # 0.146
    
    rm0fp=Chem.RDKFingerprint(products_mol_list[1])
    trm0fp=Chem.RDKFingerprint(template_product_mol_list_rmH[1])
    DataStructs.FingerprintSimilarity(rm0fp,trm0fp) # 0.207
    """
    """
    分子排布方式不同其fingerprint相同吗？ 相同
    m = Chem.MolFromSmiles('OC1C2C1CC2')
    Draw.MolToImage(m)
    mm = Chem.MolFromSmiles('C1CC2C1C2O')
    Draw.MolToImage(mm)
    fpm=Chem.RDKFingerprint(m)
    fpmm=Chem.RDKFingerprint(mm)
    DataStructs.FingerprintSimilarity(fpm,fpmm) # 1
    """
    


