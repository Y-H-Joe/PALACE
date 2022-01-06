#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 10:59:13 2021

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ discription ==============================####
linear programming problem. I have the compound name and CHEBI ID (maybe truncated),
but don't know the linkage. The solution is to link the compound name to its
CHEBI ID sets intersection. If all compound name successfully have a linked CHEBI
ID and detect no conflicts, problem would be solved.

#================================== input =====================================
reaction        CHEBI
cholate + NADP(+) = 3alpha,7alpha-dihydroxy-12-oxo-5beta-cholanate + H(+) + NADPH       11901;15378;29747;57783;58349
(1R,4R)-bornane-2,5-dione + FMNH2 + O2 = (1R,4R)-5-oxo-1,2-campholide + FMN + H(+) + H2O        15377;15378;15379;15392;18130;57618;58210
#================================== output ====================================
output_reformat_reaction=str(dp+".reformat_reaction")
output_compound_CHEBI_dict=str(dp+".compound_CHEBI")
output_error_compound=str(dp+".error_compound")

#================================ parameters ==================================

#================================== example ===================================

#================================== warning ===================================

####=======================================================================####
"""
import re
import sys

dp=r'uniprot_sprot_bacteria.dat.enzyme.tsv_reaction_CHEBI'
output_reformat_reaction=str(dp+".reformat_reaction")
output_compound_CHEBI_dict=str(dp+".compound_CHEBI")
output_error_compound=str(dp+".error_compound")
output_compound_list=str(dp+".compound_list")

compound_CHEBI_dict={}
compound_track_origin={}
temp_compound_CHEBI_dict={}
error_compound_list=[]
#error_reaction_list=[]
compound_list=[]
reformat_reaction_list=[]

def Intersection(lst1, lst2):
    return list(set(lst1).intersection(lst2))

with open(dp,'r') as d:
    next(d)
    for index,line in enumerate(d):
        ## don't know how to split with '+' while not split with '(+)' as 
        ## the same time, so replace '(+)' with '*'
        #new_line=line.replace('(+)','*')
        line_split1=[x.strip().strip('.') for x in re.split('\t',line) if x!='']
        
        ## reaction
        reaction_part=line_split1[0]
        line_split2=[x.strip().strip('.') for x in re.split('=',reaction_part) if x!='']
        
        if len(line_split2)!=2:
            print("========================")
            print('reaction has problem.')
            print('problem reaction: ',reaction_part)
            print("========================")
            #error_reaction_list.append(reaction_part)
            #sys.exit()
            reformat_reaction_list.append('-')
            continue
            
        reagent_part=line_split2[0]
        product_part=line_split2[1]
        
        reagent_list=[x.strip().strip('.') for x in re.split('\+ | \+| \+ ',reagent_part) if x != '']
        product_list=[x.strip().strip('.') for x in re.split('\+ | \+| \+ ',product_part) if x != '']
        
        compound_list=list(set(compound_list+reagent_list+product_list))
        ## CHEBI
        CHEBI_part=line_split1[1]
        CHEBI_list=[x.strip().strip('.') for x in re.split(';',CHEBI_part) if x != '']
        ## reformat
        reformat_reaction=str(' + '.join(reagent_list)+' = '+' + '.join(product_list))
        reformat_reaction_list.append(reformat_reaction)
        ## linear programming
        for c in reagent_list+product_list:
            if c in temp_compound_CHEBI_dict.keys():
                temp_compound_CHEBI_dict[c]=Intersection(CHEBI_list,temp_compound_CHEBI_dict[c])
                if len(temp_compound_CHEBI_dict[c])==1: ## 1 vs 1
                    ## 1 vs 1, but with wrong linkage
                    if (c in compound_CHEBI_dict.keys()) and compound_CHEBI_dict[c]!=temp_compound_CHEBI_dict[c]:
                        error_compound_list.append(c)
                        continue
                    ## 1 vs 1, with right linkage
                    compound_CHEBI_dict[c]=temp_compound_CHEBI_dict[c]
                    compound_track_origin[c]=index
                    
                    continue
                if len(temp_compound_CHEBI_dict[c])==0:
                    error_compound_list.append(c)
            else:
                temp_compound_CHEBI_dict[c]=CHEBI_list

## check repeat linkage
compound_CHEBI_dict_value_list=list(compound_CHEBI_dict.values())
for key,value in compound_CHEBI_dict.items():
    if compound_CHEBI_dict_value_list.count(value)>1:
        error_compound_list.append(key)
    
## check whether all compounds have corresponding CHEBI
if len(compound_list)+len(error_compound_list)!=len(compound_CHEBI_dict.keys()):
    for k in compound_CHEBI_dict.keys():
        if k not in compound_list:
            error_compound_list.append(k)
            
error_compound_list=list(set(error_compound_list))

compound_CHEBI_dict_keys_list=list(compound_CHEBI_dict.keys())
for key in compound_CHEBI_dict_keys_list:
    if key in error_compound_list:
        del compound_CHEBI_dict[key]
    else:
        compound_CHEBI_dict[key]=compound_CHEBI_dict[key][0]


## output
with open(output_reformat_reaction,'w') as f:
    for r in reformat_reaction_list:
        f.write(r)
        f.write('\n')
with open(output_compound_CHEBI_dict,'w') as f:
    for key,value in compound_CHEBI_dict.items():
        f.write('\t'.join([key,value]))
        f.write('\n')
with open(output_error_compound,'w') as f:
    for e in error_compound_list:
        f.write(e)
        f.write('\n')
with open(output_compound_list,'w') as f:
    for c in compound_list:
        f.write(c)
        f.write('\n')




