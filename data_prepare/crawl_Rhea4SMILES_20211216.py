#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 16:04:25 2021

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ discription ==============================####

#================================== input =====================================
(n-1)diphosphate	-
quinolinate	25888
S-sulfo-L-cysteine	52420
#================================== output ====================================

#================================ parameters ==================================

#================================== example ===================================

#================================== warning ===================================

####=======================================================================####
"""
from urllib.request import urlopen
from bs4 import BeautifulSoup
import certifi
import ssl
import sys
import time
from random import randint
import pandas as pd
#from urllib.error import HTTPError
dp="uniprot_trembl_bacteria.enzyme.tsv_reaction_CHEBI.compound_list_Rhea"

df=pd.read_csv(dp,header=None,sep='\t')

Rhea_list=df[1].to_list()
output='Rhea_name_CHEBI_SMILES_trembl.tsv'
def return_shortest_list(*lists):
    lengths=[len(x) for x in lists]
    shortest_length=min(lengths)
    for i,l in enumerate(lists):
        if len(l)==shortest_length:
            return l


def crawl_Rhea(Rhea,output):
    Rhea=str(Rhea)
    url=str('https://www.rhea-db.org/rhea/'+Rhea)

    
    # =============================================================================
    # https://stackoverflow.com/questions/27835619/urllib-and-ssl-certificate-verify-failed-error
    # =============================================================================
    html = urlopen(url, context=ssl.create_default_context(cafile=certifi.where()))
    bsObj=BeautifulSoup(html,"html.parser")
    participant_list=bsObj.findAll(name="li", attrs={"class" :"participant"})
    
    name=[]
    CHEBI=[]
    SMILES=[]
    skip_next_name=False
    for ip,p in enumerate(participant_list):
        span_list=p.findAll(name='span')
        ## make up missing record
        if not (len(name)==len(CHEBI)==len(SMILES)):
            return_shortest_list(name,CHEBI,SMILES).append('-')
        
        for index,i in enumerate(span_list):
            value=i.findAll(text=True)
            if ' Name ' in value or 'Name' in value:
                if skip_next_name:
                    skip_next_name=False
                    continue
                try:
                    name_record=span_list[index+1].findAll(text=True)
                    #print(ip,":",index)
                    name.append(''.join([x for x in name_record if x !='\n'] ))
                except:
                    print(Rhea+' has name problem.')
                    name.append('-')
                    
            if 'Identifier' in value:
                CHEBI_record=span_list[index+1].findAll(text=True)
                possible_CHEBI=[x for x in CHEBI_record if x !='\n'][0]
                
                ## tricky part, the Rhea sometimes have dual annotation to one compound
                if 'RHEA-COMP' in possible_CHEBI:
                    #print('hittt')
                    #name.pop()
                    skip_next_name=True
                    break
                
                if 'CHEBI:' in possible_CHEBI:
                    try:
                        CHEBI.append(possible_CHEBI.replace('CHEBI:',''))
                    except:
                        print(Rhea+' has CHEBI problem.')
                        CHEBI.append('-')
                else:
                    pass
                    
            if 'SMILES' in value:
                try:
                    SMILES_record=span_list[index+1].findAll(text=True)
                    SMILES.append([x for x in SMILES_record if x !='\n'][0])
                except:
                    print(Rhea+' has SMILES problem.')
                    SMILES.append('-')
    
    if not(len(name)==len(CHEBI)==len(SMILES)):
        name=[x for x in name if x!='-']
        CHEBI=[x for x in CHEBI if x!='-']
        SMILES=[x for x in SMILES if x!='-']
        if not(len(name)==len(CHEBI)==len(SMILES)):
            print('Unknown error of '+Rhea)
            """
            with open(output,'a') as o:
                o.write('\t'.join([Rhea,'-','-','-']))
            """
            return
    
    with open(output,'a') as o:
        for m in map('\t'.join,zip(name,CHEBI,SMILES)):
            o.write(Rhea)
            o.write('\t')
            o.write(m)
            o.write('\n')

for R in Rhea_list:
    if R!='-':
        crawl_Rhea(R,output=output)
        time.sleep(randint(5,10))



