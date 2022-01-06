#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 10:55:53 2021

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ discription ==============================####

#================================== input =====================================
CHEBI_obo:
[Term]
id: CHEBI:30151
name: aluminide(1-)
subset: 3_STAR
property_value: http://purl.obolibrary.org/obo/chebi/formula "Al" xsd:string
property_value: http://purl.obolibrary.org/obo/chebi/charge "-1" xsd:string
property_value: http://purl.obolibrary.org/obo/chebi/monoisotopicmass "26.98209" xsd:string
property_value: http://purl.obolibrary.org/obo/chebi/mass "26.98154" xsd:string
property_value: http://purl.obolibrary.org/obo/chebi/inchi "InChI=1S/Al/q-1" xsd:string
property_value: http://purl.obolibrary.org/obo/chebi/smiles "[Al-]" xsd:string
property_value: http://purl.obolibrary.org/obo/chebi/inchikey "SBLSYFIUPXRQRY-UHFFFAOYSA-N" xsd:string
is_a: CHEBI:33429
is_a: CHEBI:33627


#================================== output ====================================
30151\t"[Al-]"

#================================ parameters ==================================

#================================== example ===================================

#================================== warning ===================================

####=======================================================================####
"""
import sys
#dp='aa'
dp=sys.argv[1]
output=str(dp+'.CHEBI_SMILES')

CHEBI=''
SMILES='-'
with open(dp,'r') as d:
    with open(output,'w') as o:
        for line in d:
            if line.startswith('[Term]'):
                if CHEBI !='':
                    o.write(CHEBI)
                    o.write('\t')
                    o.write(SMILES)
                    o.write('\n')
                CHEBI=''
                SMILES='-'
            if line.startswith('id: CHEBI:'):
                CHEBI=line.strip().split(':')[-1]
            if "http://purl.obolibrary.org/obo/chebi/smiles" in line:
                SMILES=line.strip().split(' ')[-2]


