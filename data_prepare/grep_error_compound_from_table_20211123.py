#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 10:02:25 2021

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ discription ==============================####

#================================== input =====================================
table:
UnitProtID	taxaID	RheaID	EC	reaction	CHEBI	protein
P21215	1507	14129	1.1.1.176	cholate + NADP(+) = 3alpha,7alpha-dihydroxy-12-oxo-5beta-cholanate + H(+) + NADPH	11901;15378;29747;57783;58349	MIFDGKVAIITGGGKAKSIGYGIAVAYAK

error_compound:
xenobiotic(Side 2).
Peptidoglycan-N-acetyl-D-glucosamine
a 5'-hydroxy-ribonucleotide-3'-(RNAfragment).
#================================== output ====================================

#================================ parameters ==================================

#================================== example ===================================

#================================== warning ===================================
need to manually annotate using the output. Rheal and KEGG enzyme databases are
very useful.
####=======================================================================####
"""
query=r"uniprot_trembl_bacteria.dat.enzyme.tsv_reaction_CHEBI.error_compound"
database=r"uniprot_trembl_bacteria.enzyme.tsv"
output=str(query+"_greped")

with open(query,'r') as qq:
    query_list=[x.strip() for x in qq.readlines()]
    with open(database,'r') as dd:
        with open(output,'w') as oo:
            for d in dd:
                for q in query_list:
                    if q in d:
                        oo.write(q)
                        oo.write('\t')
                        oo.write(d)
                        query_list.remove(q)
                        
            















