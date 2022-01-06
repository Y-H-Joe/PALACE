#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 16:47:23 2022

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
import gzip

dp=r"uniprot_trembl_bacteria.dat.enzyme.train_v2.gz"
dp_id=r'uniprot_trembl.16to2500.nonrd90.fasta.id'
output=r"uniprot_trembl_bacteria.dat.enzyme.train_v3.gz"

with open(dp_id,'r') as f:
    id_set=set([x.strip() for x in f.readlines()])

with gzip.open(dp,'rb') as f:
    with gzip.open(output,'wb') as o:
        o.write(b'UnitProtID\ttaxaID\tRheaID\tEC\treaction\tCHEBI\tprotein\treformat_reaction\tCHEBI_reacion\tSMILES_reaction\r\n')
        for line in f:
            #print(line)
            ID=line.strip().split(b'\t')[0]
            if ID.decode("utf-8")  in id_set:
                o.write(line)











