# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 15:02:15 2021

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
import random
seqs_per_rxn = 3000

with open('tok.for_non_enzyme.test.tsv','r') as tok,\
    open('nonenzyme.16to2500.nonrd90.test.fasta','r') as fa,\
    open('PALACE_test.nonenzyme.tsv','w') as P:
        seqs = fa.readlines()
        seq_num = len(seqs)/2
        if type(seq_num) is float:
            assert seq_num.is_integer()
        seq_num = int(seq_num)
        for rxn in tok:
            rand_seq_ids = random.sample(range(seq_num),seqs_per_rxn)
            for ID in rand_seq_ids:
                seq = seqs[ID * 2 + 1].strip()
                title = seqs[ID * 2].strip('>').strip()
                EC = 'N.A.'
                reagents = rxn.split(' >> ')[0]
                products = reagents
                P.write('\t'.join([title,EC,seq,reagents,products])+'\n')










