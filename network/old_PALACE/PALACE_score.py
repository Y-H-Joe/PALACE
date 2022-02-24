#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 10:24:36 2022

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
from __future__ import division, unicode_literals
#import argparse
from rdkit import Chem
import pandas as pd
from argparse import Namespace

opt = Namespace(beam_size=5, invalid_smiles=False, 
                predictions='experiments/predictions_STEREO_separated_augm_model_average_20.pt_on_test.txtt', 
                targets='data/tgt-test.txtt')

def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return ''

def get_rank(row, base, max_rank):
    for i in range(1, max_rank+1):
        if row['target'] == row['{}{}'.format(base, i)]:
            return i
    return 0

def main(opt):
    with open(opt.targets, 'r') as f:
        targets = [''.join(line.strip().split(' ')) for line in f.readlines()]

    predictions = [[] for i in range(opt.beam_size)]

    test_df = pd.DataFrame(targets)
    test_df.columns = ['target']
    total = len(test_df)

    with open(opt.predictions, 'r') as f:
        for i, line in enumerate(f.readlines()):
            predictions[i % opt.beam_size].append(''.join(line.strip().split(' ')))

    for i, preds in enumerate(predictions):
        test_df['prediction_{}'.format(i + 1)] = preds
        test_df['canonical_prediction_{}'.format(i + 1)] = test_df['prediction_{}'.format(i + 1)].apply(
            lambda x: canonicalize_smiles(x))

    test_df['rank'] = test_df.apply(lambda row: get_rank(row, 'canonical_prediction_', opt.beam_size), axis=1)

    correct = 0

    for i in range(1, opt.beam_size+1):
        correct += (test_df['rank'] == i).sum()
        invalid_smiles = (test_df['canonical_prediction_{}'.format(i)] == '').sum()
        if opt.invalid_smiles:
            print('Top-{}: {:.1f}% || Invalid SMILES {:.2f}%'.format(i, correct/total*100,
                                                                     invalid_smiles/total*100))
        else:
            print('Top-{}: {:.1f}%'.format(i, correct / total * 100))



if __name__ == "__main__":
# =============================================================================
#     parser = argparse.ArgumentParser(
#         description='PALACE_score.py',
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# 
#     parser.add_argument('-beam_size', type=int, default=5,
#                        help='Beam size')
#     parser.add_argument('-invalid_smiles', action="store_true",
#                        help='Show % of invalid SMILES')
#     parser.add_argument('-predictions', type=str, default="",
#                        help="Path to file containing the predictions")
#     parser.add_argument('-targets', type=str, default="",
#                        help="Path to file containing targets")
# 
#     opt = parser.parse_args()
# =============================================================================
    main(opt)
    