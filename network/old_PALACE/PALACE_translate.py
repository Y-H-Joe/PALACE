#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 10:13:27 2022

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
from argparse import Namespace

from PALACE.utils.logging import init_logger
from PALACE.utils.misc import split_corpus
from PALACE.translate.translator import build_translator
from PALACE.utils.parse import ArgumentParser
#import PALACE.opts as opts

opt = Namespace(align_debug=False, alpha=0.0, attn_debug=False, avg_raw_probs=False,
                batch_size=4, batch_type='sents', beam_size=5, beta=-0.0, 
                block_ngram_repeat=0, config=None, coverage_penalty='none',
                data_type='text', dump_beam='', dynamic_dict=False, fp32=True,
                gpu=0, ignore_when_blocking=[], image_channel_size=3, 
                length_penalty='none', log_file='', log_file_level='0',
                max_length=200, max_sent_length=None, min_length=0,
                models=[r'trained_models/STEREO_separated_augm_model_average_20.pt'], 
                n_best=5,
                output=r'experiments/predictions_STEREO_separated_augm_model_average_20.pt_on_test.txt',
                phrase_table='', random_sampling_temp=1.0, random_sampling_topk=1, 
                ratio=-0.0, replace_unk=True, report_align=False, report_time=False, 
                sample_rate=16000, save_config=None, seed=829, shard_size=10000,
                share_vocab=False, src='data/src-test.txtt', src_dir='',
                stepwise_penalty=False, tgt=None, tgt_prefix=False, 
                verbose=False, window='hamming', window_size=0.02, window_stride=0.01)

def translate(opt):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, logger=logger, report_score=True)
    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size)
    shard_pairs = zip(src_shards, tgt_shards)

    for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
        logger.info("Translating shard %d." % i)
        translator.translate(
            src=src_shard,
            tgt=tgt_shard,
            src_dir=opt.src_dir,
            batch_size=opt.batch_size,
            batch_type=opt.batch_type,
            attn_debug=opt.attn_debug,
            align_debug=opt.align_debug
            )


# =============================================================================
# def _get_parser():
#     parser = ArgumentParser(description='PALACE_translate.py')
# 
#     opts.config_opts(parser)
#     opts.translate_opts(parser)
#     return parser
# =============================================================================


def main():
    #parser = _get_parser()

    #opt = parser.parse_args()
    translate(opt)


if __name__ == "__main__":
    main()
