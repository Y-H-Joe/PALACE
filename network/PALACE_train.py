#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 17:03:18 2022

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

## prepare args
from argparse import Namespace

opt = Namespace(aan_useffn=False, accum_count=[4], accum_steps=[0], 
                adagrad_accumulator_init=0, adam_beta1=0.9, adam_beta2=0.998, 
                alignment_heads=0, alignment_layer=-3, apex_opt_level='O1', 
                attention_dropout=[0.1], audio_enc_pooling='1', average_decay=0,
                average_every=1, batch_size=4096, batch_size_multiple=None, 
                batch_type='tokens', bidir_edges=True, bridge=False, 
                bridge_extra_node=True, brnn=None, cnn_kernel_width=3,
                config=None, context_gate=None, copy_attn=False, 
                copy_attn_force=False, copy_attn_type=None,
                copy_loss_by_seqlength=False, coverage_attn=False, data='data/',
                data_ids=[None], data_to_noise=[], data_weights=[1], 
                dec_layers=2, dec_rnn_size=500, decay_method='noam', 
                decay_steps=10000, decoder_type='transformer', dropout=[0.1],
                dropout_steps=[0], early_stopping=0, early_stopping_criteria=None, 
                enc_layers=2, enc_rnn_size=500, encoder_type='transformer',
                epochs=0, exp='', exp_host='', feat_merge='concat', 
                feat_vec_exponent=0.7, feat_vec_size=-1, fix_word_vecs_dec=False, 
                fix_word_vecs_enc=False, full_context_alignment=False,
                generator_function='softmax', global_attention='general',
                global_attention_function='softmax', gpu_backend='nccl', 
                gpu_ranks=[0], gpu_verbose_level=0, gpuid=[], heads=8, 
                image_channel_size=3, input_feed=1, keep_checkpoint=20,
                label_smoothing=0.0, lambda_align=0.0, lambda_coverage=0.0, 
                layers=4, learning_rate=2.0, learning_rate_decay=0.5, log_file='log', 
                log_file_level='0', loss_scale=0, master_ip='localhost',
                master_port=10000, max_generator_batches=32, max_grad_norm=0.0,
                max_relative_positions=0, model_dtype='fp32', model_type='text',
                n_edge_types=2, n_node=2, n_steps=2, normalization='tokens',
                optim='adam', param_init=0.0, param_init_glorot=True, pool_factor=8192,
                position_encoding=True, pre_word_vecs_dec=None,
                pre_word_vecs_enc=None, queue_size=40, report_every=1000, 
                reset_optim='none', reuse_copy_attn=False, rnn_size=512,
                rnn_type='LSTM', sample_rate=16000, save_checkpoint_steps=10, 
                save_config=None, save_model='experiments/checkpoints_model', 
                seed=42, self_attn_type='scaled-dot', share_decoder_embeddings=False,
                share_embeddings=True, single_pass=False, src_noise=[],
                src_noise_prob=[], src_vocab='', src_word_vec_size=500, 
                start_decay_steps=50000, state_dim=512, tensorboard=False, 
                tensorboard_log_dir='runs/onmt', tgt_word_vec_size=500, train_from='',
                train_steps=5, transformer_ff=1024, truncated_decoder=0, 
                valid_batch_size=32, valid_steps=10000, warmup_steps=8000, 
                window_size=0.02, word_vec_size=512, world_size=1)

## multi-GPU
import torch
import random
## onmt
import onmt
from onmt.utils.logging import init_logger, logger
from onmt.inputters.inputter import build_dataset_iter, lazily_load_dataset, \
    _load_fields, _collect_report_features

## PALACE
import PALACE
from PALACE.utils.logging import init_logger, logger
