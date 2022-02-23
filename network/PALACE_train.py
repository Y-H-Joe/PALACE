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
import os
import signal  # signal可以被用来进程间通信和异步处理
from itertools import cycle

import torch

#from PALACE.opts import config_opts,model_opts,train_opts
from PALACE.utils.parse import ArgumentParser
from PALACE.utils.misc import set_random_seed
from PALACE.utils.logging import init_logger, logger
from PALACE.inputters.inputter import build_dataset_iter, patch_fields, \
    load_old_vocab, old_style_vocab, build_dataset_iter_multiple
from PALACE.utils.distributed import multi_init
from PALACE.train_single import main as single_main

# Fix CPU tensor sharing strategy
torch.multiprocessing.set_sharing_strategy('file_system')
# world_size = 4, gpu_ranks = [0,1,2,3]
# 多机多卡 refer to https://zhuanlan.zhihu.com/p/38949622
opt = Namespace(aan_useffn=False, accum_count=[4], accum_steps=[0], 
                adagrad_accumulator_init=0, adam_beta1=0.9, adam_beta2=0.998, 
                alignment_heads=0, alignment_layer=-3, apex_opt_level='O1', 
                attention_dropout=[0.1], audio_enc_pooling='1', average_decay=0,
                average_every=1, batch_size=8, batch_size_multiple=None, 
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
                window_size=0.02, word_vec_size=512, world_size=1,
                protein_encoding=True)

class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)

def run(opt, device_id, error_queue, batch_queue, semaphore):
    """ run process """
    try:
        gpu_rank = multi_init(opt, device_id)
        if gpu_rank != opt.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")
        single_main(opt, device_id, batch_queue, semaphore)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((opt.gpu_ranks[device_id], traceback.format_exc()))


def batch_producer(generator_to_serve, queues, semaphore, opt):
    init_logger(opt.log_file)
    set_random_seed(opt.seed, False)
    # generator_to_serve = iter(generator_to_serve)

    def pred(x):
        """
        Filters batches that belong only
        to gpu_ranks of current node
        """
        for rank in opt.gpu_ranks:
            if x[0] % opt.world_size == rank:
                return True

    generator_to_serve = filter(
        pred, enumerate(generator_to_serve))

    def next_batch(device_id):
        new_batch = next(generator_to_serve)
        semaphore.acquire()
        return new_batch[1]

    b = next_batch(0)

    for device_id, q in cycle(enumerate(queues)):
        b.dataset = None
        # hack to dodge unpicklable `dict_keys`
        b.fields = list(b.fields)
        q.put(b)
        b = next_batch(device_id)

def train(opt):
    ArgumentParser.validate_train_opts(opt)
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)

    set_random_seed(opt.seed, False)

    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        vocab = checkpoint['vocab']
    else:
        vocab = torch.load(opt.data + '.vocab.pt')

    # check for code where vocab is saved instead of fields
    # (in the future this will be done in a smarter way)
    if old_style_vocab(vocab):
        fields = load_old_vocab(
            vocab, opt.model_type, dynamic_dict=opt.copy_attn)
    else:
        fields = vocab

    # patch for fields that may be missing in old data/model
    patch_fields(opt, fields)

    if len(opt.data_ids) > 1:
        train_shards = []
        for train_id in opt.data_ids:
            shard_base = "train_" + train_id
            train_shards.append(shard_base)
        train_iter = build_dataset_iter_multiple(train_shards, fields, opt)
    else:
        if opt.data_ids[0] is not None:
            shard_base = "train_" + opt.data_ids[0]
        else:
            shard_base = "train"
        #print(opt)
        train_iter = build_dataset_iter(shard_base, fields, opt)

    nb_gpu = len(opt.gpu_ranks)

    if opt.world_size > 1:
        queues = []
        mp = torch.multiprocessing.get_context('spawn')
        # Semaphore管理一个内置的计数器，每当调用acquire()时内置计数器-1；调用release() 时内置计数器+1
        semaphore = mp.Semaphore(opt.world_size * opt.queue_size)
        # Create a thread to listen for errors in the child processes.
        error_queue = mp.SimpleQueue()
        error_handler = ErrorHandler(error_queue)
        # Train with multiprocessing.
        procs = []
        for device_id in range(nb_gpu):
            q = mp.Queue(opt.queue_size)
            queues += [q]
            procs.append(mp.Process(target=run, args=(
                opt, device_id, error_queue, q, semaphore), daemon=True))
            procs[device_id].start()
            logger.info(" Starting process pid: %d  " % procs[device_id].pid)
            error_handler.add_child(procs[device_id].pid)
        producer = mp.Process(target=batch_producer,
                              args=(train_iter, queues, semaphore, opt,),
                              daemon=True)
        producer.start()
        error_handler.add_child(producer.pid)

        for p in procs:
            p.join()
        producer.terminate()

    elif nb_gpu == 1:  # case 1 GPU only
        single_main(opt, 0)
    else:   # case only CPU
        single_main(opt, -1)

# =============================================================================
# def _get_parser():
#     parser = ArgumentParser(description='PALACE_train.py')
# 
#     config_opts(parser)
#     model_opts(parser)
#     train_opts(parser)
#     return parser
# =============================================================================

def main():
    #parser = _get_parser()
    #opt = parser.parse_args()
    print(opt)
    train(opt)

if __name__ == "__main__":
    main()