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

opt = Namespace(accum_count=4, adagrad_accumulator_init=0, adam_beta1=0.9, 
                adam_beta2=0.998, audio_enc_pooling='1', batch_size=4096,
                batch_type='tokens', bridge=False, brnn=None, cnn_kernel_width=3, 
                context_gate=None, copy_attn=False, copy_attn_force=False, 
                copy_loss_by_seqlength=False, coverage_attn=False, data='data/', 
                dec_layers=2, dec_rnn_size=500, decay_method='noam', 
                decay_steps=10000, decoder_type='transformer', dropout=0.1, 
                enc_layers=2, enc_rnn_size=500, encoder_type='transformer', 
                epochs=0, exp='', exp_host='', feat_merge='concat', 
                feat_vec_exponent=0.7, feat_vec_size=-1, fix_word_vecs_dec=False, 
                fix_word_vecs_enc=False, generator_function='log_softmax', 
                global_attention='general', global_attention_function='softmax', 
                gpu_backend='nccl', gpu_ranks=[], gpu_verbose_level=0, gpuid=[], 
                heads=8, image_channel_size=3, input_feed=1, keep_checkpoint=20, 
                label_smoothing=0.0, lambda_coverage=1, layers=4, learning_rate=2.0,
                learning_rate_decay=0.5, log_file='', master_ip='localhost', 
                master_port=10000, max_generator_batches=32, max_grad_norm=0.0, 
                model_type='text', normalization='tokens', optim='adam', 
                param_init=0.0, param_init_glorot=True, position_encoding=True,
                pre_word_vecs_dec=None, pre_word_vecs_enc=None, report_every=1000, 
                reuse_copy_attn=False, rnn_size=256, rnn_type='LSTM',
                sample_rate=16000, save_checkpoint_steps=10,
                save_model='experiments/checkpoints_model', 
                seed=42, self_attn_type='scaled-dot', 
                share_decoder_embeddings=False, share_embeddings=True, 
                src_word_vec_size=500, start_decay_steps=50000, tensorboard=False, 
                tensorboard_log_dir='runs/onmt', tgt_word_vec_size=500, 
                train_from='', train_steps=5, transformer_ff=2048, 
                truncated_decoder=0, valid_batch_size=32, valid_steps=10000,
                warmup_steps=8000, window_size=0.02, word_vec_size=256, world_size=1)

## multi-GPU
import onmt
import torch
import random
from onmt.utils.logging import init_logger, logger
from onmt.inputters.inputter import build_dataset_iter, lazily_load_dataset, \
    _load_fields, _collect_report_features
    
def run(opt, device_id, error_queue):
    """ run process """
    try:
        gpu_rank = onmt.utils.distributed.multi_init(opt, device_id)
        if gpu_rank != opt.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")
        single_main(opt, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((opt.gpu_ranks[device_id], traceback.format_exc()))


def training_opt_postprocessing(opt, device_id):
    if opt.word_vec_size != -1:
        opt.src_word_vec_size = opt.word_vec_size
        opt.tgt_word_vec_size = opt.word_vec_size

    if opt.layers != -1:
        opt.enc_layers = opt.layers
        opt.dec_layers = opt.layers

    if opt.rnn_size != -1:
        opt.enc_rnn_size = opt.rnn_size
        opt.dec_rnn_size = opt.rnn_size
        if opt.model_type == 'text' and opt.enc_rnn_size != opt.dec_rnn_size:
            raise AssertionError("""We do not support different encoder and
                                 decoder rnn sizes for translation now.""")

    opt.brnn = (opt.encoder_type == "brnn")

    if opt.rnn_type == "SRU" and not opt.gpu_ranks:
        raise AssertionError("Using SRU requires -gpu_ranks set.")

    if torch.cuda.is_available() and not opt.gpu_ranks:
        logger.info("WARNING: You have a CUDA device, \
                    should run with -gpu_ranks")

    if opt.seed > 0:
        """
        为什么使用相同的网络结构，跑出来的效果完全不同，用的学习率，迭代次数，batch size 都是一样？
        固定随机数种子是非常重要的。但是如果你使用的是PyTorch等框架，还要看一下框架的种子是否固定了。
        还有，如果用了cuda，别忘了cuda的随机数种子。这里还需要用到torch.backends.cudnn.deterministic.
        """
        torch.manual_seed(opt.seed)
        # this one is needed for torchtext random call (shuffled iterator)
        # in multi gpu it ensures datasets are read in the same order
        random.seed(opt.seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        if opt.seed > 0:
            # These ensure same initialization in multi gpu mode
            torch.cuda.manual_seed(opt.seed)

    return opt


def single_main(opt, device_id):
    opt = training_opt_postprocessing(opt, device_id)
    init_logger(opt.log_file)
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
    else:
        checkpoint = None
        model_opt = opt

    # Peek the first dataset to determine the data_type.
    # (All datasets have the same data_type).
    """
    data_type: type of the source input. Options are [text|img|audio]
    """
    first_dataset = next(lazily_load_dataset("train", opt))
    data_type = first_dataset.data_type

    # Load fields generated from preprocess phase.
    fields = _load_fields(first_dataset, data_type, opt, checkpoint)

    # Report src/tgt features.

    src_features, tgt_features = _collect_report_features(fields)
    for j, feat in enumerate(src_features):
        logger.info(' * src feature %d size = %d'
                    % (j, len(fields[feat].vocab)))
    for j, feat in enumerate(tgt_features):
        logger.info(' * tgt feature %d size = %d'
                    % (j, len(fields[feat].vocab)))

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)
    _check_save_model_path(opt)

    # Build optimizer.
    optim = build_optim(model, opt, checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    trainer = build_trainer(opt, device_id, model, fields,
                            optim, data_type, model_saver=model_saver)

    def train_iter_fct(): return build_dataset_iter(
        lazily_load_dataset("train", opt), fields, opt)

    def valid_iter_fct(): return build_dataset_iter(
        lazily_load_dataset("valid", opt), fields, opt, is_train=False)

    # Do training.
    trainer.train(train_iter_fct, valid_iter_fct, opt.train_steps,
                  opt.valid_steps)

    if opt.tensorboard:
        trainer.report_manager.tensorboard_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)

    opt = parser.parse_args()
    main(opt)


nb_gpu = len(opt.gpu_ranks)

if opt.world_size > 1:
    mp = torch.multiprocessing.get_context('spawn')
    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)
    # Train with multiprocessing.
    procs = []
    for device_id in range(nb_gpu):
        procs.append(mp.Process(target=run, args=(
            opt, device_id, error_queue, ), daemon=True))
        procs[device_id].start()
        logger.info(" Starting process pid: %d  " % procs[device_id].pid)
        error_handler.add_child(procs[device_id].pid)
    for p in procs:
        p.join()

elif nb_gpu == 1:  # case 1 GPU only
    single_main(opt, 0)
else:   # case only CPU
    single_main(opt, -1)
