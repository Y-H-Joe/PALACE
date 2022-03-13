#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 15:50:24 2022

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
import codecs #面对复杂文本的读取，尤其是爬虫获得的以及原始的复杂文本，使用open读取后编码不统一的情况，建议用codecs.open()
import glob #glob是python自己带的一个文件操作相关模块，用它可以查找符合自己目的的文件,主要方法就是glob,该方法返回所有匹配的文件路径列表
from collections import Counter, defaultdict
import gc # garbage collection
from argparse import Namespace
from functools import partial #拓展函数的args与kwargs,让原本不兼容的代码可以一起工作
from multiprocessing import Pool

import torch

from PALACE.utils.parse import ArgumentParser
from PALACE.utils.misc import split_corpus
from PALACE.opts import config_opts , preprocess_opts
from PALACE.utils.logging import init_logger, logger
from PALACE.inputters import Dataset,str2sortkey,filter_example,get_fields,str2reader
from PALACE.inputters.inputter import _load_vocab,_build_fields_vocab,old_style_vocab,load_old_vocab

opt = Namespace(config=None, data_type='text', dynamic_dict=False, features_vocabs_prefix='', 
                filter_valid=False, image_channel_size=3, log_file='', log_file_level='0', 
                lower=False, max_shard_size=0, num_threads=32, overwrite=True, 
                report_every=100000, sample_rate=16000, save_config=None,
                save_data='data/', seed=3435, shard_size=1000000, share_vocab=True, 
                shuffle=0, src_dir='', src_seq_length=10000, src_seq_length_trunc=None,
                src_vocab='', src_vocab_size=10000, src_words_min_frequency=0, 
                subword_prefix='▁', subword_prefix_is_joiner=False, tgt_seq_length=10000, 
                tgt_seq_length_trunc=None, tgt_vocab='', tgt_vocab_size=10000, 
                tgt_words_min_frequency=0, train_align=[None], train_ids=[None],
                train_src=['data/src-train.txtt'], train_tgt=['data/tgt-train.txtt'],
                valid_align=None, valid_src='data/src-val.txtt', 
                valid_tgt='data/tgt-val.txtt', vocab_size_multiple=1, window='hamming',
                window_size=0.02, window_stride=0.01)


# =============================================================================
# def _get_parser():
#     parser = ArgumentParser(description='preprocess.py')
# 
#     config_opts(parser)
#     preprocess_opts(parser)
#     return parser
# =============================================================================


def maybe_load_vocab(corpus_type, counters, opt):
    src_vocab = None
    tgt_vocab = None
    existing_fields = None
    if corpus_type == "train":
        if opt.src_vocab != "":
            try:
                logger.info("Using existing vocabulary...")
                existing_fields = torch.load(opt.src_vocab)
            except torch.serialization.pickle.UnpicklingError:
                logger.info("Building vocab from text file...")
                src_vocab, src_vocab_size = _load_vocab(
                    opt.src_vocab, "src", counters,
                    opt.src_words_min_frequency)
        if opt.tgt_vocab != "":
            tgt_vocab, tgt_vocab_size = _load_vocab(
                opt.tgt_vocab, "tgt", counters,
                opt.tgt_words_min_frequency)
    return src_vocab, tgt_vocab, existing_fields


def check_existing_pt_files(opt, corpus_type, ids, existing_fields):
    """ Check if there are existing .pt files to avoid overwriting them """
    existing_shards = []
    for maybe_id in ids:
        if maybe_id:
            shard_base = corpus_type + "_" + maybe_id
        else:
            shard_base = corpus_type
        pattern = opt.save_data + '.{}.*.pt'.format(shard_base)
        if glob.glob(pattern):
            if opt.overwrite:
                maybe_overwrite = ("will be overwritten because "
                                   "`-overwrite` option is set.")
            else:
                maybe_overwrite = ("won't be overwritten, pass the "
                                   "`-overwrite` option if you want to.")
            logger.warning("Shards for corpus {} already exist, {}"
                           .format(shard_base, maybe_overwrite))
            existing_shards += [maybe_id]
    return existing_shards

def process_one_shard(corpus_params, params):
    corpus_type, fields, src_reader, tgt_reader, align_reader, opt,\
         existing_fields, src_vocab, tgt_vocab = corpus_params
    i, (src_shard, tgt_shard, align_shard, maybe_id, filter_pred) = params
    # create one counter per shard
    sub_sub_counter = defaultdict(Counter)
    assert len(src_shard) == len(tgt_shard)
    logger.info("Building shard %d." % i)

    src_data = {"reader": src_reader, "data": src_shard, "dir": opt.src_dir}
    tgt_data = {"reader": tgt_reader, "data": tgt_shard, "dir": None}
    align_data = {"reader": align_reader, "data": align_shard, "dir": None}
    _readers, _data, _dir = Dataset.config(
        [('src', src_data), ('tgt', tgt_data), ('align', align_data)])

    dataset = Dataset(
        fields, readers=_readers, data=_data, dirs=_dir,
        sort_key=str2sortkey[opt.data_type],
        filter_pred=filter_pred,
        corpus_id=maybe_id
    )
    if corpus_type == "train" and existing_fields is None:
        for ex in dataset.examples:
            sub_sub_counter['corpus_id'].update(
                ["train" if maybe_id is None else maybe_id])
            for name, field in fields.items():
                if (opt.data_type in ["audio", "vec"]) and name == "src":
                    continue
                try:
                    f_iter = iter(field)
                except TypeError:
                    f_iter = [(name, field)]
                    all_data = [getattr(ex, name, None)]
                else:
                    all_data = getattr(ex, name)
                for (sub_n, sub_f), fd in zip(
                        f_iter, all_data):
                    has_vocab = (sub_n == 'src' and
                                 src_vocab is not None) or \
                                (sub_n == 'tgt' and
                                 tgt_vocab is not None)
                    if (hasattr(sub_f, 'sequential')
                            and sub_f.sequential and not has_vocab):
                        val = fd
                        sub_sub_counter[sub_n].update(val)
    if maybe_id:
        shard_base = corpus_type + "_" + maybe_id
    else:
        shard_base = corpus_type
    data_path = "{:s}.{:s}.{:d}.pt".\
        format(opt.save_data, shard_base, i)

    logger.info(" * saving %sth %s data shard to %s."
                % (i, shard_base, data_path))

    dataset.save(data_path)

    del dataset.examples
    gc.collect()
    del dataset
    gc.collect()

    return sub_sub_counter


def build_save_dataset(corpus_type, fields, src_reader, tgt_reader,
                       align_reader, opt):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        counters = defaultdict(Counter)
        srcs = opt.train_src
        tgts = opt.train_tgt
        ids = opt.train_ids
        aligns = opt.train_align
    elif corpus_type == 'valid':
        counters = None
        srcs = [opt.valid_src]
        tgts = [opt.valid_tgt]
        ids = [None]
        aligns = [opt.valid_align]

    src_vocab, tgt_vocab, existing_fields = maybe_load_vocab(
        corpus_type, counters, opt)

    existing_shards = check_existing_pt_files(
        opt, corpus_type, ids, existing_fields)

    # every corpus has shards, no new one
    if existing_shards == ids and not opt.overwrite:
        return

    def shard_iterator(srcs, tgts, ids, aligns, existing_shards,
                       existing_fields, corpus_type, opt):
        """
        Builds a single iterator yielding every shard of every corpus.
        """
        for src, tgt, maybe_id, maybe_align in zip(srcs, tgts, ids, aligns):
            if maybe_id in existing_shards:
                if opt.overwrite:
                    logger.warning("Overwrite shards for corpus {}"
                                   .format(maybe_id))
                else:
                    if corpus_type == "train":
                        assert existing_fields is not None,\
                            ("A 'vocab.pt' file should be passed to "
                             "`-src_vocab` when adding a corpus to "
                             "a set of already existing shards.")
                    logger.warning("Ignore corpus {} because "
                                   "shards already exist"
                                   .format(maybe_id))
                    continue
            if ((corpus_type == "train" or opt.filter_valid)
                    and tgt is not None):
                filter_pred = partial(
                    filter_example,
                    use_src_len=opt.data_type == "text",
                    max_src_len=opt.src_seq_length,
                    max_tgt_len=opt.tgt_seq_length)
            else:
                filter_pred = None
            src_shards = split_corpus(src, opt.shard_size)
            tgt_shards = split_corpus(tgt, opt.shard_size)
            align_shards = split_corpus(maybe_align, opt.shard_size)
            for i, (ss, ts, a_s) in enumerate(
                    zip(src_shards, tgt_shards, align_shards)):
                yield (i, (ss, ts, a_s, maybe_id, filter_pred))

    shard_iter = shard_iterator(srcs, tgts, ids, aligns, existing_shards,
                                existing_fields, corpus_type, opt)

    with Pool(opt.num_threads) as p:
        dataset_params = (corpus_type, fields, src_reader, tgt_reader,
                          align_reader, opt, existing_fields,
                          src_vocab, tgt_vocab)
        func = partial(process_one_shard, dataset_params)
        for sub_counter in p.imap(func, shard_iter):
            if sub_counter is not None:
                for key, value in sub_counter.items():
                    counters[key].update(value)

    if corpus_type == "train":
        vocab_path = opt.save_data + '.vocab.pt'
        new_fields = _build_fields_vocab(
            fields, counters, opt.data_type,
            opt.share_vocab, opt.vocab_size_multiple,
            opt.src_vocab_size, opt.src_words_min_frequency,
            opt.tgt_vocab_size, opt.tgt_words_min_frequency,
            subword_prefix=opt.subword_prefix,
            subword_prefix_is_joiner=opt.subword_prefix_is_joiner)
        if existing_fields is None:
            fields = new_fields
        else:
            fields = existing_fields

        if old_style_vocab(fields):
            fields = load_old_vocab(
                fields, opt.data_type, dynamic_dict=opt.dynamic_dict)

        # patch corpus_id
        if fields.get("corpus_id", False):
            fields["corpus_id"].vocab = new_fields["corpus_id"].vocab_cls(
                counters["corpus_id"])

        torch.save(fields, vocab_path)

def count_features(path):
    """
    path: location of a corpus file with whitespace-delimited tokens and
                    ￨-delimited features within the token
    returns: the number of features in the dataset
    """
    with codecs.open(path, "r", "utf-8") as f:
        # None (the default sep) means split according to any whitespace
        # 1 in split is maxsplit, Maximum number of splits to do
        first_tok = f.readline().split(None, 1)[0] 
        return len(first_tok.split(u"￨")) - 1

def preprocess(opt):
    ArgumentParser.validate_preprocess_args(opt)
    torch.manual_seed(opt.seed)

    init_logger(opt.log_file)

    logger.info("Extracting features...")

    src_nfeats = 0
    tgt_nfeats = 0
    src_nfeats = count_features(opt.train_src[0]) if opt.data_type == 'text' else 0
    tgt_nfeats = count_features(opt.train_tgt[0])  # tgt always text so far
    if len(opt.train_src) > 1 and opt.data_type == 'text':
        for src, tgt in zip(opt.train_src[1:], opt.train_tgt[1:]):
            assert src_nfeats == count_features(src),\
                "%s seems to mismatch features of "\
                "the other source datasets" % src
            assert tgt_nfeats == count_features(tgt),\
                "%s seems to mismatch features of "\
                "the other target datasets" % tgt
    logger.info(" * number of source features: %d." % src_nfeats)
    logger.info(" * number of target features: %d." % tgt_nfeats)

    logger.info("Building `Fields` object...")
    fields = get_fields(
        opt.data_type,
        src_nfeats,
        tgt_nfeats,
        dynamic_dict=opt.dynamic_dict,
        with_align=opt.train_align[0] is not None,
        src_truncate=opt.src_seq_length_trunc,
        tgt_truncate=opt.tgt_seq_length_trunc)

    src_reader = str2reader[opt.data_type].from_opt(opt)
    tgt_reader = str2reader["text"].from_opt(opt)
    align_reader = str2reader["text"].from_opt(opt)

    logger.info("Building & saving training data...")
    build_save_dataset(
        'train', fields, src_reader, tgt_reader, align_reader, opt)

    if opt.valid_src and opt.valid_tgt:
        logger.info("Building & saving validation data...")
        build_save_dataset(
            'valid', fields, src_reader, tgt_reader, align_reader, opt)

def main():
    #parser = _get_parser()

    #opt = parser.parse_args()
    #print(opt)
    preprocess(opt)

if __name__ == '__main__':
    main()
    