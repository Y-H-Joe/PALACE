#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 18:06:33 2022

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
"""Module defining inputters.

Inputters implement the logic of transforming raw data to vectorized inputs,
e.g., from a line of text to a sequence of embeddings.
"""
from PALACE.inputters.inputter import \
    load_old_vocab, get_fields, OrderedIterator, \
    build_vocab, old_style_vocab, filter_example
from PALACE.inputters.dataset_base import Dataset
#from onmt.inputters.image_dataset import img_sort_key, ImageDataReader
#from onmt.inputters.audio_dataset import audio_sort_key, AudioDataReader
from PALACE.inputters.text_dataset import text_sort_key, TextDataReader
from PALACE.inputters.vec_dataset import vec_sort_key, VecDataReader
from PALACE.inputters.datareader_base import DataReaderBase

str2reader = {
    "text": TextDataReader, "vec": VecDataReader}
str2sortkey = {
    'text': text_sort_key, 'vec': vec_sort_key}


__all__ = ['Dataset', 'load_old_vocab', 'get_fields', 'DataReaderBase',
           'filter_example', 'old_style_vocab',
           'build_vocab', 'OrderedIterator',
           'text_sort_key', 'vec_sort_key',
           'TextDataReader', 'ImageDataReader', 'AudioDataReader',
           'VecDataReader']

"""
str2reader = {
    "text": TextDataReader, "img": ImageDataReader, "audio": AudioDataReader,
    "vec": VecDataReader}
str2sortkey = {
    'text': text_sort_key, 'img': img_sort_key, 'audio': audio_sort_key,
    'vec': vec_sort_key}


__all__ = ['Dataset', 'load_old_vocab', 'get_fields', 'DataReaderBase',
           'filter_example', 'old_style_vocab',
           'build_vocab', 'OrderedIterator',
           'text_sort_key', 'img_sort_key', 'audio_sort_key', 'vec_sort_key',
           'TextDataReader', 'ImageDataReader', 'AudioDataReader',
           'VecDataReader']
"""