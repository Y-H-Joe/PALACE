#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 18:00:18 2022

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
from itertools import islice, repeat

def split_corpus(path, shard_size, default=None):
    """yield a `list` containing `shard_size` line of `path`,
    or repeatly generate `default` if `path` is None.
    """
    if path is not None:
        return _split_corpus(path, shard_size)
    else:
        return repeat(default)


def _split_corpus(path, shard_size):
    """Yield a `list` containing `shard_size` line of `path`.
    """
    with open(path, "rb") as f:
        if shard_size <= 0:
            yield f.readlines()
        else:
            while True:
                shard = list(islice(f, shard_size))
                if not shard:
                    break
                yield shard


