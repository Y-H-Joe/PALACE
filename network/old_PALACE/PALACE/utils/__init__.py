#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 17:51:59 2022

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
"""Module defining various utilities."""
from PALACE.utils.misc import split_corpus, aeq, use_gpu, set_random_seed
from PALACE.utils.alignment import make_batch_align_matrix
from PALACE.utils.report_manager import ReportMgr, build_report_manager
from PALACE.utils.statistics import Statistics
#from onmt.utils.optimizers import MultipleOptimizer,Optimizer, AdaFactor
from PALACE.utils.earlystopping import EarlyStopping, scorers_from_opts

__all__ = ["split_corpus", "aeq", "use_gpu", "set_random_seed", "ReportMgr",
           "build_report_manager", "Statistics",
           "MultipleOptimizer", "Optimizer", "AdaFactor", "EarlyStopping",
           "scorers_from_opts", "make_batch_align_matrix"]

"""
__all__ = ["split_corpus", "aeq", "use_gpu", "set_random_seed", "ReportMgr",
           "build_report_manager", "Statistics",
           "MultipleOptimizer", "Optimizer", "AdaFactor", "EarlyStopping",
           "scorers_from_opts", "make_batch_align_matrix"]
"""