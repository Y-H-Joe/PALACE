#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 14:46:45 2022

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

from PALACE.modules.embeddings import Embeddings, PositionalEncoding, VecEmbedding, ProteinEncoding
from PALACE.modules.multi_headed_attn import MultiHeadedAttention
from PALACE.modules.average_attn import AverageAttention
from PALACE.modules.copy_generator import CopyGenerator,CopyGeneratorLoss,CopyGeneratorLossCompute


"""
编写库时，经常会在 __init__.py 中暴露整个包的 API，而这些 API 的实现可能是在包的其他模块中。
如果仅仅这样写：from xxx import a, b，一些代码检查工具，如 pyflakes 会报错，认为变量 a和 b 
import 了但没被使用。一个可行的方法是把这个警告压掉：from xxx import a, b # noqa 
（No Q/A，即无质量保证），但更好的方法是显式定义 __all__，这样代码检查工具就会理解，从而不再报
 unused variables 的警告。
"""

__all__ = ["Elementwise", "context_gate_factory", "ContextGate",
           "GlobalAttention", "ConvMultiStepAttention", "CopyGenerator",
           "CopyGeneratorLoss", "CopyGeneratorLossCompute",
           "MultiHeadedAttention", "Embeddings", "PositionalEncoding",
           "WeightNormConv2d", "AverageAttention", "VecEmbedding","ProteinEncoding"]

"""
from onmt.modules.util_class import Elementwise
from onmt.modules.gate import context_gate_factory, ContextGate
from onmt.modules.global_attention import GlobalAttention
from onmt.modules.conv_multi_step_attention import ConvMultiStepAttention
from onmt.modules.copy_generator import CopyGenerator, CopyGeneratorLoss, \
    CopyGeneratorLossCompute
from onmt.modules.multi_headed_attn import MultiHeadedAttention
from onmt.modules.embeddings import Embeddings, PositionalEncoding, \
    VecEmbedding
from onmt.modules.weight_norm import WeightNormConv2d
from onmt.modules.average_attn import AverageAttention

import onmt.modules.source_noise # noqa

__all__ = ["Elementwise", "context_gate_factory", "ContextGate",
           "GlobalAttention", "ConvMultiStepAttention", "CopyGenerator",
           "CopyGeneratorLoss", "CopyGeneratorLossCompute",
           "MultiHeadedAttention", "Embeddings", "PositionalEncoding",
           "WeightNormConv2d", "AverageAttention", "VecEmbedding"]
"""