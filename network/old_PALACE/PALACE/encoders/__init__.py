#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 14:07:39 2022

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
"""Module defining encoders."""
from PALACE.encoders.encoder import EncoderBase
from PALACE.encoders.transformer import TransformerEncoder
#from onmt.encoders.ggnn_encoder import GGNNEncoder
#from onmt.encoders.rnn_encoder import RNNEncoder
#from onmt.encoders.cnn_encoder import CNNEncoder
#from onmt.encoders.mean_encoder import MeanEncoder
#from onmt.encoders.audio_encoder import AudioEncoder
#from onmt.encoders.image_encoder import ImageEncoder

str2enc = { "transformer": TransformerEncoder}

__all__ = ["EncoderBase", "TransformerEncoder","str2enc"]
"""
str2enc = {"ggnn": GGNNEncoder, "rnn": RNNEncoder, "brnn": RNNEncoder,
           "cnn": CNNEncoder, "transformer": TransformerEncoder,
           "img": ImageEncoder, "audio": AudioEncoder, "mean": MeanEncoder}

__all__ = ["EncoderBase", "TransformerEncoder", "RNNEncoder", "CNNEncoder",
           "MeanEncoder", "str2enc"]
"""