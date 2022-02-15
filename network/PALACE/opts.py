#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 16:27:39 2022

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
import configargparse

from PALACE.models.sru import CheckSRU


def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """

    # Embedding Options
    group = parser.add_argument_group('Model-Embeddings')
    group.add('--src_word_vec_size', '-src_word_vec_size',
              type=int, default=500,
              help='Word embedding size for src.')
    group.add('--tgt_word_vec_size', '-tgt_word_vec_size',
              type=int, default=500,
              help='Word embedding size for tgt.')
    group.add('--word_vec_size', '-word_vec_size', type=int, default=-1,
              help='Word embedding size for src and tgt.')

    group.add('--share_decoder_embeddings', '-share_decoder_embeddings',
              action='store_true',
              help="Use a shared weight matrix for the input and "
                   "output word  embeddings in the decoder.")
    group.add('--share_embeddings', '-share_embeddings', action='store_true',
              help="Share the word embeddings between encoder "
                   "and decoder. Need to use shared dictionary for this "
                   "option.")
    group.add('--position_encoding', '-position_encoding', action='store_true',
              help="Use a sin to mark relative words positions. "
                   "Necessary for non-RNN style models.")

    group = parser.add_argument_group('Model-Embedding Features')
    group.add('--feat_merge', '-feat_merge', type=str, default='concat',
              choices=['concat', 'sum', 'mlp'],
              help="Merge action for incorporating features embeddings. "
                   "Options [concat|sum|mlp].")
    group.add('--feat_vec_size', '-feat_vec_size', type=int, default=-1,
              help="If specified, feature embedding sizes "
                   "will be set to this. Otherwise, feat_vec_exponent "
                   "will be used.")
    group.add('--feat_vec_exponent', '-feat_vec_exponent',
              type=float, default=0.7,
              help="If -feat_merge_size is not set, feature "
                   "embedding sizes will be set to N^feat_vec_exponent "
                   "where N is the number of values the feature takes.")

    # Encoder-Decoder Options
    group = parser.add_argument_group('Model- Encoder-Decoder')
    group.add('--model_type', '-model_type', default='text',
              choices=['text', 'img', 'audio', 'vec'],
              help="Type of source model to use. Allows "
                   "the system to incorporate non-text inputs. "
                   "Options are [text|img|audio|vec].")
    group.add('--model_dtype', '-model_dtype', default='fp32',
              choices=['fp32', 'fp16'],
              help='Data type of the model.')

    group.add('--encoder_type', '-encoder_type', type=str, default='rnn',
              choices=['rnn', 'brnn', 'ggnn', 'mean', 'transformer', 'cnn'],
              help="Type of encoder layer to use. Non-RNN layers "
                   "are experimental. Options are "
                   "[rnn|brnn|ggnn|mean|transformer|cnn].")
    group.add('--decoder_type', '-decoder_type', type=str, default='rnn',
              choices=['rnn', 'transformer', 'cnn'],
              help="Type of decoder layer to use. Non-RNN layers "
                   "are experimental. Options are "
                   "[rnn|transformer|cnn].")

    group.add('--layers', '-layers', type=int, default=-1,
              help='Number of layers in enc/dec.')
    group.add('--enc_layers', '-enc_layers', type=int, default=2,
              help='Number of layers in the encoder')
    group.add('--dec_layers', '-dec_layers', type=int, default=2,
              help='Number of layers in the decoder')
    group.add('--rnn_size', '-rnn_size', type=int, default=-1,
              help="Size of rnn hidden states. Overwrites "
                   "enc_rnn_size and dec_rnn_size")
    group.add('--enc_rnn_size', '-enc_rnn_size', type=int, default=500,
              help="Size of encoder rnn hidden states. "
                   "Must be equal to dec_rnn_size except for "
                   "speech-to-text.")
    group.add('--dec_rnn_size', '-dec_rnn_size', type=int, default=500,
              help="Size of decoder rnn hidden states. "
                   "Must be equal to enc_rnn_size except for "
                   "speech-to-text.")
    group.add('--audio_enc_pooling', '-audio_enc_pooling',
              type=str, default='1',
              help="The amount of pooling of audio encoder, "
                   "either the same amount of pooling across all layers "
                   "indicated by a single number, or different amounts of "
                   "pooling per layer separated by comma.")
    group.add('--cnn_kernel_width', '-cnn_kernel_width', type=int, default=3,
              help="Size of windows in the cnn, the kernel_size is "
                   "(cnn_kernel_width, 1) in conv layer")

    group.add('--input_feed', '-input_feed', type=int, default=1,
              help="Feed the context vector at each time step as "
                   "additional input (via concatenation with the word "
                   "embeddings) to the decoder.")
    group.add('--bridge', '-bridge', action="store_true",
              help="Have an additional layer between the last encoder "
                   "state and the first decoder state")
    group.add('--rnn_type', '-rnn_type', type=str, default='LSTM',
              choices=['LSTM', 'GRU', 'SRU'],
              action=CheckSRU,
              help="The gate type to use in the RNNs")
    # group.add('--residual', '-residual',   action="store_true",
    #                     help="Add residual connections between RNN layers.")

    group.add('--brnn', '-brnn', action=DeprecateAction,
              help="Deprecated, use `encoder_type`.")

    group.add('--context_gate', '-context_gate', type=str, default=None,
              choices=['source', 'target', 'both'],
              help="Type of context gate to use. "
                   "Do not select for no context gate.")

    # The following options (bridge_extra_node to src_vocab) are used
    # for training with --encoder_type ggnn (Gated Graph Neural Network).
    group.add('--bridge_extra_node', '-bridge_extra_node',
              type=bool, default=True,
              help='Graph encoder bridges only extra node to decoder as input')
    group.add('--bidir_edges', '-bidir_edges', type=bool, default=True,
              help='Graph encoder autogenerates bidirectional edges')
    group.add('--state_dim', '-state_dim', type=int, default=512,
              help='Number of state dimensions in the graph encoder')
    group.add('--n_edge_types', '-n_edge_types', type=int, default=2,
              help='Number of edge types in the graph encoder')
    group.add('--n_node', '-n_node', type=int, default=2,
              help='Number of nodes in the graph encoder')
    group.add('--n_steps', '-n_steps', type=int, default=2,
              help='Number of steps to advance graph encoder')
    # The ggnn uses src_vocab during training because the graph is built
    # using edge information which requires parsing the input sequence.
    group.add('--src_vocab', '-src_vocab', default="",
              help="Path to an existing source vocabulary. Format: "
                   "one word per line.")


class DeprecateAction(configargparse.Action):
    """ Deprecate action """

    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(DeprecateAction, self).__init__(option_strings, dest, nargs=0,
                                              help=help, **kwargs)

    def __call__(self, parser, namespace, values, flag_name):
        help = self.help if self.help is not None else ""
        msg = "Flag '%s' is deprecated. %s" % (flag_name, help)
        raise configargparse.ArgumentTypeError(msg)


def config_opts(parser):
    parser.add('-config', '--config', required=False,
               is_config_file_arg=True, help='config file path')
    parser.add('-save_config', '--save_config', required=False,
               is_write_out_config_file_arg=True,
               help='config file save path')


def preprocess_opts(parser):
    """ Pre-procesing options """
    # Data options
    group = parser.add_argument_group('Data')
    group.add('--data_type', '-data_type', default="text",
              help="Type of the source input. "
                   "Options are [text|img|audio|vec].")

    group.add('--train_src', '-train_src', required=True, nargs='+',
              help="Path(s) to the training source data")
    group.add('--train_tgt', '-train_tgt', required=True, nargs='+',
              help="Path(s) to the training target data")
    group.add('--train_align', '-train_align', nargs='+', default=[None],
              help="Path(s) to the training src-tgt alignment")
    group.add('--train_ids', '-train_ids', nargs='+', default=[None],
              help="ids to name training shards, used for corpus weighting")

    group.add('--valid_src', '-valid_src',
              help="Path to the validation source data")
    group.add('--valid_tgt', '-valid_tgt',
              help="Path to the validation target data")
    group.add('--valid_align', '-valid_align', default=None,
              help="Path(s) to the validation src-tgt alignment")

    group.add('--src_dir', '-src_dir', default="",
              help="Source directory for image or audio files.")

    group.add('--save_data', '-save_data', required=True,
              help="Output file for the prepared data")

    group.add('--max_shard_size', '-max_shard_size', type=int, default=0,
              help="""Deprecated use shard_size instead""")

    group.add('--shard_size', '-shard_size', type=int, default=1000000,
              help="Divide src_corpus and tgt_corpus into "
                   "smaller multiple src_copus and tgt corpus files, then "
                   "build shards, each shard will have "
                   "opt.shard_size samples except last shard. "
                   "shard_size=0 means no segmentation "
                   "shard_size>0 means segment dataset into multiple shards, "
                   "each shard has shard_size samples")

    group.add('--num_threads', '-num_threads', type=int, default=1,
              help="Number of shards to build in parallel.")

    group.add('--overwrite', '-overwrite', action="store_true",
              help="Overwrite existing shards if any.")

    # Dictionary options, for text corpus

    group = parser.add_argument_group('Vocab')
    # if you want to pass an existing vocab.pt file, pass it to
    # -src_vocab alone as it already contains tgt vocab.
    group.add('--src_vocab', '-src_vocab', default="",
              help="Path to an existing source vocabulary. Format: "
                   "one word per line.")
    group.add('--tgt_vocab', '-tgt_vocab', default="",
              help="Path to an existing target vocabulary. Format: "
                   "one word per line.")
    group.add('--features_vocabs_prefix', '-features_vocabs_prefix',
              type=str, default='',
              help="Path prefix to existing features vocabularies")
    group.add('--src_vocab_size', '-src_vocab_size', type=int, default=50000,
              help="Size of the source vocabulary")
    group.add('--tgt_vocab_size', '-tgt_vocab_size', type=int, default=50000,
              help="Size of the target vocabulary")
    group.add('--vocab_size_multiple', '-vocab_size_multiple',
              type=int, default=1,
              help="Make the vocabulary size a multiple of this value")

    group.add('--src_words_min_frequency',
              '-src_words_min_frequency', type=int, default=0)
    group.add('--tgt_words_min_frequency',
              '-tgt_words_min_frequency', type=int, default=0)

    group.add('--dynamic_dict', '-dynamic_dict', action='store_true',
              help="Create dynamic dictionaries")
    group.add('--share_vocab', '-share_vocab', action='store_true',
              help="Share source and target vocabulary")

    # Truncation options, for text corpus
    group = parser.add_argument_group('Pruning')
    group.add('--src_seq_length', '-src_seq_length', type=int, default=50,
              help="Maximum source sequence length")
    group.add('--src_seq_length_trunc', '-src_seq_length_trunc',
              type=int, default=None,
              help="Truncate source sequence length.")
    group.add('--tgt_seq_length', '-tgt_seq_length', type=int, default=50,
              help="Maximum target sequence length to keep.")
    group.add('--tgt_seq_length_trunc', '-tgt_seq_length_trunc',
              type=int, default=None,
              help="Truncate target sequence length.")
    group.add('--lower', '-lower', action='store_true', help='lowercase data')
    group.add('--filter_valid', '-filter_valid', action='store_true',
              help='Filter validation data by src and/or tgt length')

    # Data processing options
    group = parser.add_argument_group('Random')
    group.add('--shuffle', '-shuffle', type=int, default=0,
              help="Shuffle data")
    group.add('--seed', '-seed', type=int, default=3435,
              help="Random seed")

    group = parser.add_argument_group('Logging')
    group.add('--report_every', '-report_every', type=int, default=100000,
              help="Report status every this many sentences")
    group.add('--log_file', '-log_file', type=str, default="",
              help="Output logs to a file under this path.")
    group.add('--log_file_level', '-log_file_level', type=str,
              action=StoreLoggingLevelAction,
              choices=StoreLoggingLevelAction.CHOICES,
              default="0")

    # Options most relevant to speech
    group = parser.add_argument_group('Speech')
    group.add('--sample_rate', '-sample_rate', type=int, default=16000,
              help="Sample rate.")
    group.add('--window_size', '-window_size', type=float, default=.02,
              help="Window size for spectrogram in seconds.")
    group.add('--window_stride', '-window_stride', type=float, default=.01,
              help="Window stride for spectrogram in seconds.")
    group.add('--window', '-window', default='hamming',
              help="Window type for spectrogram generation.")

    # Option most relevant to image input
    group.add('--image_channel_size', '-image_channel_size',
              type=int, default=3,
              choices=[3, 1],
              help="Using grayscale image can training "
                   "model faster and smaller")

    # Options for experimental source noising (BART style)
    group = parser.add_argument_group('Noise')
    group.add('--subword_prefix', '-subword_prefix',
              type=str, default="▁",
              help="subword prefix to build wordstart mask")
    group.add('--subword_prefix_is_joiner', '-subword_prefix_is_joiner',
              action='store_true',
              help="mask will need to be inverted if prefix is joiner")

class StoreLoggingLevelAction(configargparse.Action):
    """ Convert string to logging level """
    import logging
    LEVELS = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET
    }

    CHOICES = list(LEVELS.keys()) + [str(_) for _ in LEVELS.values()]

    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(StoreLoggingLevelAction, self).__init__(
            option_strings, dest, help=help, **kwargs)

    def __call__(self, parser, namespace, value, option_string=None):
        # Get the key 'value' in the dict, or just use 'value'
        level = StoreLoggingLevelAction.LEVELS.get(value, value)
        setattr(namespace, self.dest, level)



