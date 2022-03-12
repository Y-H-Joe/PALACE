# -*- coding: utf-8 -*-
"""

@author: JChonpca_Huang

"""

import torch
from torch import nn
from d2ll import torch as d2l
from matplotlib import pyplot as plt

#@save
class AttentionDecoder(d2l.Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError

class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        
        self.attention = d2l.AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_lens = state
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights



embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1

batch_size, num_steps = 64, 10

lr, num_epochs, device = 0.005, 100, d2l.try_gpu()

train_iter, test_iter, src_vocab, tgt_vocab, source, target = d2l.load_data_nmt(batch_size, num_steps)

encoder = d2l.Seq2SeqEncoder(
    len(src_vocab), embed_size, num_hiddens, num_layers, dropout)

decoder = Seq2SeqAttentionDecoder(
    len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)

net = d2l.EncoderDecoder(encoder, decoder)

metric = d2l.train_seq2seq(net, train_iter,test_iter, lr, num_epochs, tgt_vocab, device)

plt.plot(metric[0][:,0],metric[0][:,1])

plt.plot(metric[1][:,0],metric[1][:,1])

plt.plot(metric[0][:,0],metric[0][:,1],label='train')

plt.plot(metric[1][:,0],metric[1][:,1],label='test')

plt.legend()

plt.show()