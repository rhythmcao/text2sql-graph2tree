#coding=utf8
import torch.nn as nn
from model.model_utils import Registrable, PoolingFunction
from model.encoder.graph_encoder import *
from model.decoder.sql_parser import *

@Registrable.register('text2sql')
class Text2SQL(nn.Module):

    def __init__(self, args, transition_system):
        super(Text2SQL, self).__init__()
        self.encoder = Registrable.by_name('encoder')(args)
        self.encoder2decoder = PoolingFunction(args.gnn_hidden_size, args.lstm_hidden_size, method='attentive-pooling')
        self.decoder = Registrable.by_name('decoder')(args, transition_system)

    def forward(self, batch, sample_size=5, gtol_size=5, n_best=1, order_method='all', cum_method='sum', method='gtol'):
        """ This function is used during training, which returns the entire training loss
        """
        encodings, value_memory, (vr_loss, gp_loss) = self.encoder(batch, sample_size=sample_size)
        h0 = self.encoder2decoder(encodings, mask=batch.mask)
        if method == 'gtol':
            loss, hyps = self.decoder.gtol_score(encodings, value_memory, h0, batch, beam_size=gtol_size, n_best=n_best, order_method=order_method, cum_method=cum_method)
            return (loss, hyps), vr_loss, gp_loss
        else:
            loss = self.decoder.score(encodings, value_memory, h0, batch)
            return loss, vr_loss, gp_loss

    def parse(self, batch, beam_size=5, order_method='controller'):
        """ This function is used for decoding, which returns a batch of hypothesis
        """
        encodings, value_memory, (value_candidates, schema_gate) = self.encoder(batch)
        h0 = self.encoder2decoder(encodings, mask=batch.mask)
        hyps = self.decoder.parse(encodings, value_memory, h0, batch, beam_size, order_method)
        return hyps, value_candidates, schema_gate

    def pad_embedding_grad_zero(self, index=None):
        """ For glove.42B.300d word vectors, gradients for <pad> symbol is always 0;
        Most words (starting from index) in the word vocab are also fixed except most frequent words
        """
        self.encoder.input_layer.pad_embedding_grad_zero(index)
