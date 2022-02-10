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


    def forward(self, batch, sample_size=5, gtl_size=4, n_best=1, ts_order='random', uts_order='enum'):
        """ This function is used during training, which returns the entire training loss
        """
        encodings, value_memory, (vr_loss, gp_loss) = self.encoder(batch, sample_size=sample_size)
        h0 = self.encoder2decoder(encodings, mask=batch.mask)
        if ts_order == 'enum' or uts_order == 'enum':
            ast_loss, hyps = self.decoder.parse(encodings, value_memory, h0, batch, beam_size=gtl_size,
                gtl_training=True, ts_order=ts_order, uts_order=uts_order, n_best=n_best, cumulate_method='sum')
            return {'ast_loss': ast_loss, 'vr_loss': vr_loss, 'gp_loss': gp_loss, 'hyps': hyps}
        else:
            ast_loss = self.decoder.score(encodings, value_memory, h0, batch)
            return {'ast_loss': ast_loss, 'vr_loss': vr_loss, 'gp_loss': gp_loss, 'hyps': []}


    def parse(self, batch, beam_size=5, ts_order='controller'):
        """ This function is used for decoding, which returns a batch of hypothesis
        """
        encodings, value_memory, (value_candidates, schema_gate) = self.encoder(batch)
        h0 = self.encoder2decoder(encodings, mask=batch.mask)
        hyps = self.decoder.parse(encodings, value_memory, h0, batch, beam_size=beam_size, gtl_training=False, ts_order=ts_order)
        return hyps, value_candidates, schema_gate


    def pad_embedding_grad_zero(self, index=None):
        """ For glove.42B.300d word vectors, gradients for <pad> symbol is always 0;
        Most words (starting from index) in the word vocab are also fixed except most frequent words
        """
        self.encoder.input_layer.pad_embedding_grad_zero(index)
