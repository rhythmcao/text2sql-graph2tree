#coding=utf8
import torch.nn as nn
from model.encoder.graph_input import *
from model.encoder.irnet import IRNet
from model.encoder.rgatsql import RGATSQL
from model.encoder.lgesql import LGESQL
from model.encoder.graph_output import *
from model.model_utils import Registrable

@Registrable.register('encoder')
class GraphEncoderSpider(nn.Module):

    def __init__(self, args):
        super(GraphEncoderSpider, self).__init__()
        lazy_load = args.lazy_load if hasattr(args, 'lazy_load') else False
        self.input_layer = GraphInputLayer(args.embed_size, args.gnn_hidden_size, args.word_vocab, dropout=args.dropout, schema_aggregation=args.schema_aggregation) \
            if args.plm is None else GraphInputLayerPLM(args.plm, args.gnn_hidden_size, dropout=args.dropout,
                subword_aggregation=args.subword_aggregation, schema_aggregation=args.schema_aggregation, lazy_load=lazy_load)
        self.hidden_layer = Registrable.by_name(args.encode_method)(args)
        self.output_layer = GraphOutputLayer(args)

    def forward(self, batch, sample_size=0):
        outputs = self.input_layer(batch)
        outputs = self.hidden_layer(outputs, batch)
        return self.output_layer(outputs, batch, sample_size)
