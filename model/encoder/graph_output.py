#coding=utf8
import torch.nn as nn
from model.encoder.auxiliary import AuxiliaryModule

class GraphOutputLayer(nn.Module):

    def __init__(self, args):
        super(GraphOutputLayer, self).__init__()
        self.hidden_size = args.gnn_hidden_size
        self.auxiliary_module = AuxiliaryModule(self.hidden_size, args.num_heads, args.dropout, args.score_function, args.value_aggregation)

    def forward(self, inputs, batch, sample_size=5):
        outputs = inputs.new_zeros(len(batch), batch.mask.size(1), self.hidden_size)
        outputs = outputs.masked_scatter_(batch.mask.unsqueeze(-1), inputs)
        # aux_outputs is (vr_loss, gp_loss) during training and (value_candidates, schema_gates) during evaluation
        value_memory, aux_outputs = self.auxiliary_module(inputs, batch, sample_size)
        return outputs, value_memory, aux_outputs
