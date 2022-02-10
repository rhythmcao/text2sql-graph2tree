#coding=utf8
import math
import torch
import torch.nn as nn
import dgl.function as fn
from model.encoder.functions import *
from model.model_utils import Registrable, FFN


@Registrable.register('irnet')
class IRNet(nn.Module):

    def __init__(self, args):
        super(IRNet, self).__init__()
        self.num_layers = args.gnn_num_layers
        self.graph_view = 'multiview' if args.local_and_nonlocal == 'mmc' else args.local_and_nonlocal
        gnn_layer = MultiViewIRNetLayer if self.graph_view == 'multiview' else IRNetLayer
        self.gnn_layers = nn.ModuleList([gnn_layer(args.gnn_hidden_size, num_heads=args.num_heads, feat_drop=args.dropout)
            for _ in range(self.num_layers)])

    def forward(self, x, batch):
        if self.graph_view == 'multiview':
            # multi-view multi-head concatenation
            local_g, global_g = batch.graph.local_g, batch.graph.global_g
            for i in range(self.num_layers):
                x = self.gnn_layers[i](x, local_g, global_g)
        else:
            graph = batch.graph.local_g if self.graph_view == 'local' else batch.graph.global_g
            for i in range(self.num_layers):
                x = self.gnn_layers[i](x, graph)
        return x


class IRNetLayer(nn.Module):

    def __init__(self, ndim, num_heads=8, feat_drop=0.2):
        super(IRNetLayer, self).__init__()
        self.ndim = ndim
        self.num_heads = num_heads
        self.d_k = ndim // self.num_heads
        self.affine_q, self.affine_k, self.affine_v = nn.Linear(self.ndim, self.ndim),\
            nn.Linear(self.ndim, self.ndim, bias=False), nn.Linear(self.ndim, self.ndim, bias=False)
        self.affine_o = nn.Linear(self.ndim, self.ndim)
        self.layernorm = nn.LayerNorm(self.ndim)
        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.ffn = FFN(self.ndim)

    def forward(self, x, g):
        """ @Params:
                x: node feats, num_nodes x ndim
                g: dgl.graph
        """
        # pre-mapping q/k/v affine
        q, k, v = self.affine_q(self.feat_dropout(x)), self.affine_k(self.feat_dropout(x)), self.affine_v(self.feat_dropout(x))
        with g.local_scope():
            g.ndata['q'] = q.view(-1, self.num_heads, self.d_k)
            g.ndata['k'] = k.view(-1, self.num_heads, self.d_k)
            g.ndata['v'] = v.view(-1, self.num_heads, self.d_k)
            out_x = self.propagate_attention(g)

        out_x = self.layernorm(x + self.affine_o(out_x.view(-1, self.num_heads * self.d_k)))
        out_x = self.ffn(out_x)
        return out_x

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('k', 'q', 'score'))
        g.apply_edges(scaled_exp('score', math.sqrt(self.d_k)))
        # Update node state
        g.update_all(fn.u_mul_e('v', 'score', 'v'), fn.sum('v', 'wv'))
        g.update_all(fn.copy_edge('score', 'score'), fn.sum('score', 'z'), div_by_z('wv', 'z', 'o'))
        out_x = g.ndata['o']
        return out_x


class MultiViewIRNetLayer(IRNetLayer):

    def forward(self, x, local_g, global_g):
        """ @Params:
                x: node feats, num_nodes x ndim
                local_g: dgl.graph, a local graph for node update
                global_g: dgl.graph, a complete graph for node update
        """
        # pre-mapping q/k/v affine
        q, k, v = self.affine_q(self.feat_dropout(x)), self.affine_k(self.feat_dropout(x)), self.affine_v(self.feat_dropout(x))
        q, k, v = q.view(-1, self.num_heads, self.d_k), k.view(-1, self.num_heads, self.d_k), v.view(-1, self.num_heads, self.d_k)
        with local_g.local_scope():
            local_g.ndata['q'], local_g.ndata['k'] = q[:, :self.num_heads // 2], k[:, :self.num_heads // 2]
            local_g.ndata['v'] = v[:, :self.num_heads // 2]
            out_x1 = self.propagate_attention(local_g)
        with global_g.local_scope():
            global_g.ndata['q'], global_g.ndata['k'] = q[:, self.num_heads // 2:], k[:, self.num_heads // 2:]
            global_g.ndata['v'] = v[:, self.num_heads // 2:]
            out_x2 = self.propagate_attention(global_g)
        out_x = torch.cat([out_x1, out_x2], dim=1)
        out_x = self.layernorm(x + self.affine_o(out_x.view(-1, self.num_heads * self.d_k)))
        out_x = self.ffn(out_x)
        return out_x