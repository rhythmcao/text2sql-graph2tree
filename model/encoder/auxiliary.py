#coding=utf8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import dgl.function as fn
from model.model_utils import PoolingFunction, lens2mask
from model.encoder.functions import scaled_exp, div_by_z, src_dot_dst
from model.encoder.beam import Beam
from preprocess.process_utils import ValueCandidate

class ScoreFunction(nn.Module):

    def __init__(self, hidden_size, output_size, mlp=1, method='biaffine'):
        super(ScoreFunction, self).__init__()
        assert method in ['bilinear', 'affine', 'biaffine']
        self.mlp = int(mlp)
        self.hidden_size = hidden_size // self.mlp
        self.output_size = output_size
        if self.mlp > 1: # use mlp to perform dim reduction
            self.mlp_node = nn.Sequential(nn.Linear(hidden_size, self.hidden_size), nn.Tanh())
            self.mlp_context = nn.Sequential(nn.Linear(hidden_size, self.hidden_size), nn.Tanh())
        self.method = method
        if self.method == 'bilinear':
            self.W = nn.Bilinear(self.hidden_size, self.hidden_size, self.output_size)
        elif self.method == 'affine':
            self.affine = nn.Linear(self.hidden_size * 2, self.output_size)
        elif self.method == 'biaffine': # bilinear + affine
            self.W = nn.Bilinear(self.hidden_size, self.hidden_size, self.output_size, bias=False)
            self.affine = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, node, context):
        """
        @args:
            node(torch.FloatTensor): num_nodes x hidden_size
            context(torch.FloatTensor): num_nodes x hidden_size
        @return:
            scores(torch.FloatTensor): num_nodes x output_size
        """
        if self.mlp > 1:
            node, context = self.mlp_node(node), self.mlp_context(context)
        if self.method == 'bilinear':
            scores = self.W(node, context)
        elif self.method == 'affine':
            scores = self.affine(torch.cat([context, node], dim=-1))
        elif self.method == 'biaffine':
            scores = self.W(node, context) + self.affine(torch.cat([node, context], dim=-1))
        else:
            raise ValueError('[Error]: Unrecognized score function method %s!' % (self.method))
        return scores

class LabelSmoothCELoss(nn.Module):
    def __init__(self):
        super(LabelSmoothCELoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = - F.log_softmax(inputs, dim=-1)
        return (targets * inputs).sum()

class AuxiliaryModule(nn.Module):

    def __init__(self, hidden_size, num_heads, dropout, score_function, value_aggregation):
        super(AuxiliaryModule, self).__init__()
        self.hidden_size = hidden_size
        self.cross_attention = DGLCrossAttention(self.hidden_size, self.hidden_size, num_heads=num_heads, feat_drop=dropout)
        self.question_score_function = ScoreFunction(self.hidden_size, 3, mlp=2, method=score_function)
        self.value_recognition_loss = LabelSmoothCELoss() #nn.CrossEntropyLoss(reduction='sum')
        self.value_aggregation = PoolingFunction(self.hidden_size, self.hidden_size, method=value_aggregation)
        self.schema_score_function = ScoreFunction(self.hidden_size, 1, mlp=2, method=score_function)
        self.graph_pruning_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, inputs, batch, sample_size=5):
        """
        @args:
            inputs: torch.FloatTensor, node_num(question+schema nodes) x hidden_size
        @return:
            value_memory: bsize x max_value_num x hidden_size
            if train:
                vr_loss: CrossEntropyLoss of Sequence Labeling BIO for Value Recognition
                gp_loss: BCELoss for each schema item
            else:
                value_candidates: for unparsing SQL AST
                schema_gates: True/False, bsize x (max_table_len + max_column_len)
        """
        g = batch.graph
        context = self.cross_attention(inputs, batch.graph.mg)
        
        if batch.predict_value:
            question, question_context = inputs[g.question_mask], context[g.question_mask]
            vr_score = self.question_score_function(question, question_context)
        schema, schema_context = inputs[g.schema_mask], context[g.schema_mask]
        gp_score = self.schema_score_function(schema, schema_context).squeeze(-1)
        
        if self.training:
            if batch.predict_value:
                vr_loss = self.value_recognition_loss(vr_score, g.question_prob)
                value_memory, _ = self.parse_value_memory(vr_score, question, batch, sample_size)
            else:
                vr_loss = inputs.new_zeros(1)[0]
                value_memory = inputs.new_zeros((len(batch), 0, self.hidden_size))
            gp_loss = self.graph_pruning_loss(gp_score, g.schema_prob)
            return value_memory, (vr_loss, gp_loss)
        else:
            if batch.predict_value:
                value_memory, value_candidates = self.parse_value_memory(vr_score, question, batch, sample_size)
            else: value_memory, value_candidates = inputs.new_zeros((len(batch), 0, self.hidden_size)), [[] for _ in range(len(batch))]
            schema_gates = torch.sigmoid(gp_score) >= 0.5
            return value_memory, (value_candidates, schema_gates)

    def parse_value_memory(self, scores, inputs, batch, sample_size=5):
        """
        @args:
            scores: torch.FloatTensor, num_question_nodes x 3 (BIO labels)
            inputs: torch.FloatTensor, num_question_nodes x hidden_size
            batch: require field value_lens (num_of_words of each value), value_nums (num_of_values of each sample)
        @return:
        """
        value_lens, value_nums, select_index, value_candidates = self.decode_value(scores, batch, sample_size=sample_size)
        total_value_num = value_lens.numel()
        if total_value_num > 0: # not empty in the current batch
            max_value_len = value_lens.max().item()
            # select_inputs = inputs.masked_select(select_mask.unsqueeze(-1))
            select_inputs = torch.index_select(inputs, 0, select_index)
            outputs = inputs.new_zeros((total_value_num, max_value_len, self.hidden_size))
            value_mask = lens2mask(value_lens)
            outputs = outputs.masked_scatter_(value_mask.unsqueeze(-1), select_inputs)
            values = self.value_aggregation(outputs.view(total_value_num, max_value_len, -1), value_mask)
            value_memory = inputs.new_zeros((len(batch), value_nums.max().item(), self.hidden_size))
            value_memory = value_memory.masked_scatter_(lens2mask(value_nums).unsqueeze(-1), values)
        else: # empty value in the current batch
            value_memory = inputs.new_zeros((len(batch), 0, self.hidden_size))
        return value_memory, value_candidates

    def decode_value(self, scores, batch, sample_size=5):
        """
        @args:
            labels: num_question_nodes x BIO label size(3)
            batch: use fields ``question_lens", ``max_question_len" and ``examples"
            sample_size: sample noise for value during training
        @return:
            value_lens, value_nums, value_candidates
        """
        # use beam search to sample noise
        if self.training:
            if sample_size <= 0:
                return batch.graph.value_lens, batch.graph.value_nums, batch.graph.select_index, None
            # sampling according to model predictions
            probs = torch.softmax(scores, dim=1)
            question_lens = batch.question_lens.tolist()
            batch_probs = probs.split(question_lens, dim=0)
            batch_labels, sample_size = [], torch.Size([sample_size])
            for idx in range(len(question_lens)): # process each sample
                length, probs = question_lens[idx], batch_probs[idx]
                labels = []
                for t in range(length):
                    cur_label = Categorical(probs[t]).sample(sample_shape=sample_size)
                    labels.append(cur_label)
                labels = torch.stack(labels, dim=0).transpose(0, 1).tolist()
                batch_labels.append(labels)
            # sampling with beam_search
            # if sample_size == 1:
                # labels = scores.max(dim=-1)[1]
                # batch_labels = labels.split(batch.question_lens.tolist(), dim=0)
                # batch_labels = [[labels.tolist()] for labels in batch_labels] # add a wrapper to be compatible with sample size > 1
            # else:
                # logprobs = F.log_softmax(scores, dim=1)
                # question_lens = batch.question_lens.tolist()
                # beams = [Beam(sample_size, question_lens[i], scores.device) for i in range(len(question_lens))]
                # batch_logprobs = logprobs.split(question_lens, dim=0)
                # for t in range(batch.max_question_len):
                    # for idx, b in enumerate(beams):
                        # if not b.done:
                            # b.advance(batch_logprobs[idx][t])
                # batch_labels = [b.get_hyps() for b in beams]
        else: # greedy decode outperforms beam search (maybe due to smaller propagation error)
            labels = scores.max(dim=-1)[1]
            batch_labels = labels.split(batch.question_lens.tolist(), dim=0)
            batch_labels = [[labels.tolist()] for labels in batch_labels]
        value_lens, value_nums, select_index, value_candidates = self.decode_value_labels(batch_labels, batch)
        batch.graph.value_nums = value_nums
        return value_lens, value_nums, select_index, value_candidates

    def decode_value_labels(self, batch_labels, batch):
        """
        @args:
            batch_labels: for example, 3 instances with sample_size 2
            [ [[0,0,0],[0,1,1]] , [[0,0,1,2],[0,1,2,2]] , [[0,0,0,1,2,2],[0,1,2,0,1,2]] ]
            batch: use field ``examples" to create ValueCandidates if not train
        @return:
            value_lens(torch.LongTensor): the length (number of tokens) of each value considering the entire minibatch
            value_nums(torch.LongTensor): the number of values for each sample in the minibatch
            select_index(torch.LongTensor): use to select the index of question nodes
            value_candidates(list): used for evaluation
        """
        value_lens, value_nums, select_index, value_candidates = [], [], [], []
        bias = 0
        for sample, labels in zip(batch.examples, batch_labels):
            for cur_labels in labels: # each hyp in the beam
                prev, start, index_pairs = 0, -1, set()
                # O -> 0 ; B -> 1 ; I -> 2
                for i, l in enumerate(cur_labels):
                    if l == 0:
                        if prev == 1 or prev == 2:
                            index_pairs.add((start, i))
                        prev = l
                    elif l == 1:
                        if prev == 1 or prev == 2:
                            index_pairs.add((start, i))
                        start, prev = i, l
                    else: # l == 2
                        if prev == 0:
                            start, prev = i, 1
                        else:
                            prev = l
                if prev == 1 or prev == 2:
                    index_pairs.add((start, len(cur_labels)))
            index_pairs = sorted(index_pairs)
            if self.training: # prepend gold candidates first
                gold_candidates = sample.ex['candidates']
                gold_index_pairs = [vc.matched_index for vc in gold_candidates]
                index_pairs = gold_index_pairs + [pair for pair in index_pairs if pair not in gold_index_pairs]
            candidates = []
            for pair in index_pairs:
                question_toks = sample.ex['uncased_question_toks']
                question_span = ' '.join(question_toks[pair[0]:pair[1]])
                cased_question_toks = sample.ex['cased_question_toks']
                cased_question_span = ' '.join(cased_question_toks[pair[0]:pair[1]])
                vc = ValueCandidate(pair, question_span, cased_question_span)
                candidates.append(vc)
                value_lens.append(pair[1] - pair[0])
                select_index.extend(list(range(bias + pair[0], bias + pair[1])))
            bias += len(labels[0]) # num of tokens in the current question
            value_nums.append(len(candidates))
            value_candidates.append(candidates)
        value_lens = torch.tensor(value_lens, dtype=torch.long).to(batch.device)
        value_nums = torch.tensor(value_nums, dtype=torch.long).to(batch.device)
        select_index = torch.tensor(select_index, dtype=torch.long).to(batch.device)
        return value_lens, value_nums, select_index, value_candidates

class DGLCrossAttention(nn.Module):
    """ Cross Multi-head Attention implemented with DGL lib,
    save memory usage but sacrifice running time
    """
    def __init__(self, hidden_size, output_size, num_heads=8, feat_drop=0.2):
        super(DGLCrossAttention, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.d_k = self.hidden_size // self.num_heads
        self.affine_q, self.affine_k, self.affine_v = nn.Linear(self.output_size, self.hidden_size),\
            nn.Linear(self.hidden_size, self.hidden_size, bias=False), nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.affine_o = nn.Linear(self.hidden_size, self.output_size)
        self.feat_dropout = nn.Dropout(p=feat_drop)

    def forward(self, inputs, graph):
        q, k, v = self.affine_q(self.feat_dropout(inputs)), self.affine_k(self.feat_dropout(inputs)), \
            self.affine_v(self.feat_dropout(inputs))
        with graph.local_scope():
            graph.ndata['q'] = q.view(-1, self.num_heads, self.d_k)
            graph.ndata['k'] = k.view(-1, self.num_heads, self.d_k)
            graph.ndata['v'] = v.view(-1, self.num_heads, self.d_k)
            context = self.propagate_attention(graph, self.affine_o)
        return context

    def propagate_attention(self, g, output_module):
        # Compute attention score
        g.apply_edges(src_dot_dst('k', 'q', 'score'))
        g.apply_edges(scaled_exp('score', math.sqrt(self.d_k)))
        # Update node state
        g.update_all(fn.src_mul_edge('v', 'score', 'v'), fn.sum('v', 'wv'))
        g.update_all(fn.copy_edge('score', 'score'), fn.sum('score', 'z'), div_by_z('wv', 'z', 'o'))
        out_x = g.ndata['o']
        return output_module(out_x.view(-1, self.hidden_size))
