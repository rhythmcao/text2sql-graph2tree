#coding=utf8
import dgl, torch


class GraphExample():

    pass


class BatchedGraph():

    pass


class GraphFactory():

    def __init__(self, encode_method='rgatsql', relation_vocab=None):
        super(GraphFactory, self).__init__()
        self.encode_method = eval('self.encode_' + encode_method)
        self.batch_method = eval('self.batch_' + encode_method)
        self.relation_vocab = relation_vocab


    def graph_construction(self, ex: dict, db: dict):
        return self.encode_method(ex, db)


    def encode_irnet(self, ex, db):
        graph = GraphExample()
        # only use the graph without edge features
        graph.local_g, graph.global_g = ex['graph'].local_g, ex['graph'].global_g
        # for value recognition and graph pruning
        graph.mg = ex['graph'].mg
        graph.question_mask = torch.tensor(ex['graph'].question_mask, dtype=torch.bool)
        graph.schema_mask = torch.tensor(ex['graph'].schema_mask, dtype=torch.bool)
        if hasattr(ex['graph'], 'question_label'): # value recognition training
            graph.question_label = torch.tensor(ex['graph'].question_label, dtype=torch.long)
            graph.value_len = torch.tensor(ex['graph'].value_len, dtype=torch.long)
            graph.value_num = graph.value_len.size(0)
        if hasattr(ex['graph'], 'schema_label'): # graph pruning training
            graph.schema_label = torch.tensor(ex['graph'].schema_label, dtype=torch.bool)
        return graph


    def encode_rgatsql(self, ex, db):
        graph = self.encode_irnet(ex, db)
        # add edge features
        local_edges = ex['graph'].local_edges
        rel_ids = list(map(lambda r: self.relation_vocab[r[2]], local_edges))
        graph.local_edges = torch.tensor(rel_ids, dtype=torch.long)
        global_edges = ex['graph'].global_edges
        rel_ids = list(map(lambda r: self.relation_vocab[r[2]], global_edges))
        graph.global_edges = torch.tensor(rel_ids, dtype=torch.long)
        # extract local relations (used in msde/mmc), global_edges = local_edges + nonlocal_edges
        local_enum, global_enum = len(local_edges), len(global_edges)
        graph.local_mask = torch.tensor([1] * local_enum + [0] * (global_enum - local_enum), dtype=torch.bool)
        return graph


    def encode_lgesql(self, ex, db):
        graph = self.encode_rgatsql(ex, db)
        # add line graph such that edge features can be updated layerwise
        graph.lg = ex['graph'].lg
        return graph


    def batch_graphs(self, ex_list, device, train=True, **kwargs):
        """ Batch graphs in example list """
        return self.batch_method(ex_list, device, train=train, **kwargs)


    def batch_irnet(self, ex_list, device, train=True, **kwargs):
        bg = BatchedGraph()
        graph_list = [ex.graph for ex in ex_list]
        bg.local_g = dgl.batch([ex.local_g for ex in graph_list]).to(device)
        bg.global_g = dgl.batch([ex.global_g for ex in graph_list]).to(device)

        # for value recognition and graph pruning
        bg.mg = dgl.batch([ex.mg for ex in graph_list]).to(device)
        bg.question_mask = torch.cat([ex.question_mask for ex in graph_list], dim=0).to(device)
        bg.schema_mask = torch.cat([ex.schema_mask for ex in graph_list], dim=0).to(device)

        if train:
            if hasattr(graph_list[0], 'question_label'): # labels for value recognition
                bg.value_lens = torch.cat([ex.value_len for ex in graph_list], dim=0).to(device)
                bg.value_nums = torch.tensor([ex.value_num for ex in graph_list], dtype=torch.long).to(device)
                candidates = [[vc.matched_index for vc in ex.ex['candidates']] for ex in ex_list]
                question_lens = [len(ex.question) for ex in ex_list]
                select_index, bias = [], 0
                for idx, matched_index in enumerate(candidates):
                    for pair in matched_index:
                        select_index.extend(list(range(bias + pair[0], bias + pair[1])))
                    bias += question_lens[idx]
                bg.select_index = torch.tensor(select_index, dtype=torch.long).to(device)
                question_label = torch.cat([ex.question_label for ex in graph_list], dim=0)
                smoothing = kwargs.pop('smoothing', 0.0)
                question_prob = torch.full((question_label.size(0), 3), smoothing / 2)
                question_prob = question_prob.scatter_(1, question_label.unsqueeze(1), 1 - smoothing)
                bg.question_prob = question_prob.to(device)
                bg.question_label = question_label.to(device)
            # labels for graph pruning
            schema_label = torch.cat([ex.schema_label for ex in graph_list], dim=0)
            schema_prob = schema_label.float()
            schema_prob = schema_prob.masked_fill_(~ schema_label, 2 * smoothing) - smoothing
            bg.schema_prob = schema_prob.to(device)
        return bg


    def batch_rgatsql(self, ex_list, device, train=True, **kwargs):
        bg = self.batch_irnet(ex_list, device, train=train, **kwargs)
        bg.local_mask = torch.cat([ex.graph.local_mask for ex in ex_list], dim=0).to(device)
        bg.global_edges = torch.cat([ex.graph.global_edges for ex in ex_list], dim=0).to(device)
        return bg


    def batch_lgesql(self, ex_list, device, train=True, **kwargs):
        bg = self.batch_rgatsql(ex_list, device, train=train, **kwargs)
        src_ids, dst_ids = bg.local_g.edges(order='eid')
        bg.src_ids, bg.dst_ids = src_ids.long(), dst_ids.long()
        bg.lg = dgl.batch([ex.graph.lg for ex in ex_list]).to(device)
        return bg