#coding=utf8
import math, dgl, torch
import numpy as np
from utils.graphs import GraphExample
from utils.constants import MAX_RELATIVE_DIST


# mapping special column * as an ordinary column
nonlocal_relations = [
    'question-question-generic', 'table-table-generic', 'column-column-generic', 'table-column-generic', 'column-table-generic',
    'table-table-fk', 'table-table-fkr', 'table-table-fkb', 'column-column-sametable',
    'question-question-identity', 'table-table-identity', 'column-column-identity'] + [
    'question-question-dist' + str(i) for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1, 1) if i not in [-1, 0, 1]
]

class GraphProcessor():

    def __init__(self, encode_method='rgatsql') -> None:
        super(GraphProcessor, self).__init__()
        self.encode_method = encode_method
        self.process_method = eval('self.process_' + encode_method)


    def process_irnet(self, ex: dict, db: dict, relation: list):
        graph = GraphExample()
        num_nodes = int(math.sqrt(len(relation)))
        local_edges = [(idx // num_nodes, idx % num_nodes, r)
            for idx, r in enumerate(relation) if r not in nonlocal_relations]
        nonlocal_edges = [(idx // num_nodes, idx % num_nodes,  r)
            for idx, r in enumerate(relation) if r in nonlocal_relations]
        global_edges = local_edges + nonlocal_edges
        src_ids, dst_ids = list(map(lambda r: r[0], global_edges)), list(map(lambda r: r[1], global_edges))
        graph.global_g = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes, idtype=torch.int32)
        src_ids, dst_ids = list(map(lambda r: r[0], local_edges)), list(map(lambda r: r[1], local_edges))
        graph.local_g = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes, idtype=torch.int32)
        ex['graph'] = graph
        return ex


    def process_rgatsql(self, ex: dict, db: dict, relation: list):
        graph = GraphExample()
        num_nodes = int(math.sqrt(len(relation)))
        local_edges = [(idx // num_nodes, idx % num_nodes, r)
            for idx, r in enumerate(relation) if r not in nonlocal_relations]
        nonlocal_edges = [(idx // num_nodes, idx % num_nodes,  r)
            for idx, r in enumerate(relation) if r in nonlocal_relations]
        global_edges = local_edges + nonlocal_edges
        src_ids, dst_ids = list(map(lambda r: r[0], global_edges)), list(map(lambda r: r[1], global_edges))
        graph.global_g = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes, idtype=torch.int32)
        graph.global_edges = global_edges
        src_ids, dst_ids = list(map(lambda r: r[0], local_edges)), list(map(lambda r: r[1], local_edges))
        graph.local_g = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes, idtype=torch.int32)
        graph.local_edges = local_edges
        ex['graph'] = graph
        return ex


    def process_lgesql(self, ex: dict, db: dict, relation: list):
        graph = self.process_rgatsql(ex, db, relation)['graph']
        lg = graph.local_g.line_graph(backtracking=False)
        # prevent information propagate through matching edges
        match_ids = [idx for idx, r in enumerate(graph.global_edges) if 'match' in r[2]]
        src, dst, eids = lg.edges(form='all', order='eid')
        eids = [e for u, v, e in zip(src.tolist(), dst.tolist(), eids.tolist()) if not (u in match_ids and v in match_ids)]
        graph.lg = lg.edge_subgraph(eids, preserve_nodes=True).remove_self_loop().add_self_loop()
        ex['graph'] = graph
        return ex


    def process_graph_utils(self, ex: dict, db: dict):
        """ Example should be preprocessed by self.pipeline
        """
        q = np.array(ex['relations'], dtype='<U100')
        s = np.array(db['relations'], dtype='<U100')
        q_s = np.array(ex['schema_linking'][0], dtype='<U100')
        s_q = np.array(ex['schema_linking'][1], dtype='<U100')
        relation = np.concatenate([
            np.concatenate([q, q_s], axis=1),
            np.concatenate([s_q, s], axis=1)
        ], axis=0)
        relation = relation.flatten().tolist()

        ex = self.process_method(ex, db, relation)

        graph = ex['graph']
        q_num, s_num = q_s.shape[0], q_s.shape[1]
        edges = [(i, j) for i in range(q_num) for j in range(q_num, q_num + s_num)] + \
            [(j, i) for i in range(q_num) for j in range(q_num, q_num + s_num)]
        src_ids, dst_ids = list(map(lambda r: r[0], edges)), list(map(lambda r: r[1], edges))
        # mg -> matching graph, for value recognition and graph pruning
        graph.mg = dgl.graph((src_ids, dst_ids), num_nodes=q_num + s_num, idtype=torch.int32)
        graph.question_mask = [1] * q_num + [0] * s_num
        graph.schema_mask = [0] * q_num + [1] * s_num
        return ex