#coding=utf8
from asdl.asdl import ASDLGrammar
from asdl.transition_system import TransitionSystem
from utils.constants import DATASETS
from preprocess.nl2sql.value_utils import ValueExtractor

class OutputProcessor():

    def __init__(self, table_path=None, db_dir=None, **kargs) -> None:
        super(OutputProcessor, self).__init__()
        grammar = ASDLGrammar.from_filepath(DATASETS['nl2sql']['grammar'])
        self.trans = TransitionSystem.get_class_by_dataset('nl2sql')(grammar, table_path, db_dir)
        self.value_extractor = ValueExtractor()

    def pipeline(self, entry: dict, db: dict, verbose: bool = False):
        # extract schema sub-graph for graph pruning, entry key: 'used_tables' and 'used_columns'
        entry = self.extract_subgraph(entry, db, verbose=verbose)
        # extract bio sequence and all SQLValue's first, entry key: 'values', 'candidates'
        entry = self.value_extractor.extract_values(entry, db, verbose=verbose)
        # add auxiliary labels for value recognition and graph pruning
        entry = self.auxiliary_labels(entry, db)
        # generate golden ast
        ast = self.trans.surface_code_to_ast(entry['sql'], entry['values'])
        entry['ast'] = ast
        return entry

    def auxiliary_labels(self, entry: dict, db: dict):
        graph = entry['graph']
        q_num, s_num = len(entry['cased_question_toks']), len(db['table_names']) + len(db['column_names'])
        # by default: O -> 0 ; B -> 1 ; I -> 2
        index_pairs = [val.matched_index for val in entry['candidates']]
        question_label = [0] * q_num
        value_len = [] # record the token length of each value, for pooling and re-scatter
        for start, end in index_pairs:
            question_label[start:end] = [1] + [2] * (end - start - 1)
            value_len.append(end - start)
        graph.question_label = question_label
        graph.value_len = value_len

        t_num = len(db['table_names'])
        def check_node(i):
            if i < t_num and i in entry['used_tables']:
                return 1.0
            elif i >= t_num and i - t_num in entry['used_columns']:
                return 1.0
            else: return 0.0
        graph.schema_label = list(map(check_node, range(s_num)))
        return entry

    def extract_subgraph(self, entry: dict, db: dict, verbose: bool = False):
        used_columns, sql = set(), entry['sql']
        sel, conds = sql['sel'], [cond[0] for cond in sql['conds']]
        used_columns.update(sel + conds)
        entry['used_tables'] = [0] # only one table
        entry['used_columns'] = sorted(used_columns)

        if verbose:
            print('Used tables:', entry['used_tables'])
            print('Used columns:', entry['used_columns'], '\n')
        return entry