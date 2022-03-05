#coding=utf8
from asdl.asdl import ASDLGrammar
from asdl.transition_system import TransitionSystem
from utils.constants import DATASETS
from preprocess.dusql.value_utils import ValueExtractor

class OutputProcessor():

    def __init__(self, table_path=None, db_dir=None, **kargs) -> None:
        super(OutputProcessor, self).__init__()
        grammar = ASDLGrammar.from_filepath(DATASETS['dusql']['grammar'])
        self.trans = TransitionSystem.get_class_by_dataset('dusql')(grammar, table_path, db_dir)
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
        # number of schema items + 1, due to TIME_NOW
        q_num, s_num = len(entry['cased_question_toks']), len(db['table_names']) + len(db['column_names']) + 1
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
        sql = entry['sql']
        used_schema = {'table': set(), 'column': set()}
        used_schema = self.extract_subgraph_from_sql(sql, used_schema)
        entry['used_tables'] = sorted(list(used_schema['table']))
        # add 1 due to prepended special column TIME_NOW
        entry['used_columns'] = [i + 1 for i in sorted(list(used_schema['column']))]

        if verbose:
            print('Used tables:', entry['used_tables'])
            print('Used columns:', entry['used_columns'], '\n')
        return entry

    def extract_subgraph_from_sql(self, sql: dict, used_schema: dict):
        select_items = sql['select']
        # select clause
        for _, val_unit in select_items:
            if val_unit[0] == 0:
                col_unit = val_unit[1]
                used_schema['column'].add(int(col_unit[1]))
            else:
                col_unit1, col_unit2 = val_unit[1:]
                col_id = -1 if type(col_unit1[1]) == str else int(col_unit1[1])
                used_schema['column'].add(col_id)
                col_id = -1 if type(col_unit2[1]) == str else int(col_unit2[1])
                used_schema['column'].add(col_id)
        # from clause conds
        table_units = sql['from']['table_units']
        for _, t in table_units:
            if type(t) == dict:
                used_schema = self.extract_subgraph_from_sql(t, used_schema)
            else: used_schema['table'].add(t)
        # from, where and having conds
        used_schema = self.extract_subgraph_from_conds(sql['from']['conds'], used_schema, 'from')
        used_schema = self.extract_subgraph_from_conds(sql['where'], used_schema, 'where')
        used_schema = self.extract_subgraph_from_conds(sql['having'], used_schema, 'having')
        # groupBy and orderBy clause
        groupBy = sql['groupBy']
        for col_unit in groupBy:
            used_schema['column'].add(int(col_unit[1]))
        orderBy = sql['orderBy']
        if len(orderBy) > 0:
            orderBy = orderBy[1]
            for _, val_unit in orderBy:
                if val_unit[0] == 0:
                    col_unit = val_unit[1]
                    used_schema['column'].add(int(col_unit[1]))
                else:
                    col_unit1, col_unit2 = val_unit[1:]
                    col_unit1, col_unit2 = val_unit[1:]
                    col_id = -1 if type(col_unit1[1]) == str else int(col_unit1[1])
                    used_schema['column'].add(col_id)
                    col_id = -1 if type(col_unit2[1]) == str else int(col_unit2[1])
                    used_schema['column'].add(col_id)
        # union, intersect and except clause
        if sql['intersect']:
            used_schema = self.extract_subgraph_from_sql(sql['intersect'], used_schema)
        if sql['union']:
            used_schema = self.extract_subgraph_from_sql(sql['union'], used_schema)
        if sql['except']:
            used_schema = self.extract_subgraph_from_sql(sql['except'], used_schema)
        return used_schema

    def extract_subgraph_from_conds(self, conds: list, used_schema: dict, clause: str = 'where'):
        if len(conds) == 0:
            return used_schema
        for cond in conds:
            if cond in ['and', 'or']:
                continue
            val_unit, val1, _ = cond[2:] # no between
            if val_unit[0] == 0:
                col_unit = val_unit[1]
                used_schema['column'].add(int(col_unit[1]))
            else:
                col_unit1, col_unit2 = val_unit[1:]
                col_id = -1 if type(col_unit1[1]) == str else int(col_unit1[1])
                used_schema['column'].add(col_id)
                col_id = -1 if type(col_unit2[1]) == str else int(col_unit2[1])
                used_schema['column'].add(col_id)
            if type(val1) == dict:
                used_schema = self.extract_subgraph_from_sql(val1, used_schema)
            elif clause == 'from' and type(val1) == int:
                used_schema['column'].add(int(val1))
        return used_schema
