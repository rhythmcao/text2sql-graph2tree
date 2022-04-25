#coding=utf8
from asdl.asdl import ASDLGrammar
from asdl.transition_system import TransitionSystem
from utils.constants import DATASETS
from preprocess.wikisql.value_utils import ValueExtractor
from preprocess.nl2sql.output_utils import OutputProcessor as BaseOutputProcessor

class OutputProcessor(BaseOutputProcessor):

    def __init__(self, table_path=None, db_dir=None, **kargs) -> None:
        grammar = ASDLGrammar.from_filepath(DATASETS['wikisql']['grammar'])
        self.trans = TransitionSystem.get_class_by_dataset('wikisql')(grammar, table_path, db_dir)
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

    def extract_subgraph(self, entry: dict, db: dict, verbose: bool = False):
        used_columns, sql = set(), entry['sql']
        sel, conds = [sql['sel']], [cond[0] for cond in sql['conds']]
        used_columns.update(sel + conds)
        entry['used_tables'] = [0] # only one table
        entry['used_columns'] = sorted(used_columns)

        if verbose:
            print('Used tables:', entry['used_tables'])
            print('Used columns:', entry['used_columns'], '\n')
        return entry