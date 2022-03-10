#coding=utf8
from asdl.asdl import ASDLGrammar
from asdl.transition_system import TransitionSystem
from preprocess.cspider.value_utils import ValueExtractor
from preprocess.spider.output_utils import OutputProcessor as BaseOutputProcessor
from utils.constants import DATASETS

class OutputProcessor(BaseOutputProcessor):

    def __init__(self, table_path=None, db_dir=None, **kargs) -> None:
        super(OutputProcessor, self).__init__()
        grammar = ASDLGrammar.from_filepath(DATASETS['cspider']['grammar'])
        self.trans = TransitionSystem.get_class_by_dataset('cspider')(grammar, table_path, db_dir)
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