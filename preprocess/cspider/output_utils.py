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
        self.predict_value = DATASETS['cspider']['predict_value']
        self.trans = TransitionSystem.get_class_by_dataset('cspider')(grammar, table_path, db_dir)
        if self.predict_value:
            self.value_extractor = ValueExtractor()