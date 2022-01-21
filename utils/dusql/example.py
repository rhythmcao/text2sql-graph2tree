#coding=utf8
import os, pickle
import random
import numpy as np
from asdl.asdl import ASDLGrammar
from asdl.transition_system import TransitionSystem
from asdl.action_info import get_action_infos
from transformers import AutoTokenizer
from utils.constants import SCHEMA_TYPES_MAPPING, RELATIONS, DEBUG, GRAMMAR_FILEPATH
from utils.graphs import GraphFactory
from utils.vocab import Vocab
from utils.dataset import SQLDataset
from utils.evaluator import Evaluator
from itertools import chain


class Example():

    @classmethod
    def configuration(cls, plm=None, method='lgesql', table_path='data/dusql/tables.json', tables='data/dusql/tables.bin', db_dir='data/dusql/db_content.json', order_path=None):
        cls.dataset = 'dusql'
        cls.plm, cls.method = plm, method
        cls.tables = pickle.load(open(table_path, 'rb')) if type(table_path) == str else table_path
        cls.grammar = ASDLGrammar.from_filepath(GRAMMAR_FILEPATH(cls.dataset))
        cls.trans = TransitionSystem.get_class_by_dataset(cls.dataset)(cls.grammar, tables, db_dir)
        cls.evaluator = Evaluator.get_class_by_dataset(cls.dataset)(cls.trans, table_path, db_dir)
        
        cls.order_path = None
        if order_path is not None:
            order_path = os.path.join(order_path, 'order.bin')
            if os.path.exists(order_path):
                cls.order_path = order_path
                order = pickle.load(open(order_path, 'rb'))
                best_order = cls.grammar.order_controller.compute_best_order(order)
                cls.grammar.order_controller.set_order(best_order)
                print('Load pre-defined order from:', cls.order_path)
        cls.tokenizer = AutoTokenizer.from_pretrained(os.path.join('./pretrained_models', plm))
        cls.word_vocab = cls.tokenizer.get_vocab()
        cls.relation_vocab = Vocab(padding=False, unk=False, boundary=False, iterable=RELATIONS, default=None)
        cls.graph_factory = GraphFactory(cls.method, cls.relation_vocab)


    @classmethod
    def load_dataset(cls, choice, order_seed=999, shuffle=False):
        assert choice in ['train', 'dev']
        fp = os.path.join('data', 'dusql', choice + '.bin') if not DEBUG else os.path.join('data', 'dusql', 'train.bin')
        datasets = pickle.load(open(fp, 'rb'))
        examples = []
        # control the order of fixed training examples
        state = np.random.get_state()
        np.random.seed(order_seed)
        tp_shuffle = shuffle and choice == 'train' and cls.order_path is None
        if tp_shuffle:
            cls.grammar.order_controller.shuffle_order()
        for ex in datasets:
            examples.append(cls(ex, cls.tables[ex['db_id']], shuffle, ex['question_id']))
            if DEBUG and len(examples) >= 100:
                break
        np.random.set_state(state)

        question_lens = [len(ex.input_id) for ex in examples]
        print('Max/Min/Avg input length in %s dataset is: %d/%d/%.2f' % (choice, max(question_lens), min(question_lens), np.mean(question_lens)))
        action_lens = [len(ex.tgt_action) for ex in examples]
        print('Max/Min/Avg action length in %s dataset is: %d/%d/%.2f' % (choice, max(action_lens), min(action_lens), np.mean(action_lens)))
        return SQLDataset(examples)


    def __init__(self, ex: dict, db: dict, shuffle: bool = False, id: str = 'qid', fixed=False):
        super(Example, self).__init__()
        self.id = id
        self.ex = ex
        self.db = db

        """ Mapping word to corresponding index """
        t = Example.tokenizer
        self.question = [q for q in ex['uncased_question_toks']]
        self.question_id = [t.cls_token_id] # map token to id
        self.question_mask_plm = [] # remove SEP token in our case
        self.question_subword_len = [] # subword len for each word, exclude SEP token
        for w in self.question:
            toks = t.convert_tokens_to_ids(t.tokenize(w))
            self.question_id.extend(toks)
            self.question_subword_len.append(len(toks))
        self.question_mask_plm = [0] + [1] * (len(self.question_id) - 1) + [0]
        self.question_id.append(t.sep_token_id)

        self.table = [[SCHEMA_TYPES_MAPPING['table']] + toks for toks in db['table_toks']]
        self.table_id, self.table_mask_plm, self.table_subword_len = [], [], []
        self.table_word_len = []
        for s in self.table:
            l = 0
            for w in s:
                toks = t.convert_tokens_to_ids(t.tokenize(w))
                self.table_id.extend(toks)
                self.table_subword_len.append(len(toks))
                l += len(toks)
            self.table_word_len.append(l)
        self.table_mask_plm = [1] * len(self.table_id)

        self.column = [[SCHEMA_TYPES_MAPPING['time'], '当前时间'], [SCHEMA_TYPES_MAPPING['text'], '任意列']] + \
            [[SCHEMA_TYPES_MAPPING[db['column_types'][idx + 1]]] + toks for idx, toks in enumerate(db['column_toks'][1:])]
            # [[SCHEMA_TYPES_MAPPING[db['column_types'][idx + 1]]] + toks + ex['cells'][idx + 1] for idx, toks in enumerate(db['column_toks'][1:])]
        self.column_id, self.column_mask_plm, self.column_subword_len = [], [], []
        self.column_word_len = []
        for s in self.column:
            l = 0
            for w in s:
                toks = t.convert_tokens_to_ids(t.tokenize(w))
                self.column_id.extend(toks)
                self.column_subword_len.append(len(toks))
                l += len(toks)
            self.column_word_len.append(l)
        self.column_mask_plm = [1] * len(self.column_id) + [0]
        self.column_id.append(t.sep_token_id)

        self.input_id = self.question_id + self.table_id + self.column_id
        self.segment_id = [0] * len(self.question_id) + [1] * (len(self.table_id) + len(self.column_id))

        self.question_mask_plm = self.question_mask_plm + [0] * (len(self.table_id) + len(self.column_id))
        self.table_mask_plm = [0] * len(self.question_id) + self.table_mask_plm + [0] * len(self.column_id)
        self.column_mask_plm = [0] * (len(self.question_id) + len(self.table_id)) + self.column_mask_plm

        self.graph = Example.graph_factory.graph_construction(ex, db)

        # outputs
        self.query, self.ast, self.tgt_action = '', None, []
        self.used_tables, self.used_columns = [], []
        if 'values' in ex and 'used_tables' in ex:
            self.query = ' '.join(ex['query'].split('\t'))
            self.ast = Example.trans.surface_code_to_ast(ex['sql'], ex['values'])
            if fixed:
                self.tgt_action = get_action_infos(ex['actions'])
            else:
                self.tgt_action = get_action_infos(Example.trans.get_field_action_pairs(self.ast, untyped_random=shuffle))
            self.used_tables, self.used_columns = ex['used_tables'], ex['used_columns']


def get_position_ids(ex, shuffle=True):
    # cluster columns with their corresponding table and randomly shuffle tables and columns
    # [CLS] q1 q2 ... [SEP] TIME_NOW * t1 c1 c2 c3 t2 c4 c5 ... [SEP]
    db, table_word_len, column_word_len = ex.db, ex.table_word_len, ex.column_word_len
    table_num, column_num = len(db['table_names']), 1 + len(db['column_names'])
    question_position_id = list(range(len(ex.question_id)))
    start = len(question_position_id)
    table_position_id, column_position_id = [None] * table_num, [None] * column_num
    column_position_id[0] = list(range(start, start + column_word_len[0]))
    start += column_word_len[0] # special symbol TIME_NOW first
    column_position_id[1] = list(range(start, start + column_word_len[1]))
    start += column_word_len[1] # special symbol * next
    table_idxs = list(range(table_num))
    if shuffle:
        random.shuffle(table_idxs)
    for idx in table_idxs:
        col_idxs = db['table2columns'][idx]
        table_position_id[idx] = list(range(start, start + table_word_len[idx]))
        start += table_word_len[idx]
        if shuffle:
            random.shuffle(col_idxs)
        for col_id in col_idxs:
            col_id = col_id + 1 # plus 1 due to column TIME_NOW
            column_position_id[col_id] = list(range(start, start + column_word_len[col_id]))
            start += column_word_len[col_id]
    position_id = question_position_id + list(chain.from_iterable(table_position_id)) + \
        list(chain.from_iterable(column_position_id)) + [start]
    assert len(position_id) == len(ex.input_id)
    return position_id
