#coding=utf8
import json
import os, pickle, random
import numpy as np
from asdl.asdl import ASDLGrammar
from asdl.action_info import get_action_infos
from asdl.transition_system import TransitionSystem
from eval.evaluator import Evaluator
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils.vocab import Vocab
from utils.graphs import GraphFactory
from utils.word2vec import Word2vecUtils
from utils.constants import DATASETS, UNK, DEBUG, TEST
from itertools import chain


class SQLDataset(Dataset):

    def __init__(self, examples) -> None:
        super(SQLDataset, self).__init__()
        self.examples = examples


    def __len__(self):
        return len(self.examples)


    def __getitem__(self, index: int):
        return self.examples[index]


class Example():

    collate_fn = lambda x: list(x)

    @classmethod
    def configuration(cls, dataset, plm=None, encode_method='lgesql',
            table_path=None, tables=None, db_dir=None, ts_order_path=None):
        cls.dataset = dataset
        cls.plm, cls.encode_method = plm, encode_method
        cls.predict_value = DATASETS[cls.dataset]['predict_value']
        cls.db_dir = db_dir if db_dir is not None else DATASETS[cls.dataset]['database']
        cls.data_dir = DATASETS[cls.dataset]['data']
        table_path = table_path if table_path is not None else os.path.join(cls.data_dir, 'tables.json')
        cls.tables = tables if tables is not None else pickle.load(open(os.path.join(cls.data_dir, 'tables.bin'), 'rb'))

        cls.grammar = ASDLGrammar.from_filepath(DATASETS[cls.dataset]['grammar'])
        cls.trans = TransitionSystem.get_class_by_dataset(cls.dataset)(cls.grammar, cls.tables, cls.db_dir)
        cls.order_controller = cls.trans.order_controller
        cls.evaluator = Evaluator.get_class_by_dataset(cls.dataset)(cls.trans, table_path, cls.db_dir)

        if ts_order_path is not None and os.path.exists(ts_order_path):
            cls.order_controller.load_and_set_ts_order(ts_order_path)
        else:
            cls.order_controller.shuffle_ts_order()

        if plm is None: # do not use PLM, currently only for English text-to-SQL
            cls.word2vec = Word2vecUtils()
            cls.tokenizer = lambda x: x
            cls.word_vocab = Vocab(padding=True, unk=True, boundary=True, default=UNK,
                filepath=f'./pretrained_models/glove.42b.300d/vocab_{cls.dataset}.txt',
                specials=list(DATASETS[cls.dataset]['schema_types'].values()))
        else:
            cls.tokenizer = AutoTokenizer.from_pretrained(os.path.join('./pretrained_models', plm))
            cls.word_vocab = cls.tokenizer.get_vocab()
        cls.relation_vocab = Vocab(padding=False, unk=False, boundary=False, iterable=DATASETS[cls.dataset]['relation'], default=None)
        cls.graph_factory = GraphFactory(cls.encode_method, cls.relation_vocab)


    @classmethod
    def load_dataset(cls, choice='train', dataset=None, translator='none'):
        if dataset is None:
            assert choice in ['train', 'dev', 'test']
            choice = 'train' if DEBUG else choice
            if 'cspider' in Example.dataset and translator != 'none':
                fp = os.path.join(cls.data_dir, choice + '_' + translator + '.lgesql.bin')
            else: fp = os.path.join(cls.data_dir, choice + '.lgesql.bin')
            # fp = os.path.join(cls.data_dir, choice + '.' + cls.encode_method + '.bin')
            dataset = pickle.load(open(fp, 'rb'))
        else: choice = 'test'

        examples = []
        for idx, ex in enumerate(dataset):
            examples.append(cls(ex, cls.tables[ex['db_id']], choice + str(idx)))
            if DEBUG and len(examples) >= 100:
                break

        dataset = SQLDataset(examples)
        if choice == 'train':
            dataset = cls.order_controller.set_canonical_order_per_sample(dataset, ts_control=True, ts_shuffle=False, uts_shuffle=True)
            for ex in dataset: ex.canonical_action = get_action_infos(Example.trans.get_field_action_pairs(ex.ast, False, False))
        return dataset

    @classmethod
    def use_database_testsuite(cls):
        if cls.evaluator is not None:
            cls.evaluator.change_database(DATASETS[cls.dataset]['database_testsuite'])


    def __init__(self, ex: dict, db: dict, id: str = ''):
        super(Example, self).__init__()
        self.id = id
        self.ex = ex
        self.db = db

        """ Mapping word to corresponding index """
        if Example.plm is None: # only for english datasets, currently
            # prefix 'processed_' in key name means lemmatized version of raw lower-cased word
            self.question = ex['processed_question_toks']
            self.question_id = [Example.word_vocab[w] for w in self.question]
            self.table = [[DATASETS[Example.dataset]['schema_types']['table']] + t for t in db['processed_table_toks']]
            self.table_id = [[Example.word_vocab[w] for w in t] for t in self.table]
            self.column = [[DATASETS[Example.dataset]['schema_types'][db['column_types'][idx]]] + c + ex['processed_cells'][idx]
                if DATASETS[Example.dataset]['bridge'] else [DATASETS[Example.dataset]['schema_types'][db['column_types'][idx]]] + c
                for idx, c in enumerate(db['processed_column_toks'])]
            self.column_id = [[Example.word_vocab[w] for w in c] for c in self.column]
        else:
            t = Example.tokenizer
            self.question = [q.lower() for q in ex['uncased_question_toks']]
            self.question_id = [t.cls_token_id] # map token to id
            self.question_mask_plm = [] # remove SEP token in our case
            self.question_subword_len = [] # subword len for each word, exclude SEP token
            for w in self.question:
                toks = t.convert_tokens_to_ids(t.tokenize(w))
                self.question_id.extend(toks)
                self.question_subword_len.append(len(toks))
            self.question_mask_plm = [0] + [1] * (len(self.question_id) - 1) + [0]
            self.question_id.append(t.sep_token_id)

            # table
            self.table = [[DATASETS[Example.dataset]['schema_types']['table']] + t for t in db['table_toks']]
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

            # self.column: if bridge, [ ['text', '*'] , ['text', 'student', 'name', '=', 'alice'], ['number', 'age'], ... ]
            # prepend column types, some columns may have cell values if BRIDGE
            column = [[DATASETS[Example.dataset]['schema_types'][db['column_types'][idx]]] + c + ex['cells'][idx]
                if DATASETS[Example.dataset]['bridge'] else [DATASETS[Example.dataset]['schema_types'][db['column_types'][idx]]] + c
                for idx, c in enumerate(db['column_toks'])]
            self.column = [[DATASETS[Example.dataset]['schema_types']['time'], '当前', '时间']] + column if Example.dataset == 'dusql' else column
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
            self.segment_id = [0] * len(self.question_id) + [1] * (len(self.table_id) + len(self.column_id)) \
                if Example.plm != 'grappa_large_jnt' and not Example.plm.startswith('roberta') and 'xlm' not in Example.plm \
                else [0] * (len(self.question_id) + len(self.table_id) + len(self.column_id))

            # masks used to extract question/table/column subwords from PLM output
            self.question_mask_plm = self.question_mask_plm + [0] * (len(self.table_id) + len(self.column_id))
            self.table_mask_plm = [0] * len(self.question_id) + self.table_mask_plm + [0] * len(self.column_id)
            self.column_mask_plm = [0] * (len(self.question_id) + len(self.table_id)) + self.column_mask_plm

        self.graph = Example.graph_factory.graph_construction(ex, db)

        self.query, self.ast, self.canonical_action = '', None, []
        if 'ast' in ex: # labeled outputs
            self.query = ' '.join(ex['query'].split('\t'))
            self.ast = ex['ast']


def get_position_ids(ex, shuffle=True, dataset='spider'):
    # cluster columns with their corresponding table and randomly shuffle tables and columns
    # [CLS] q1 q2 ... [SEP] {optional TIME_NOW} {optional *} t1 c1 c2 c3 t2 c4 c5 ... [SEP]
    time_now = (dataset == 'dusql')
    db, table_word_len, column_word_len = ex.db, ex.table_word_len, ex.column_word_len
    table_num, column_num = len(db['table_names']), len(db['column_names']) + int(time_now)
    question_position_id = list(range(len(ex.question_id)))
    table_position_id, column_position_id = [None] * table_num, [None] * column_num

    # start after the question, special columns such as TIME_NOW and * first
    start = len(question_position_id)
    if dataset != 'wikisql': # wikisql does not contain wildcard column *
        column_position_id[0] = list(range(start, start + column_word_len[0]))
        start += column_word_len[0]
    if time_now:
        column_position_id[1] = list(range(start, start + column_word_len[1]))
        start += column_word_len[1]

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
            col_id = col_id + int(time_now) # maybe shifted due to TIME_NOW
            column_position_id[col_id] = list(range(start, start + column_word_len[col_id]))
            start += column_word_len[col_id]
    position_id = question_position_id + list(chain.from_iterable(table_position_id)) + \
        list(chain.from_iterable(column_position_id)) + [start]
    assert len(position_id) == len(ex.input_id)
    return position_id
