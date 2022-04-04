#coding=utf8
import os
from collections import OrderedDict

DEBUG = False
TEST = False

PAD = '[PAD]'
BOS = '[CLS]'
EOS = '[SEP]'
UNK = '[UNK]'

MAX_RELATIVE_DIST = 2
MAX_CELL_NUM = 2

# relations: type_1-type_2-relation_name, r represents reverse edge, b represents bidirectional edge
NONLOCAL_RELATIONS = [
    'question-question-generic', 'table-table-generic', 'column-column-generic', 'table-column-generic', 'column-table-generic',
    'table-table-fk', 'table-table-fkr', 'table-table-fkb', 'column-column-sametable',
    'question-question-identity', 'table-table-identity', 'column-column-identity'] + [
    'question-question-dist' + str(i) for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1, 1) if i not in [-1, 0, 1]
]
# column-column-time for link between TIME_NOW and any column with type 'time'
RELATIONS = ['question-question-dist' + str(i) if i != 0 else 'question-question-identity' for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1)] + \
    ['table-table-identity', 'table-table-fk', 'table-table-fkr', 'table-table-fkb'] + \
    ['column-column-identity', 'column-column-sametable', 'column-column-fk', 'column-column-fkr'] + \
    ['table-column-pk', 'column-table-pk', 'table-column-has', 'column-table-has'] + \
    ['question-column-exactmatch', 'question-column-partialmatch', 'question-column-nomatch', 'question-column-valuematch',
    'column-question-exactmatch', 'column-question-partialmatch', 'column-question-nomatch', 'column-question-valuematch'] + \
    ['question-table-exactmatch', 'question-table-partialmatch', 'question-table-nomatch',
    'table-question-exactmatch', 'table-question-partialmatch', 'table-question-nomatch'] + \
    ['question-question-generic', 'table-table-generic', 'column-column-generic', 'table-column-generic', 'column-table-generic'] + \
    ['column-column-time'] # only used for DuSQL, which has a special column 'TIME_NOW'

DATASETS = {
    'spider': {
        'grammar': os.path.join('asdl', 'spider', 'spider_grammar.txt'), # os.path.join('asdl', 'cspider_raw', 'cspider_raw_grammar_simple.txt')
        'relation': RELATIONS[:-1],
        'data': os.path.join('data', 'spider'),
        'database': os.path.join('data', 'spider', 'database'),
        'database_testsuite': os.path.join('data', 'spider', 'database-testsuite'),
        'db_content': True,
        'bridge': True, # False
        'predict_value': True, # False,
        'schema_types': OrderedDict([(t, t) for t in ['table', 'text', 'time', 'number', 'boolean', 'others']]),
    },
    'dusql': {
        'grammar': os.path.join('asdl', 'dusql', 'dusql_grammar.txt'),
        'relation': RELATIONS,
        'data': os.path.join('data', 'dusql'),
        'database': os.path.join('data', 'dusql', 'db_content.json'),
        'database_testsuite': os.path.join('data', 'dusql', 'db_content.json'),
        'db_content': True,
        'bridge': False,
        'predict_value': True,
        'schema_types': OrderedDict([('table', '表格'), ('text', '文本'), ('time', '时间'), ('number', '数值'), ('binary', '真假值'), ('others', '其它类型')]),
    },
    'cspider': {
        'grammar': os.path.join('asdl', 'cspider', 'cspider_grammar.txt'), # os.path.join('asdl', 'cspider_raw', 'cspider_raw_grammar_simple.txt')
        'relation': RELATIONS[:-1],
        'data': os.path.join('data', 'cspider'),
        'database': os.path.join('data', 'cspider', 'db_content.json'),
        'database_testsuite': os.path.join('data', 'cspider', 'db_content.json'),
        'cache_folder': './pretrained_models',
        'db_content': True,
        'bridge': False,
        'predict_value': False, # True,
        'schema_types': OrderedDict([(t, t) for t in ['table', 'text', 'time', 'number', 'boolean', 'others']]),
    },
    'cspider_raw': {
        'grammar': os.path.join('asdl', 'cspider_raw', 'cspider_raw_grammar.txt'), # os.path.join('asdl', 'cspider_raw', 'cspider_raw_grammar_simple.txt'),
        'relation': RELATIONS[:-1],
        'data': os.path.join('data', 'cspider_raw'),
        'database': os.path.join('data', 'cspider_raw', 'database'),
        'database_testsuite': os.path.join('data', 'cspider_raw', 'database'),
        'cache_folder': './pretrained_models',
        'db_content': True,
        'bridge': False,
        'predict_value': False,
        'schema_types': OrderedDict([(t, t) for t in ['table', 'text', 'time', 'number', 'boolean', 'others']]),
    },
    'nl2sql': {
        'grammar': os.path.join('asdl', 'nl2sql', 'nl2sql_grammar.txt'),
        'relation': RELATIONS[:-1],
        'data': os.path.join('data', 'nl2sql'),
        'database': os.path.join('data', 'nl2sql', 'db_content.json'),
        'database_testsuite': os.path.join('data', 'nl2sql', 'db_content.json'),
        'db_content': True,
        'bridge': True,
        'predict_value': True,
        'schema_types': OrderedDict([('table', '表格'), ('text', '文本'), ('real', '数值')]),
    },
}

# Index for BIO Labels in value recognition:
# O -> 0 ; B -> 1 ; I -> 2
