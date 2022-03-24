#coding=utf8
import re, json, string
import numpy as np
from LAC import LAC
from itertools import combinations
from numpy.core.fromnumeric import cumsum
from utils.constants import MAX_RELATIVE_DIST
from preprocess.graph_utils import GraphProcessor
from preprocess.process_utils import is_number, quote_normalization, QUOTATION_MARKS, load_db_contents, extract_db_contents

NUMBER_REPLACEMENT = list(zip('０１２３４５６７８９％：．～', '0123456789%:.~'))

STOPWORDS = set(["的", "是", "有", "多少", "哪些", "我", "什么", "你", "知道", "啊", "给出", "以及", "之", "从", "找", "找到", "哪里", "该", "种", "吧", "请",
"来自", "一下", "吗", "在", "请问", "或者", "或", "想", "和", "为", "后", "那个", "是什么", "这", "对应", "并", "于", "找出", "她们", "她", "那么", "查查", "就",
"被", "了", "并且", "都", "呢", "前", "哪个", "还有", "这个", "上", "下", "就是", "其", "它们", "及", "所", "所在", "那些", "他", "他们", "如果", "可", "哪家", "里",
"没有", "它", "要求", "谁", "了解", "不足", "时候", "个", "能", "那", "问", "中", "这些", "比", "拥有", "且", "同时", "这里", "那里", "啥", "由", "由于", "看看",
"没", "可以", "起来", "哪", "其他", "叫", "分别", "及其", "当", "之后", "都是", "过", "与", "额", "几个", "到", "占", "数", "的话", "等于", "各", "按", "给", "哦",
"每个", "每一个", "人", "属于", "不", "不是", "值", "包含", "各个", "但", "但是", "多多少", "多少次", "多少年", "含", "加", "按照", "所有", "时", "长", "需要", "还是",
"小于", "大于", "至少", "超过", "不少", "少于", "不止", "多于", "低于", "高于", "超", "多", "少", "高", "低", "总共", "一共", "正好", "不到", "不在", "想问问", "做",
"帮", "你好", "查", "呀", "要", "以上", "想知道", "这本书", "这本", "告诉我", "想要", "达到", "什么时候", "也", "而且", "麻烦", "问一下", "又", "诶", "谢谢", "算算",
"能不能", "有没有", "几家", "还", "去", "只", "得", "现在", "跟", "这次", "目前", "查询", "看", "哪几个", "怎样", "这部", "多大", "进行", "以下", "怎么", "目前有",
"说说", "哪位", "呗", "告知", "用", "如何", "这种", "才能", "了解到", "呃", "这边", "话", "哪一个", "看一下", "这样", "已经", "那本书", "怎么样", "几本", "看到"])

class InputProcessor():

    def __init__(self, encode_method='lgesql', db_dir='data/nl2sql/db_content.json', db_content=True, bridge=False, **kargs):
        super(InputProcessor, self).__init__()
        self.db_dir = db_dir
        self.db_content = db_content
        tools = LAC(mode='seg')
        self.nlp = lambda s: tools.run(s)
        self.stopwords = STOPWORDS | set(QUOTATION_MARKS + list('，。！￥？（）《》、；·…' + string.punctuation))
        if self.db_content:
            self.contents = load_db_contents(self.db_dir)
        self.bridge = bridge # whether extract candidate cell values for each column given the question
        self.graph_processor = GraphProcessor(encode_method)
        self.table_pmatch, self.table_ematch = 0, 0
        self.column_pmatch, self.column_ematch, self.column_vmatch = 0, 0, 0

    def pipeline(self, entry: dict, db: dict, verbose: bool = False):
        """ db should be preprocessed """
        entry = self.preprocess_question(entry, verbose=verbose)
        # entry = self.schema_linking(entry, db, verbose=verbose)
        # entry = self.graph_processor.process_graph_utils(entry, db)
        return entry

    def preprocess_database(self, db: dict, verbose: bool = False):
        """ Tokenize, lemmatize, lowercase table and column names for each database """
        table_toks = [['数据库']] # only one table, use a special name
        # normalize brackets
        mappings = dict(zip('（）：％‘’\'`', '():%“”""'))
        column_toks = [self.nlp(re.sub(r'[（）：％‘’\'`]', lambda obj: mappings[obj.group(0)], col)) for _, col in db['column_names']]
        column_toks = [re.sub(r'\s+', ' ', ' '.join(toks)).split(' ') for toks in column_toks]
        db['table_toks'], db['column_toks'] = table_toks, column_toks

        column2table = list(map(lambda x: x[0], db['column_names'])) # from column id to table id
        table2columns = [list(range(1, len(db['column_names'])))] # from table id to column ids list
        db['column2table'], db['table2columns'] = column2table, table2columns

        t_num, c_num, dtype = len(db['table_names']), len(db['column_names']), '<U100'

        # relations in tables, tab_num * tab_num
        tab_mat = np.array([['table-table-identity']], dtype=dtype)

        # relations in columns, c_num * c_num
        col_mat = np.array([['column-column-sametable'] * c_num for _ in range(c_num)], dtype=dtype)
        col_mat[list(range(c_num)), list(range(c_num))] = 'column-column-identity'

        # relations between tables and columns, t_num*c_num and c_num*t_num
        tab_col_mat = np.array([['table-column-has'] * c_num for _ in range(t_num)], dtype=dtype)
        col_tab_mat = np.array([['column-table-has'] * t_num for _ in range(c_num)], dtype=dtype)

        relations = np.concatenate([
            np.concatenate([tab_mat, tab_col_mat], axis=1),
            np.concatenate([col_tab_mat, col_mat], axis=1)
        ], axis=0)
        db['relations'] = relations.tolist()

        if self.db_content:
            db['cells'] = extract_db_contents(self.contents, db)
        if verbose:
            print('Tokenized tables:', ', '.join(['|'.join(tab) for tab in table_toks]))
            print('Tokenized columns:', ', '.join(['|'.join(col) for col in column_toks]), '\n')
        return db

    def normalize_question(self, entry):
        question = entry['question']
        # quote and numbers normalization
        question = quote_normalization(question)
        for raw, new in NUMBER_REPLACEMENT:
            question = question.replace(raw, new)
        question = re.sub(r'那个', '', question).replace('一平', '每平') # influence value recognition
        entry['question'] = re.sub(r'\s+', ' ', question).strip(' \t\n!&,.:;=?@^_`~，。？！；、…')
        return entry

    def preprocess_question(self, entry: dict, verbose: bool = False):
        """ Tokenize, lemmatize, lowercase question"""
        # LAC tokenize
        question = self.normalize_question(entry)['question'].replace('丰和支行', ' 丰和支行 ')
        cased_toks = re.sub(r'\s+', ' ', ' '.join(self.nlp(question))).split(' ')

        # tokenization errors in train/dev set which will influence the value extractor

        toks = [w.lower() for w in cased_toks]
        entry['cased_question_toks'] = cased_toks
        entry['uncased_question_toks'] = toks
        entry['question'] = ''.join(cased_toks)
        # map raw question_char_position_id to question_word_position_id, and reverse
        entry['char2word_id_mapping'] = [idx for idx, w in enumerate(toks) for _ in range(len(w))]
        entry['word2char_id_mapping'] = cumsum([0] + [len(w) for w in toks]).tolist()

        # relations in questions, q_num * q_num
        q_num, dtype = len(toks), '<U100'
        if q_num <= MAX_RELATIVE_DIST + 1:
            dist_vec = ['question-question-dist' + str(i) if i != 0 else 'question-question-identity'
                for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1, 1)]
            starting = MAX_RELATIVE_DIST
        else:
            dist_vec = ['question-question-generic'] * (q_num - MAX_RELATIVE_DIST - 1) + \
                ['question-question-dist' + str(i) if i != 0 else 'question-question-identity' \
                    for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1, 1)] + \
                    ['question-question-generic'] * (q_num - MAX_RELATIVE_DIST - 1)
            starting = q_num - 1
        q_mat = np.array([dist_vec[starting - i: starting - i + q_num] for i in range(q_num)], dtype=dtype)
        entry['relations'] = q_mat.tolist()

        if verbose:
            print('Tokenized question:', ' '.join(entry['cased_question_toks']))
        return entry

    def schema_linking(self, entry: dict, db: dict, verbose: bool = False):
        """ Perform schema linking: both question and database need to be preprocessed """
        question_toks = entry['uncased_question_toks']
        column_toks = db['column_toks']
        column_names = [''.join(toks) for toks in column_toks]
        q_num, question, dtype = len(question_toks), ''.join(question_toks), '<U100'

        def question_schema_matching_method(schema_toks, schema_names, category):
            assert category in ['table', 'column']
            s_num, matched_pairs = len(schema_names), {'partial': [], 'exact': []}
            q_s_mat = np.array([[f'question-{category}-nomatch'] * s_num for _ in range(q_num)], dtype=dtype)
            s_q_mat = np.array([[f'{category}-question-nomatch'] * q_num for _ in range(s_num)], dtype=dtype)
            max_len = max([len(toks) for toks in schema_toks])
            index_pairs = sorted(filter(lambda x: 0 < x[1] - x[0] <= max_len, combinations(range(q_num + 1), 2)), key=lambda x: x[1] - x[0])
            for sid, name in enumerate(schema_names):
                if category == 'column' and sid == 0: continue
                current_len = len(schema_toks[sid])
                for start, end in index_pairs:
                    if end - start > current_len: break
                    span = ''.join(question_toks[start:end])
                    if span in self.stopwords: continue
                    if (end - start == 1 and span in schema_toks[sid]) or (end - start > 1 and span in name):
                        # tradeoff between precision and recall
                        q_s_mat[range(start, end), sid] = f'question-{category}-partialmatch'
                        s_q_mat[sid, range(start, end)] = f'{category}-question-partialmatch'
                        if verbose:
                            matched_pairs['partial'].append(str((schema_names[sid], sid, span, start, end)))
                # exact match, considering tokenization errors
                idx, name = 0, re.sub(r'\(.*?\)', '', name).strip() # remove metrics in brackets
                if len(name) == 0: continue
                while idx <= len(question) - 1:
                    if name in question[idx:]:
                        start_id = question.index(name, idx)
                        start, end = entry['char2word_id_mapping'][start_id], entry['char2word_id_mapping'][start_id + len(name) - 1] + 1
                        q_s_mat[range(start, end), sid] = f'question-{category}-exactmatch'
                        s_q_mat[sid, range(start, end)] = f'{category}-question-exactmatch'
                        if verbose:
                            matched_pairs['exact'].append(str((schema_names[sid], sid, ''.join(question_toks[start:end]), start, end)))
                        idx += len(name)
                    else: break
            return q_s_mat, s_q_mat, matched_pairs

        q_tab_mat = np.array([['question-table-nomatch'] for _ in range(q_num)], dtype=dtype)
        tab_q_mat = np.array([['table-question-nomatch'] * q_num], dtype=dtype)
        q_col_mat, col_q_mat, column_matched_pairs = question_schema_matching_method(column_toks, column_names, 'column')
        self.column_pmatch += np.sum(q_col_mat == 'question-column-partialmatch')
        self.column_ematch += np.sum(q_col_mat == 'question-column-exactmatch')

        if self.db_content:
            # create question-value-match relations, be careful with item_mapping
            column_matched_pairs['value'] = []

            def extract_number(word):
                if is_number(word): return str(float(word))
                match_obj = re.search(r'^[^\d\.]*([\d\.]+)[^\d\.]*$', word)
                if match_obj:
                    span = match_obj.group(1)
                    if is_number(span):
                        return str(float(span))
                return word

            num_words = [extract_number(word) for word in question_toks]
            for cid, col_name in enumerate(column_names):
                if cid == 0: continue
                cells = [c.lower() for c in db['cells'][cid]] # list of cell values, ['2014', '2015']
                num_cells = [extract_number(c) for c in cells]
                for qid, (word, num_word) in enumerate(zip(question_toks, num_words)):
                    if 'nomatch' in q_col_mat[qid, cid] and word not in self.stopwords:
                        for c, nc in zip(cells, num_cells):
                            if word in c or num_word == nc:
                                q_col_mat[qid, cid] = 'question-column-valuematch'
                                col_q_mat[cid, qid] = 'column-question-valuematch'
                                if verbose:
                                    column_matched_pairs['value'].append(str((col_name, cid, c, word, qid, qid + 1)))
                                break
            self.column_vmatch += np.sum(q_col_mat == 'question-column-valuematch')

        # no bridge
        entry['cells'] = [[] for _ in range(len(db['column_names']))]

        # two symmetric schema linking matrix: q_num x (t_num + c_num), (t_num + c_num) x q_num
        q_col_mat[:, 0] = 'question-column-nomatch'
        col_q_mat[0] = 'column-question-nomatch'
        q_schema = np.concatenate([q_tab_mat, q_col_mat], axis=1)
        schema_q = np.concatenate([tab_q_mat, col_q_mat], axis=0)
        entry['schema_linking'] = (q_schema.tolist(), schema_q.tolist())

        if verbose:
            print('Question:', ' '.join(question_toks))
            print('Column matched: (column name, column id, question span, start id, end id)')
            print('Exact match:', ', '.join(column_matched_pairs['exact']) if column_matched_pairs['exact'] else 'empty')
            print('Partial match:', ', '.join(column_matched_pairs['partial']) if column_matched_pairs['partial'] else 'empty')
            if self.db_content:
                print('Value match: (column name, column_id, cell value, question word, start id, end id)')
                print(', '.join(column_matched_pairs['value']) if column_matched_pairs['value'] else 'empty')
            print('\n')
        return entry
