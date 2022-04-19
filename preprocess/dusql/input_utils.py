#coding=utf8
import os, sys, re, json, pickle, string
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from LAC import LAC
from rouge import Rouge
from itertools import product, combinations
from numpy.core.fromnumeric import cumsum
from utils.constants import MAX_RELATIVE_DIST, DATASETS, MAX_CELL_NUM
from preprocess.graph_utils import GraphProcessor
from preprocess.process_utils import quote_normalization, QUOTATION_MARKS, ZH_WORD2NUM, ZH_NUMBER, ZH_UNIT
from preprocess.process_utils import float_equal, is_number, is_int, load_db_contents, extract_db_contents, AGG_OP

STOPWORDS = set(["的", "是", "有", "多少", "哪些", "我", "什么", "你", "知道", "啊", "给出", "以及", "之", "从", "找", "找到", "哪里", "该", "种",
"来自", "一下", "吗", "在", "请问", "或者", "或", "想", "和", "为", "后", "那个", "是什么", "这", "对应", "并", "于", "找出", "她们", "她", "那么",
"被", "了", "并且", "都", "呢", "前", "哪个", "还有", "这个", "上", "下", "就是", "其", "它们", "及", "所", "所在", "那些", "他", "他们", "如果", "可",
"没有", "它", "要求", "谁", "了解", "不足", "时候", "个", "能", "那", "问", "中", "这些", "比", "拥有", "且", "同时", "这里", "那里", "啥", "由", "由于",
"没", "可以", "起来", "哪", "其他", "叫", "分别", "及其", "当", "之后", "都是", "过", "与", "额", "几个", "到", "占", "数", "的话", "等于", "各", "按",
"每个", "每一个", "人", "属于", "不", "不是", "值", "包含", "各个", "但", "但是", "多多少", "多少次", "多少年", "含", "加", "按照", "所有", "时", "长",
"小于", "大于", "至少", "超过", "不少", "少于", "不止", "多于", "低于", "高于", "超", "多", "少", "高", "低", "总共", "一共", "正好", "不到", "不在"])

REPLACEMENT = dict(zip('０１２３４５６７８９％：．～（）：％‘’\'`—', '0123456789%:.~():%“”""-'))
NORM = lambda s: re.sub(r'[０１２３４５６７８９％：．～（）：％‘’\'`—]', lambda c: REPLACEMENT[c.group(0)], s)

class InputProcessor():

    def __init__(self, encode_method='lgesql', db_dir='data/dusql/db_content.json', db_content=True, bridge=True, **kargs):
        super(InputProcessor, self).__init__()
        self.db_dir = db_dir
        self.db_content = db_content
        tools = LAC(mode='seg')
        self.nlp = lambda s: tools.run(s)
        self.stopwords = STOPWORDS | set(QUOTATION_MARKS + list('，。！￥？（）《》、；·…' + string.punctuation))
        if self.db_content:
            self.contents = load_db_contents(self.db_dir)
        self.bridge = bridge # whether extract candidate cell values for each column given the question
        rouge = Rouge(metrics=["rouge-1", "rouge-l"])
        self.rouge_score = lambda pred, ref: rouge.get_scores(' '.join(list(pred)), ' '.join(list(ref)))[0]
        self.graph_processor = GraphProcessor(encode_method)
        self.table_pmatch, self.table_ematch = 0, 0
        self.column_pmatch, self.column_ematch, self.column_vmatch = 0, 0, 0

    def pipeline(self, entry: dict, db: dict, verbose: bool = False):
        """ db should be preprocessed """
        entry = self.preprocess_question(entry, verbose=verbose)
        entry = self.schema_linking(entry, db, verbose=verbose)
        if self.db_content and self.bridge: # use bridge with ROUGE-L
            cells = self.bridge_content(entry, db)
            entry['cells'] = [['='] + sum([self.nlp(c) + ['，'] for c in candidates], [])[:-1] if candidates else [] for candidates in cells]
        else: entry['cells'] = [[] for _ in range(len(db['column_names']))]
        entry = self.graph_processor.process_graph_utils(entry, db)
        return entry

    def preprocess_database(self, db: dict, verbose: bool = False):
        """ Tokenize, lemmatize, lowercase table and column names for each database """
        if not db.get('table_names_original', None):
            db['table_names_original'] = db['table_names']
        if not db.get('column_names_original', None):
            db['column_names_original'] = db['column_names']

        table_toks = [list(self.nlp(tab)) for tab in db['table_names']]
        column_toks = [list(self.nlp(col)) for _, col in db['column_names']]
        db['table_toks'], db['column_toks'] = table_toks, column_toks

        column2table = list(map(lambda x: x[0], db['column_names'])) # from column id to table id
        table2columns = [[] for _ in range(len(db['table_names']))] # from table id to column ids list
        for col_id, col in enumerate(db['column_names']):
            if col_id == 0: continue
            table2columns[col[0]].append(col_id)
        db['column2table'], db['table2columns'] = column2table, table2columns

        t_num, c_num, dtype = len(db['table_names']), len(db['column_names']), '<U100'

        # relations in tables, tab_num * tab_num
        tab_mat = np.array([['table-table-generic'] * t_num for _ in range(t_num)], dtype=dtype)
        table_fks = set(map(lambda pair: (column2table[pair[0]], column2table[pair[1]]), db['foreign_keys']))
        for (tab1, tab2) in table_fks:
            if (tab2, tab1) in table_fks:
                tab_mat[tab1, tab2], tab_mat[tab2, tab1] = 'table-table-fkb', 'table-table-fkb'
            else:
                tab_mat[tab1, tab2], tab_mat[tab2, tab1] = 'table-table-fk', 'table-table-fkr'
        tab_mat[list(range(t_num)), list(range(t_num))] = 'table-table-identity'

        # relations in columns, c_num * c_num
        col_mat = np.array([['column-column-generic'] * c_num for _ in range(c_num)], dtype=dtype)
        for i in range(t_num):
            col_ids = [idx for idx, t in enumerate(column2table) if t == i]
            col1, col2 = list(zip(*list(product(col_ids, col_ids))))
            col_mat[col1, col2] = 'column-column-sametable'
        col_mat[list(range(c_num)), list(range(c_num))] = 'column-column-identity'
        if len(db['foreign_keys']) > 0:
            col1, col2 = list(zip(*db['foreign_keys']))
            col_mat[col1, col2], col_mat[col2, col1] = 'column-column-fk', 'column-column-fkr'
        col_mat[0, list(range(c_num))] = 'column-column-generic'
        col_mat[list(range(c_num)), 0] = 'column-column-generic'
        col_mat[0, 0] = 'column-column-identity'

        # relations between tables and columns, t_num*c_num and c_num*t_num
        tab_col_mat = np.array([['table-column-generic'] * c_num for _ in range(t_num)], dtype=dtype)
        col_tab_mat = np.array([['column-table-generic'] * t_num for _ in range(c_num)], dtype=dtype)
        cols, tabs = list(zip(*list(map(lambda x: (x, column2table[x]), range(1, c_num))))) # ignore *
        col_tab_mat[cols, tabs], tab_col_mat[tabs, cols] = 'column-table-has', 'table-column-has'
        if len(db['primary_keys']) > 0:
            cols, tabs = list(zip(*list(map(lambda x: (x, column2table[x]), db['primary_keys']))))
            col_tab_mat[cols, tabs], tab_col_mat[tabs, cols] = 'column-table-pk', 'table-column-pk'
        col_tab_mat[0, list(range(t_num))] = 'column-table-has' # column-table-generic
        tab_col_mat[list(range(t_num)), 0] = 'table-column-has' # table-column-generic

        # special column TIME_NOW
        tab_tn_mat = np.array([['table-column-generic'] for _ in range(t_num)], dtype=dtype)
        tn_tab_mat = np.array([['column-table-generic'] * t_num], dtype=dtype)
        col_ids = list(filter(lambda x: db['column_types'][x] == 'time', range(c_num)))
        col_tn_mat = np.array([['column-column-generic'] for _ in range(c_num)], dtype=dtype)
        col_tn_mat[col_ids, 0] = 'column-column-time'
        tn_col_mat = np.array([['column-column-generic'] * c_num], dtype=dtype)
        tn_col_mat[0, col_ids] = 'column-column-time'
        tn_mat = np.array([['column-column-identity']], dtype=dtype)

        relations = np.concatenate([
            np.concatenate([tab_mat, tab_tn_mat, tab_col_mat], axis=1),
            np.concatenate([tn_tab_mat, tn_mat, tn_col_mat], axis=1),
            np.concatenate([col_tab_mat, col_tn_mat, col_mat], axis=1)
        ], axis=0)
        db['relations'] = relations.tolist()

        if self.db_content:
            db['cells'] = extract_db_contents(self.contents, db)
            db['processed_cells'] = [[transform_word_to_number(normalize_cell_value(c)) if db['column_types'][cid] == 'number' else normalize_cell_value(c)
                for c in cvs] for cid, cvs in enumerate(db['cells'])]
        if verbose:
            print('Tokenized tables:', ', '.join(['|'.join(tab) for tab in table_toks]))
            print('Tokenized columns:', ', '.join(['|'.join(col) for col in column_toks]), '\n')
        return db

    def normalize_question(self, entry):
        question = entry['question']
        # fix some problems, tem_ -> item_
        question = re.sub(r'([^i])tem_', lambda match_obj: match_obj.group(1) + 'item_', question)
        # construct item_xxx_xxx dict, use variables item1, item2, item3, ...
        item_mapping, item_mapping_reverse, idx = {}, [], 0
        for match_obj in re.finditer(r'item_[\._a-z0-9]+', question):
            span = match_obj.group(0)
            if span not in item_mapping:
                index = 'item' + str(idx)
                item_mapping[span] = index
                item_mapping_reverse.append(span)
                idx += 1 # increase the num index
        entry['item_mapping'], entry['item_mapping_reverse'] = item_mapping, item_mapping_reverse
        for raw_item in sorted(item_mapping.keys(), key=lambda k: - len(k)): # add whitespace to avoid tokenization error
            question = question.replace(raw_item, ' ' + item_mapping[raw_item] + ' ')
        # quote and numbers normalization
        question = quote_normalization(question)
        entry['question'] = re.sub(r'\s+', ' ', NORM(question))
        return entry

    def preprocess_question(self, entry: dict, verbose: bool = False):
        """ Tokenize, lemmatize, lowercase question"""
        # LAC tokenize
        entry = self.normalize_question(entry)
        cased_toks = re.sub(r'\s+', ' ', ' '.join(self.nlp(entry['question']))).split(' ')

        # tokenization errors in train/dev set which will influence the value extractor
        if '汤姆·梅恩比巴克里希纳·多西多' in cased_toks:
            index = cased_toks.index('汤姆·梅恩比巴克里希纳·多西多')
            cased_toks[index: index + 1] = ['汤姆·梅恩', '比', '巴克里希纳·多西', '多']
        elif '北京首都国际机场' in cased_toks:
            index = cased_toks.index('北京首都国际机场')
            cased_toks[index: index + 1] = ['北京', '首都国际机场']
        elif '德克萨斯比拉斯维加斯' in cased_toks:
            index = cased_toks.index('德克萨斯比拉斯维加斯')
            cased_toks[index: index + 1] = ['德克萨斯', '比', '拉斯维加斯']

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
        table_toks, column_toks = db['table_toks'], db['column_toks']
        table_names, column_names = [''.join(toks) for toks in table_toks], [''.join(toks) for toks in column_toks]
        q_num, question, dtype = len(question_toks), ''.join(question_toks), '<U100'

        def question_schema_matching_method(schema_toks, schema_names, category):
            assert category in ['table', 'column']
            s_num, matched_pairs = len(schema_names), {'partial': [], 'exact': []}
            q_s_mat = np.array([[f'question-{category}-nomatch'] * s_num for _ in range(q_num)], dtype=dtype)
            s_q_mat = np.array([[f'{category}-question-nomatch'] * q_num for _ in range(s_num)], dtype=dtype)
            for sid, name in enumerate(schema_names):
                if category == 'column' and sid == 0: continue
                max_len = len(schema_toks[sid])
                index_pairs = sorted(filter(lambda x: 0 < x[1] - x[0] <= max_len, combinations(range(q_num + 1), 2)), key=lambda x: x[1] - x[0])
                for start, end in index_pairs:
                    span = ''.join(question_toks[start:end])
                    if span in self.stopwords: continue
                    if (end - start == 1 and span in schema_toks[sid]) or (end - start > 1 and span in name):
                        # tradeoff between precision and recall
                        q_s_mat[range(start, end), sid] = f'question-{category}-partialmatch'
                        s_q_mat[sid, range(start, end)] = f'{category}-question-partialmatch'
                        if verbose:
                            matched_pairs['partial'].append(str((schema_names[sid], sid, span, start, end)))
                # exact match, considering tokenization errors
                if name in question:
                    pos = [span.start() for span in re.finditer(name.replace('(', '\(').replace(')', '\)'), question)]
                    for start_id in pos:
                        start, end = entry['char2word_id_mapping'][start_id], entry['char2word_id_mapping'][start_id + len(name) - 1] + 1
                        q_s_mat[range(start, end), sid] = f'question-{category}-exactmatch'
                        s_q_mat[sid, range(start, end)] = f'{category}-question-exactmatch'
                        if verbose:
                            matched_pairs['exact'].append(str((schema_names[sid], sid, ''.join(question_toks[start:end]), start, end)))
            return q_s_mat, s_q_mat, matched_pairs

        q_tab_mat, tab_q_mat, table_matched_pairs = question_schema_matching_method(table_toks, table_names, 'table')
        self.table_pmatch += np.sum(q_tab_mat == 'question-table-partialmatch')
        self.table_ematch += np.sum(q_tab_mat == 'question-table-exactmatch')
        q_col_mat, col_q_mat, column_matched_pairs = question_schema_matching_method(column_toks, column_names, 'column')
        self.column_pmatch += np.sum(q_col_mat == 'question-column-partialmatch')
        self.column_ematch += np.sum(q_col_mat == 'question-column-exactmatch')

        if self.db_content:
            # create question-value-match relations, be careful with item_mapping
            column_matched_pairs['value'] = []

            def extract_number(word):
                if re.search(r'^item', word): return word
                if is_number(word): return str(float(word))
                match_obj = re.search(r'^[^\d\.]*([\d\.]+)[^\d\.]*$', word)
                if match_obj:
                    span = match_obj.group(1)
                    if is_number(span):
                        return str(float(span))
                return word

            words = [entry['item_mapping_reverse'][int(word.strip('item'))] if word.startswith('item') else word for word in question_toks]
            num_words = [extract_number(word) for word in words]
            for cid, col_name in enumerate(column_names):
                if cid == 0: continue
                cells = [c.lower() for c in db['cells'][cid]] # list of cell values, ['2014', '2015']
                num_cells = [extract_number(c) for c in cells]
                for qid, (word, num_word) in enumerate(zip(words, num_words)):
                    if 'nomatch' in q_col_mat[qid, cid] and word not in self.stopwords:
                        for c, nc in zip(cells, num_cells):
                            if word in c or num_word == nc:
                                q_col_mat[qid, cid] = 'question-column-valuematch'
                                col_q_mat[cid, qid] = 'column-question-valuematch'
                                if verbose:
                                    column_matched_pairs['value'].append(str((col_name, cid, c, word, qid, qid + 1)))
                                break
            self.column_vmatch += np.sum(q_col_mat == 'question-column-valuematch')

        # two symmetric schema linking matrix: q_num x (t_num + c_num), (t_num + c_num) x q_num
        q_col_mat[:, 0] = 'question-column-nomatch'
        col_q_mat[0] = 'column-question-nomatch'
        q_tn_mat = np.array([['question-column-nomatch'] for _ in range(q_num)], dtype=dtype)
        tn_q_mat = np.array([['column-question-nomatch'] * q_num], dtype=dtype)
        q_schema = np.concatenate([q_tab_mat, q_tn_mat, q_col_mat], axis=1)
        schema_q = np.concatenate([tab_q_mat, tn_q_mat, col_q_mat], axis=0)
        entry['schema_linking'] = (q_schema.tolist(), schema_q.tolist())

        if verbose:
            print('Question:', ' '.join(question_toks))
            print('Table matched: (table name, table id, question span, start id, end id)')
            print('Exact match:', ', '.join(table_matched_pairs['exact']) if table_matched_pairs['exact'] else 'empty')
            print('Partial match:', ', '.join(table_matched_pairs['partial']) if table_matched_pairs['partial'] else 'empty')
            print('Column matched: (column name, column id, question span, start id, end id)')
            print('Exact match:', ', '.join(column_matched_pairs['exact']) if column_matched_pairs['exact'] else 'empty')
            print('Partial match:', ', '.join(column_matched_pairs['partial']) if column_matched_pairs['partial'] else 'empty')
            if self.db_content:
                print('Value match: (column name, column_id, cell value, question word, start id, end id)')
                print(', '.join(column_matched_pairs['value']) if column_matched_pairs['value'] else 'empty')
            print('\n')
        return entry

    def bridge_content(self, entry, db):
        # extract candidate cell values for each column given the current question
        cells, raw_cells, col_types = db['processed_cells'], db['cells'], db['column_types']
        question_toks = entry['uncased_question_toks']
        numbers = extract_numbers_in_question(''.join(question_toks))
        question = ''.join(filter(lambda s: s not in self.stopwords, question_toks))

        def number_score(c, numbers):
            if (not is_number(c)) or len(numbers) == 0: return 0.
            return max([1. if float_equal(c, r) else 0.5 if float_equal(c, r, 100) or float_equal(c, r, 1e3) or float_equal(c, r, 1e4) or float_equal(c, r, 1e8) else 0. for r in numbers])

        candidates = [[]] # map column_id to candidate values relevant to the question
        for col_id, col_cells in enumerate(cells):
            if col_id == 0: continue
            tmp_candidates = []
            for c in col_cells:
                if c.startswith('item_') and c in entry['item_mapping']:
                    tmp_candidates.append(entry['item_mapping'][c])
            if len(tmp_candidates) > 0:
                candidates.append(tmp_candidates)
                continue
            if col_types[col_id] == 'binary': candidates.append([])
            elif col_types[col_id] == 'time':
                candidates.append([c for c in col_cells if c in question][:MAX_CELL_NUM])
            elif col_types[col_id] == 'number':
                scores = sorted(filter(lambda x: x[1] > 0, [(cid, number_score(c, numbers)) for cid, c in enumerate(col_cells)]), key=lambda x: - x[1])[:MAX_CELL_NUM]
                if len(scores) > 1:
                    scores = scores[:1] + list(filter(lambda x: x[1] >= 0.6, scores[1:]))
                candidates.append([normalize_cell_value(raw_cells[col_id][cid]) for cid, _ in scores])
            else: # by default, text
                scores = [(c, self.rouge_score(c, question)) for c in col_cells if 0 < len(c) < 50 and c != '.']
                scores = sorted(filter(lambda x: x[1]['rouge-l']['f'] > 0, scores), key=lambda x: (- x[1]['rouge-l']['f'], - x[1]['rouge-1']['p']))[:MAX_CELL_NUM]
                if len(scores) > 1: # at most two cells but the second one must have high rouge-1 precision
                    scores = scores[:1] + list(filter(lambda x: x[1]['rouge-1']['p'] >= 0.6, scores[1:]))
                candidates.append([c for c, _ in scores])
        return candidates

def normalize_cell_value(c):
    return re.sub(r'\s+', ' ', NORM(str(c).strip().lower()))

def extract_numbers_in_question(question):
    candidates = []
    question = re.sub(r'(千米|千克|千瓦|千卡|千斤|百分之|item\d+|km|kg|cm)', '', question, flags=re.I)
    for span in re.finditer(r'([0-9\.点负\-%s%s]+)' % (ZH_NUMBER, ZH_UNIT), question):
        s, e = span.start(), span.end()
        word = question[s: e]
        if s > 0 and re.search(r'([a-z每年月周_/]|星期)', question[s - 1]): continue
        if e < len(question) and re.search(r'[a-z些批部层楼下手共月日星号时分秒股线_\-/]', question[e]): continue
        if is_number(word):
            candidates.append(str(float(word)))
        try:
            parsed_num = ZH_WORD2NUM(word.rstrip('万亿'))
            candidates.append(str(float(parsed_num)))
        except Exception as e: pass
    return candidates

def transform_word_to_number(word):
    """ Transform the number occurred in the word
    """
    word = str(word)
    if word.startswith('item_'): return word
    word = re.sub(r'(千米|千克|千瓦|千卡|千斤|百分之|kg|km|cm)', '', word, flags=re.I)
    match_obj = re.search(r'[0-9\.负\-%s%s]+' % (ZH_NUMBER, ZH_UNIT), word)
    if match_obj:
        span = match_obj.group(0)
        s, e = match_obj.start(), match_obj.end()
        if not (s > 0 and re.search(r'([a-z每年月周_/]|星期)', word[s - 1], flags=re.I)) and not (e < len(word) and re.search(r'[a-z些批部层楼下手共月日星号时分秒股线_\-/]', word[e])):
            if is_number(span): return str(float(span))
            try:
                parsed_num = ZH_WORD2NUM(word.rstrip('万亿'))
                return str(float(parsed_num))
            except: pass
    return word


if __name__ == '__main__':
    import time
    from collections import defaultdict
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('./pretrained_models/chinese-macbert-large')
    # check BRIDGE function
    processor = InputProcessor()
    data_dir = DATASETS['dusql']['data']
    tables = pickle.load(open(os.path.join(data_dir, 'tables.bin'), 'rb'))
    train, dev = pickle.load(open(os.path.join(data_dir, 'train.lgesql.bin'), 'rb')), pickle.load(open(os.path.join(data_dir, 'dev.lgesql.bin'), 'rb'))
    count = defaultdict(lambda : [0, 0])
    bridge_count = defaultdict(lambda : [0, 0])
    input_lens, dataset = [], dev
    start_time = time.time()

    def extract_values(sql, s):
        table_units = sql['from']['table_units']
        if table_units[0][0] == 'sql':
            s = extract_values(sql['from']['table_units'][0][1], s)
            s = extract_values(sql['from']['table_units'][1][1], s)
        for cond in sql['where']:
            if cond in ['and', 'or']: continue
            _, _, val_unit, val, _ = cond
            if type(val) == dict: s = extract_values(val, s)
            else:
                col_id = val_unit[1][1]
                if type(col_id) == str: continue # TIME_NOW
                val = normalize_cell_value(str(val))
                s.add((col_id, val))
        for cond in sql['having']:
            if cond in ['and', 'or']: continue
            agg_op, _, val_unit, val, _ = cond
            if type(val) == dict: s = extract_values(val, s)
            elif AGG_OP[agg_op] in ['max', 'min']:
                col_id = val_unit[1][1]
                if type(col_id) == str: continue # TIME_NOW
                val = normalize_cell_value(str(val))
                s.add((col_id, val))
        for choice in ['intersect', 'union', 'except']:
            if sql[choice]: s = extract_values(sql[choice], s)
        return s

    for jdx, ex in enumerate(dataset):
        if (jdx + 1) % 1000 == 0:
            print('Processing %d-th example ...' % (jdx + 1))
        db = tables[ex['db_id']]
        question_toks, col_types = ex['uncased_question_toks'], db['column_types']
        cells = processor.bridge_content(ex, db)
        processed_cells = [['='] + sum([processor.nlp(c) + ['，'] for c in candidates], [])[:-1] if candidates else [] for candidates in cells]
        values = extract_values(ex['sql'], set())
        for cid, cv in values:
            ct = col_types[cid]
            if ct == 'number':
                if len(cells[cid]) > 0:
                    count[ct][0] += 1
                    count['all'][0] += 1
            else:
                if cv in cells[cid]:
                    count[ct][0] += 1
                    count['all'][0] += 1
            count[ct][1] += 1
            count['all'][1] += 1

        for cid, col_cells in enumerate(cells):
            bridge_count[col_types[cid]][0] += len(col_cells)
            bridge_count[col_types[cid]][1] += 1
            bridge_count['all'][0] += len(col_cells)
            bridge_count['all'][1] += 1

        table = [[DATASETS['dusql']['schema_types']['table']] + t for t in db['table_toks']]
        column = [[DATASETS['dusql']['schema_types'][db['column_types'][idx]]] + c + processed_cells[idx]
                for idx, c in enumerate(db['column_toks'])]

        if False:
            print(' '.join(question_toks))
            print(ex['query'])
            print('\n'.join([' '.join(db['column_toks'][cid]) + '[%s] ' % (col_types[cid]) + ' '.join(processed_cells[cid]) for cid, _ in values]))
            print('\n')

        toks = sum([question_toks] + table + column, []) # ensure that input_length < 512
        input_len = len([tokenizer.tokenize(w) for w in toks]) + 3 # plus 3, CLS SEP SEP
        input_lens.append(input_len)

    print('In total, true/all SQL value count for each column type is:\n', ' , '.join([k + '->' + '%d/%d' % (count[k][0], count[k][1]) for k in count]))
    print('In total, bridge values/columns count for each column type is:\n', ' , '.join([k + '->' + '%d/%d' % (bridge_count[k][0], bridge_count[k][1]) for k in bridge_count]))
    print('MAX/MIN/AVG input len with PLM is %s/%s/%.2f' % (max(input_lens), min(input_lens), sum(input_lens) / float(len(dataset))))
    print('Cost %.2fs .' % (time.time() - start_time))