#coding=utf8
import re, json, jieba
import numpy as np
from LAC import LAC
import cn2an
from itertools import product, combinations
from numpy.core.fromnumeric import cumsum
from utils.constants import MAX_RELATIVE_DIST
from preprocess.graph_utils import GraphProcessor
from preprocess.process_utils import is_number, quote_normalization
from preprocess.dusql.bridge_content_encoder import STOPWORDS, get_database_matches

def is_word_number(word):
    try:
        word = word.rstrip('个种类条只元米款')
        num = cn2an.cn2an(word, 'smart')
        return True
    except: return False

def load_db_contents(db_path):
    contents = json.load(open(db_path, 'r'))
    contents_dict = {}
    for db in contents:
        contents_dict[db['db_id']] = db['tables']
    return contents_dict

def extract_db_cells(contents, db):
    db_cells = [['*']]
    cells = contents[db['db_id']]
    for table_name in db['table_names']:
        table_cells = zip(*cells[table_name]['cell'])
        for column_cells in table_cells:
            cur_values = []
            for cell in set(column_cells):
                if cell.strip():
                    cur_values.append(cell.strip())
            db_cells.append(cur_values)
    return db_cells

split_alpha = lambda str: re.sub(r'[a-z0-9\.]+[十百千万亿%]*', lambda match_obj: " " + match_obj.group(0) + " ", str, flags=re.I)
# deal with metrics that is not ambiguous
split_metric1 = lambda str: re.sub(r'平方千米|平米|千米|千瓦|千克|千卡|kg|km|cm|mm|app', lambda match_obj: " " + match_obj.group(0), str, flags=re.I)
# deal with metrics with length 1, such as 4G, 1.5L, 0.5t, 0.3M, attention that 4w and 3k should not be split
split_metric2 = lambda str: re.sub(r'([^a-z0-9\.]|^)([0-9\.]+)([glmt])([^a-z0-9]|$)', lambda match_obj: ' '.join(match_obj.groups()), str, flags=re.I)
NUMBER_REPLACEMENT = list(zip('０１２３４５６７８９％：．', '0123456789%:.'))

class InputProcessor():

    def __init__(self, encode_method='lgesql', db_dir='data/dusql/db_content.json', db_content=True, bridge=True, **kargs):
        super(InputProcessor, self).__init__()
        self.db_dir = db_dir
        self.db_content = db_content
        # self.nlp = lambda s: list(jieba.cut(s))
        tools = LAC(mode='seg')
        self.nlp = lambda s: tools.run(s)
        if self.db_content:
            self.contents = load_db_contents(self.db_dir)
        self.bridge = bridge # whether extract candidate cell values for each column given the question
        self.graph_processor = GraphProcessor(encode_method)

    def pipeline(self, entry: dict, db: dict, verbose: bool = False):
        """ db should be preprocessed """
        entry = self.preprocess_question(entry, verbose=verbose)
        entry = self.schema_linking(entry, db, verbose=verbose)
        entry = self.graph_processor.process_graph_utils(entry, db)
        return entry

    def preprocess_database(self, db: dict, verbose: bool = False):
        """ Tokenize, lemmatize, lowercase table and column names for each database """
        if not db.get('table_names_original', None):
            db['table_names_original'] = db['table_names']
        if not db.get('column_names_original', None):
            db['column_names_original'] = db['column_names']
        table_toks, table_names = [], []
        for tab in db['table_names']:
            tab = self.nlp(tab)
            table_toks.append(tab)
            table_names.append(" ".join(tab))
        db['table_toks'] = table_toks
        column_toks, column_names = [], []
        for _, col in db['column_names']:
            col = self.nlp(col)
            column_toks.append(col)
            column_names.append(" ".join(col))
        db['column_toks'] = column_toks
        column2table = list(map(lambda x: x[0], db['column_names'])) # from column id to table id
        table2columns = [[] for _ in range(len(table_names))] # from table id to column ids list
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
            db['cells'] = extract_db_cells(self.contents, db)
        if verbose:
            print('Tables:', ', '.join(db['table_names']))
            print('Tokenized:', ', '.join(table_names))
            print('Columns:', ', '.join(list(map(lambda x: x[1], db['column_names']))))
            print('Tokenized:', ', '.join(column_names), '\n')
        return db

    def construct_item_mapping(self, entry: dict):
        question = entry['question']
        # fix some problems, tem_ -> item_
        question = re.sub(r'([^i])tem_', lambda match_obj: match_obj.group(1) + 'item_', question)
        item_mapping, item_mapping_reverse, idx = {}, [], 0
        for match_obj in re.finditer(r'item_[\._a-z0-9]+', question):
            span = match_obj.group(0)
            if span not in item_mapping:
                index = 'item' + str(idx)
                item_mapping[span] = index
                item_mapping_reverse.append(span)
                idx += 1 # increase the num index
        entry['item_mapping'], entry['item_mapping_reverse'] = item_mapping, item_mapping_reverse
        for raw_item in sorted(item_mapping.keys(), key=lambda k: - len(k)):
            question = question.replace(raw_item, item_mapping[raw_item])
        entry['question'] = question
        return entry

    def normalize_question(self, question):
        # quote normalization
        question = quote_normalization(question)
        for raw, new in NUMBER_REPLACEMENT:
            question = question.replace(raw, new)
        # split metrics, add whitespace before, 千米/千克/千瓦/kg/km/cm/mm/app/g/l/t/m
        question = split_metric1(question)
        question = split_metric2(question)
        question = re.sub(r'top(\d+)', lambda match_obj: 'top ' + match_obj.group(1), question, flags=re.I)
        # add whitespace before and after english/number/._-%百万亿 for better tokenization
        question = split_alpha(question)
        question = re.sub(r'\s+', ' ', question)
        return question

    def preprocess_question(self, entry: dict, verbose: bool = False):
        """ Tokenize, lemmatize, lowercase question"""
        # construct item_xxx_xxx dict, use variables item1, item2, item3, ...
        entry = self.construct_item_mapping(entry)
        # LAC tokenize
        question = self.normalize_question(entry['question'])
        toks = self.nlp(question)
        cased_toks = re.sub(r'\s+', ' ', ' '.join(toks)).strip().split(' ')
        # some tokenization errors
        if entry['question_id'] == 'qid017993':
            cased_toks = ['建筑师', '汤姆·梅恩', '比', '巴克里希纳·多西', '多', '多少', '作品']
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
            print('Question:', entry['question'])
            print('Tokenized:', ' '.join(entry['uncased_question_toks']))
            # print('Pos tags:', ' '.join(entry['pos_tags']))
            print('\n')
        return entry

    def schema_linking(self, entry: dict, db: dict, verbose: bool = False):
        """ Perform schema linking: both question and database need to be preprocessed """
        question_toks = entry['uncased_question_toks']
        question = ''.join(question_toks)
        table_toks, column_toks = db['table_toks'], db['column_toks']
        table_names, column_names = db['table_names'], list(map(lambda x: x[1], db['column_names']))
        q_num, dtype = len(question_toks), '<U100'
        t_num, c_num = len(table_toks), len(column_toks)

        def question_schema_matching(schema_toks, schema_names, category):
            assert category in ['table', 'column']
            s_num, matched_pairs = len(schema_names), {'partial': [], 'exact': []}
            q_s_mat = np.array([[f'question-{category}-nomatch'] * s_num for _ in range(q_num)], dtype=dtype)
            s_q_mat = np.array([[f'{category}-question-nomatch'] * q_num for _ in range(s_num)], dtype=dtype)
            for qid, tok in enumerate(question_toks):
                if tok in STOPWORDS: continue
                for sid, schema_tok in enumerate(schema_toks):
                    if tok in schema_tok:
                        match_type = 'partial' if len(schema_tok) > 1 else 'exact'
                        q_s_mat[qid, sid] = f'question-{category}-{match_type}match'
                        s_q_mat[sid, qid] = f'{category}-question-{match_type}match'
                        if verbose:
                            matched_pairs[match_type].append(str((schema_names[sid], sid, tok, qid, qid + 1)))
                        break
            for sid, schema in enumerate(schema_names):
                if len(schema_toks[sid]) == 1 or schema in STOPWORDS: continue
                if schema in question:
                    start_id = question.index(schema)
                    start, end = entry['char2word_id_mapping'][start_id], entry['char2word_id_mapping'][start_id + len(schema) - 1] + 1
                    q_s_mat[range(start, end), sid] = f'question-{category}-exactmatch'
                    s_q_mat[sid, range(start, end)] = f'{category}-question-exactmatch'
                    if verbose:
                        matched_pairs['exact'].append(str((schema, sid, ''.join(question_toks[start: end]), start, end)))
            return q_s_mat, s_q_mat, matched_pairs

        q_tab_mat, tab_q_mat, table_matched_pairs = question_schema_matching(table_toks, table_names, 'table')
        q_col_mat, col_q_mat, column_matched_pairs = question_schema_matching(column_toks, column_names, 'column')


        # relations between questions and tables, q_num*t_num and t_num*q_num
        # table_matched_pairs = {'partial': [], 'exact': []}
        # q_tab_mat = np.array([['question-table-nomatch'] * t_num for _ in range(q_num)], dtype=dtype)
        # tab_q_mat = np.array([['table-question-nomatch'] * q_num for _ in range(t_num)], dtype=dtype)
        # for idx, name in enumerate(table_names):
        #     max_len = len(name)
        #     index_pairs = sorted(filter(lambda x: x[1] - x[0] <= max_len, combinations(range(q_num + 1), 2)), key=lambda x: x[1] - x[0])
        #     for i, j in index_pairs:
        #         phrase = ''.join(question_toks[i: j])
        #         if phrase in STOPWORDS: continue
        #         if phrase == name: # fully match will overwrite partial match due to sort
        #             q_tab_mat[range(i, j), idx] = 'question-table-exactmatch'
        #             tab_q_mat[idx, range(i, j)] = 'table-question-exactmatch'
        #             if verbose:
        #                 table_matched_pairs['exact'].append(str((name, idx, phrase, i, j)))
        #         # elif (j - i == 1 and phrase in table_toks[idx]) or (j - i > 1 and phrase in name):
        #         elif (len(phrase) == 1 and phrase in table_toks[idx]) or (len(phrase) > 1 and phrase in name):
        #             q_tab_mat[range(i, j), idx] = 'question-table-partialmatch'
        #             tab_q_mat[idx, range(i, j)] = 'table-question-partialmatch'
        #             if verbose:
        #                 table_matched_pairs['partial'].append(str((name, idx, phrase, i, j)))

        # relations between questions and columns
        # column_matched_pairs = {'partial': [], 'exact': []}
        # q_col_mat = np.array([['question-column-nomatch'] * c_num for _ in range(q_num)], dtype=dtype)
        # col_q_mat = np.array([['column-question-nomatch'] * q_num for _ in range(c_num)], dtype=dtype)
        # for idx, (_, name) in enumerate(column_names):
        #     max_len = len(name)
        #     index_pairs = sorted(filter(lambda x: x[1] - x[0] <= max_len, combinations(range(q_num + 1), 2)), key=lambda x: x[1] - x[0])
        #     for i, j in index_pairs:
        #         phrase = ''.join(question_toks[i: j])
        #         if phrase in STOPWORDS: continue
        #         if phrase == name: # fully match will overwrite partial match due to sort
        #             q_col_mat[range(i, j), idx] = 'question-column-exactmatch'
        #             col_q_mat[idx, range(i, j)] = 'column-question-exactmatch'
        #             if verbose:
        #                 column_matched_pairs['exact'].append(str((name, idx, phrase, i, j)))
        #         # elif (j - i == 1 and phrase in column_toks[idx]) or (j - i > 1 and phrase in name):
        #         elif (len(phrase) == 1 and phrase in column_toks[idx]) or (len(phrase) > 1 and phrase in name):
        #             q_col_mat[range(i, j), idx] = 'question-column-partialmatch'
        #             col_q_mat[idx, range(i, j)] = 'column-question-partialmatch'
        #             if verbose:
        #                 column_matched_pairs['partial'].append(str((name, idx, phrase, i, j)))

        if self.db_content:
            column_matched_pairs['value'] = []
            # create question-value-match relations, be careful with item_mapping
            def normalize_numbers(num):
                return str(float(num)) if is_number(num) else str(num)

            for i, (_, col_name) in enumerate(db['column_names']):
                if i == 0: # ignore *
                    continue
                cells = db['cells'][i] # list of cell values, ['2014', '2015']
                cells = [normalize_numbers(c).lower() for c in cells]
                for j, word in enumerate(question_toks):
                    norm_word = normalize_numbers(word)
                    word = entry['item_mapping_reverse'][int(word.strip('item'))] if word.startswith('item') else word
                    for c in cells:
                        if (word in c or norm_word in c) and 'nomatch' in q_col_mat[j, i] and word not in STOPWORDS:
                            q_col_mat[j, i] = 'question-column-valuematch'
                            col_q_mat[i, j] = 'column-question-valuematch'
                            if verbose:
                                column_matched_pairs['value'].append(str((col_name, i, c, word, j, j + 1)))
                            break

        # extract candidate cell values for each column given the current question
        if self.bridge:
            cells = [[]] # map column_id to candidate values relevant to the question
            question = ''.join(question_toks)
            for col_id in range(len(db['column_names'])):
                if col_id == 0: continue
                candidates = db['cells'][col_id]
                candidates = [entry['item_mapping'][c] if c in entry['item_mapping'] else str(c).lower() for c in candidates]
                candidates = get_database_matches(question, candidates, db['column_names'][col_id][1], db['column_types'][col_id])
                if candidates:
                    candidates = [self.nlp(c) for c in candidates]
                    candidates = ['='] + sum([re.sub(r'\s+', ' ', ' '.join(toks)).strip().split(' ') + ['，'] for toks in candidates], [])[:-1]
                    cells.append(candidates)
                else: cells.append([])
        else:
            cells = [[] for _ in range(len(db['column_names']))]
        entry['cells'] = cells

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
                print('Value match: (column name, col_id, cell name, question word, start id, end id)')
                print(', '.join(column_matched_pairs['value']) if column_matched_pairs['value'] else 'empty')
            print('\n')
        return entry
