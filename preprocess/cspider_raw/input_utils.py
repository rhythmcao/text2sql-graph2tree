#coding=utf8
import os, sys, sqlite3, re, string, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import stanza
from LAC import LAC
from itertools import product
from fuzzywuzzy import process
from easynmt import EasyNMT
from nltk.corpus import stopwords
import numpy as np
from numpy.core.fromnumeric import cumsum
from utils.constants import MAX_RELATIVE_DIST, DATASETS
from preprocess.graph_utils import GraphProcessor
from preprocess.process_utils import QUOTATION_MARKS, quote_normalization, is_number


STOPWORDS = set([
    '的', '是', '和', '什么', '多少', '有', '在', '显示', '列出', '找出', '返回', '中', '哪些', '按', '为', '了', '或', '查找', '找到', '是什么', '它',
    '哪个', '过', '他们', '与', '那些', '叫', '种', '被', '给', '个', '任何', '以上', '给出', '对于', '上', '下', '具有', '谁', '哪', '我', '多少次',
    '以及', '以', '来自', '由', '请', '所在', '且', '所', '提供', '对应', '把', '我们', '其', '都', '从', '但', '并', '哪一个', '哪里', '属于', '又',
    '什么时候', '出', '里', '作为', '以下', '及其', '告诉我', '每一个', '这个', '那个', '这些', '之', '不在', '既', '之前', '以后', '能', '用', '该',
    '在哪里', '于', '这里', '那里', '而且', '它们', '并且', '吗', '他', '找', '使', '那', '这', '某些', '来', '做', '及', '呢', '而', '她', '她们',
    '在那里', '什么地方', '何', '哪所', '那种', '有没有', '请问', '什么样', '称为', '使用', '可以', '之后', '前', '后', '关于', "拥有", '到'
])


class CachedTranslator():

    def __init__(self, cache_folder=None) -> None:
        super(CachedTranslator, self).__init__()
        self.zh2en, self.en2zh = {}, {}
        self.cache_folder = cache_folder if cache_folder is not None else DATASETS['cspider_raw']['cache_folder']
        self.translator = EasyNMT('mbart50_m2m', cache_folder=self.cache_folder, load_translator=True)
        en2zh_path = os.path.join(self.cache_folder, 'translation.en2zh')
        self.en2zh = json.load(open(en2zh_path, 'r')) if os.path.exists(en2zh_path) else {}
        zh2en_path = os.path.join(self.cache_folder, 'translation.zh2en')
        self.zh2en = json.load(open(zh2en_path, 'r')) if os.path.exists(zh2en_path) else {}

    def translate(self, query: str, target_lang: str = 'en'):
        query = query.lower()
        if target_lang == 'en':
            if not re.search(u'[\u4e00-\u9fa5]', query): return query # must has chinese char
            else: # 第1, 2人, 第1个
                match_obj = re.search(r'^[^\d\.]*([\d\.]+)[^\d\.]*$', query)
                if match_obj and is_number(match_obj.group(1)):
                    return match_obj.group(1)
            res = self.zh2en.get(query, None)
            if res is None:
                # res = 'null' # comment the line below to gather all phrases to be translated
                res = self.translator.translate(query, source_lang='zh', target_lang='en').lower()
                self.zh2en[query] = res
        else:
            if not re.search(r'[a-zA-Z]', query): return query
            res = self.en2zh.get(query, None)
            if res is None:
                # res = 'null' # comment the line below to firstly gather all phrases to be translated
                res = self.translator.translate(query, source_lang='en', target_lang='zh').lower()
                self.en2zh[query] = res
        return res

    def batched_translate(self, queries: list = [], target_lang: str = 'en'):
        return [self.translate(q, target_lang=target_lang) for q in queries]

    def save_translation_memory(self, save_dir: str = None):
        save_dir = self.cache_folder if save_dir is None else save_dir
        with open(os.path.join(save_dir, 'translation.zh2en'), 'w') as of:
            json.dump(self.zh2en, of, indent=4, ensure_ascii=False)
        with open(os.path.join(save_dir, 'translation.en2zh'), 'w') as of:
            json.dump(self.en2zh, of, indent=4, ensure_ascii=False)
        return


class InputProcessor():

    def __init__(self, encode_method='lgesql', db_dir='data/cspider_raw/database', db_content=True, bridge=False, **kargs):
        super(InputProcessor, self).__init__()
        self.db_dir = db_dir
        self.db_content, self.bridge = db_content, bridge
        self.nlp_en = stanza.Pipeline('en', processors='tokenize,pos')#, use_gpu=False)
        tools = LAC(mode='seg')
        self.nlp_zh = lambda s: tools.run(s)
        self.stopwords_en = set(stopwords.words("english")) - {'no'}
        self.stopwords_zh = set(STOPWORDS)
        self.punctuations = set(QUOTATION_MARKS + list('，。！￥？（）《》、；·…' + string.punctuation))
        # to reduce the number of API calls
        self.translator = CachedTranslator()
        self.graph_processor = GraphProcessor(encode_method)
        self.table_pmatch, self.table_ematch = 0, 0
        self.column_pmatch, self.column_ematch, self.column_vmatch = 0, 0, 0

    def pipeline(self, entry: dict, db: dict, verbose: bool = False):
        """ db should be preprocessed """
        entry = self.preprocess_question(entry, verbose=verbose)
        entry = self.schema_linking(entry, db, verbose=verbose)
        entry = self.graph_processor.process_graph_utils(entry, db)
        return entry

    def preprocess_database(self, db: dict, verbose: bool = False):
        """ Tokenize, lemmatize, lowercase table and column names for each database """
        table_toks = []
        for tab in db['table_names']:
            doc = self.nlp_en(tab)
            tab = [w.text.lower() for s in doc.sentences for w in s.words]
            table_toks.append(tab)
        db['table_toks'] = table_toks

        column_toks = []
        for _, c in db['column_names']:
            doc = self.nlp_en(c)
            c = [w.text.lower() for s in doc.sentences for w in s.words]
            column_toks.append(c)
        db['column_toks'] = column_toks

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

        relations = np.concatenate([
            np.concatenate([tab_mat, tab_col_mat], axis=1),
            np.concatenate([col_tab_mat, col_mat], axis=1)
        ], axis=0)
        db['relations'] = relations.tolist()

        if verbose:
            print('Tables:', ', '.join([' '.join(t) for t in table_toks]))
            print('Columns:', ', '.join([' '.join(c) for c in column_toks]))
        return db

    def preprocess_question(self, entry: dict, verbose: bool = False):
        """ Tokenize, lemmatize, lowercase question"""
        # chinese sentence, use LAC tokenize
        question = quote_normalization(entry['question'])
        # remove all blank symbols
        question = ' '.join(self.nlp_zh(question)).replace('ACL2014', 'ACL 2014')
        cased_toks = re.sub(r'\s+', ' ', question).split(' ')
        uncased_toks = [t.lower() for t in cased_toks]
        entry['cased_question_toks'] = cased_toks
        entry['uncased_question_toks'] = uncased_toks

        # word index to char id mapping
        entry['char2word_id_mapping'] = [idx for idx, w in enumerate(uncased_toks) for _ in range(len(w))]
        entry['word2char_id_mapping'] = cumsum([0] + [len(w) for w in uncased_toks]).tolist()

        # relations in questions, q_num * q_num
        q_num, dtype = len(uncased_toks), '<U100'
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
        return entry

    def extract_numbers_and_entities(self, question_toks):
        numbers, entities = [], []
        for j, word in enumerate(question_toks):
            if is_number(word):
                numbers.append((str(float(word)), (j, j + 1)))
            else: # 第1个, 3篇, 第4
                match_obj = re.search(r'^[^\d\.]*([\d\.]+)[^\d\.]*$', word)
                if match_obj and is_number(match_obj.group(1)):
                    numbers.append((str(float(match_obj.group(1))), (j, j + 1)))
        start, prev_is_bracket, wrapped = -1, False, False # ooo""xxx""yyy""ooo, ooo"xxx"ooo"yyy"ooo, all extract xxx and yyy
        for j, word in enumerate(question_toks):
            if word in ['"', '“', '”']:
                if not wrapped:
                    wrapped = True
                elif (not prev_is_bracket) and wrapped:
                    entities.append((''.join(question_toks[start:j]), (start, j)))
                    wrapped = False
                prev_is_bracket =True
            else:
                if prev_is_bracket and wrapped:
                    start = j
                prev_is_bracket = False
        return numbers, entities

    def schema_linking(self, entry: dict, db: dict, verbose: bool = False):
        """ Perform schema linking: both question and database need to be preprocessed """
        question_toks = entry['uncased_question_toks']
        table_toks, column_toks = db['table_toks'], db['column_toks']
        table_names, column_names = db['table_names'], list(map(lambda x: x[1], db['column_names']))
        question, q_num, dtype = ''.join(question_toks), len(question_toks), '<U100'

        def question_schema_matching(schema_toks, schema_names, category):
            assert category in ['table', 'column']
            s_num, matched_pairs = len(schema_names), {'partial': [], 'exact': []}
            q_s_mat = np.array([[f'question-{category}-nomatch'] * s_num for _ in range(q_num)], dtype=dtype)
            s_q_mat = np.array([[f'{category}-question-nomatch'] * q_num for _ in range(s_num)], dtype=dtype)
            # forward translation, schema items en -> zh
            for sid, toks in enumerate(schema_toks):
                if category == 'column' and sid == 0: continue
                filtered_toks = [t for t in toks if t not in self.stopwords_en and t not in self.punctuations]
                if len(filtered_toks) == 0: continue
                exact_match_at_end = True if len(toks) == 1 else False
                if len(toks) > 1 and schema_names[sid] not in self.stopwords_en:
                    filtered_toks.append(schema_names[sid])
                    exact_match_at_end = True
                toks_zh = self.translator.batched_translate(filtered_toks, target_lang='zh')
                for j, (span, span_zh) in enumerate(zip(filtered_toks, toks_zh)):
                    span, span_zh = span.replace(' ', ''), span_zh.replace(' ', '') # exclude whitespaces
                    match_type = 'exact' if j == len(toks_zh) - 1 and exact_match_at_end else 'partial'
                    if span_zh in question and span_zh not in self.stopwords_zh:
                        start_id = question.index(span_zh)
                        start, end = entry['char2word_id_mapping'][start_id], entry['char2word_id_mapping'][start_id + len(span_zh) - 1] + 1
                        q_s_mat[range(start, end), sid] = f'question-{category}-{match_type}match'
                        s_q_mat[sid, range(start, end)] = f'{category}-question-{match_type}match'
                        if verbose:
                            matched_pairs[match_type].append(str((schema_names[sid], sid, span_zh, start, end)))
                    elif span in question and span not in self.stopwords_en:
                        start_id = question.index(span)
                        start, end = entry['char2word_id_mapping'][start_id], entry['char2word_id_mapping'][start_id + len(span) - 1] + 1
                        q_s_mat[range(start, end), sid] = f'question-{category}-{match_type}match'
                        s_q_mat[sid, range(start, end)] = f'{category}-question-{match_type}match'
                        if verbose:
                            matched_pairs[match_type].append(str((schema_names[sid], sid, span, start, end)))

            # backward translation, question_tok zh -> en, increase recall
            for qid, tok in enumerate(question_toks):
                if tok in self.stopwords_zh or tok in self.punctuations: continue
                tok_en = self.translator.translate(tok, target_lang='en')
                if tok_en in self.stopwords_en: continue
                for sid, name in enumerate(schema_names):
                    if (tok_en in name or name in tok_en) and 'exact' not in q_s_mat[qid, sid]:
                        match_type = 'exact' if name in tok_en else 'partial'
                        q_s_mat[qid, sid] = f'question-{category}-{match_type}match'
                        s_q_mat[sid, qid] = f'{category}-question-{match_type}match'
                        if verbose:
                            matched_pairs[match_type].append(str((schema_names[sid], sid, tok, qid, qid + 1)))
            return q_s_mat, s_q_mat, matched_pairs

        q_tab_mat, tab_q_mat, table_matched_pairs = question_schema_matching(table_toks, table_names, 'table')
        self.table_pmatch += np.sum(q_tab_mat == 'question-table-partialmatch')
        self.table_ematch += np.sum(q_tab_mat == 'question-table-exactmatch')
        q_col_mat, col_q_mat, column_matched_pairs = question_schema_matching(column_toks, column_names, 'column')
        self.column_pmatch += np.sum(q_col_mat == 'question-column-partialmatch')
        self.column_ematch += np.sum(q_col_mat == 'question-column-exactmatch')

        if self.db_content:
            column_matched_pairs['value'] = []
            db_file = os.path.join(self.db_dir, db['db_id'], db['db_id'] + '.sqlite')
            if not os.path.exists(db_file):
                raise ValueError('DB file not found:', db_file)
            conn = sqlite3.connect(db_file)
            conn.text_factory = lambda b: b.decode(errors='ignore')
            conn.execute('pragma foreign_keys=ON')
            numbers, entities = self.extract_numbers_and_entities(question_toks)
            for cid, (tid, col_name) in enumerate(db['column_names_original']):
                if cid == 0 or 'id' in column_toks[cid]: # ignore * and special token 'id'
                    continue
                tab_name = db['table_names_original'][tid]
                command = "SELECT DISTINCT \"%s\" FROM \"%s\";" % (col_name, tab_name)
                try:
                    cursor = conn.execute(command)
                    cell_values = cursor.fetchall()
                    cell_values = [str(each[0]).strip().lower() for each in cell_values if str(each).strip() != '']
                    if len(cell_values) == 0: continue
                    for cv in cell_values:
                        if cv in self.stopwords_en: continue
                        if cv in question: # need not translate
                            start_id = question.index(cv)
                            start, end = entry['char2word_id_mapping'][start_id], entry['char2word_id_mapping'][start_id + len(cv) - 1] + 1
                            for qid in range(start, end):
                                if 'nomatch' in q_col_mat[qid, cid]:
                                    q_col_mat[qid, cid] = 'question-column-valuematch'
                                    col_q_mat[cid, qid] = 'column-question-valuematch'
                                    if verbose:
                                        column_matched_pairs['value'].append(str((cv, start, end, column_names[cid], cid)))
                            continue
                    # normalize numbers
                    cell_values = [str(float(cv)) if is_number(cv) else cv for cv in cell_values]
                    for num, (qid, _) in numbers:
                        if num in cell_values and 'nomatch' in q_col_mat[qid, cid]:
                            q_col_mat[qid, cid] = 'question-column-valuematch'
                            col_q_mat[cid, qid] = 'column-question-valuematch'
                            if verbose:
                                column_matched_pairs['value'].append(str((column_names[cid], cid, num, qid, qid + 1)))
                    for ent, (start, end) in entities:
                        ent = self.translator.translate(ent, target_lang='en')
                        cv, score = process.extractOne(ent, cell_values)
                        if score >= 95:
                            for qid in range(start, end):
                                if 'nomatch' in q_col_mat[qid, cid]:
                                    q_col_mat[qid, cid] = 'question-column-valuematch'
                                    col_q_mat[cid, qid] = 'column-question-valuematch'
                            if verbose:
                                column_matched_pairs['value'].append(str((column_names[cid], cid, ent, start, end)))
                except Exception: print('Error raised while executing SQL:', command)
            conn.close()
            self.column_vmatch += np.sum(q_col_mat == 'question-column-valuematch')

        # no bridge
        cells = [[] for _ in range(len(db['column_names']))]
        entry['cells'] = cells

        # two symmetric schema linking matrix: q_num x (t_num + c_num), (t_num + c_num) x q_num
        q_col_mat[:, 0] = 'question-column-nomatch'
        col_q_mat[0] = 'column-question-nomatch'
        q_schema = np.concatenate([q_tab_mat, q_col_mat], axis=1)
        schema_q = np.concatenate([tab_q_mat, col_q_mat], axis=0)
        entry['schema_linking'] = (q_schema.tolist(), schema_q.tolist())

        if verbose:
            print('Question:', ' '.join(question_toks))
            print('Table matched: (table name, table id, question_span, start index, end index)')
            print('Exact match:', ', '.join(table_matched_pairs['exact']) if table_matched_pairs['exact'] else 'empty')
            print('Partial match:', ', '.join(table_matched_pairs['partial']) if table_matched_pairs['partial'] else 'empty')
            print('Column matched: (column name, column id, question_span, start index, end index)')
            print('Exact match:', ', '.join(column_matched_pairs['exact']) if column_matched_pairs['exact'] else 'empty')
            print('Partial match:', ', '.join(column_matched_pairs['partial']) if column_matched_pairs['partial'] else 'empty')
            if self.db_content:
                print('Value match:', ', '.join(column_matched_pairs['value']) if column_matched_pairs['value'] else 'empty')
            print('\n')
        return entry


if __name__ == '__main__':

    # import pickle
    # processor = InputProcessor()
    # data_dir = DATASETS['cspider_raw']['data']
    # tables = { db['db_id']: db for db in json.load(open(os.path.join(data_dir, 'tables.json'), 'r')) }
    # for db in tables:
    #     db = tables[db]
    #     tables[db['db_id']] = processor.preprocess_database(db)
    # pickle.dump(tables, open(os.path.join(DATASETS['cspider_raw']['data'], 'tables.bin'), 'wb'))
    # tables = pickle.load(open(os.path.join(DATASETS['cspider_raw']['data'], 'tables.bin'), 'rb'))
    # data = json.load(open(os.path.join(data_dir, 'train.json'), 'r')) + json.load(open(os.path.join(data_dir, 'dev.json'), 'r'))
    # for ex in data:
    #     ex = processor.preprocess_question(ex)
    #     ex = processor.schema_linking(ex, tables[ex['db_id']], verbose=False)
    # processor.translator.save_translation_memory()

    zh2en_path, en2zh_path = os.path.join(DATASETS['cspider_raw']['cache_folder'], 'translation.zh2en'), os.path.join(DATASETS['cspider_raw']['cache_folder'], 'translation.en2zh')
    zh2en, en2zh = json.load(open(zh2en_path, 'r')), json.load(open(en2zh_path, 'r'))

    import torch
    model_name = 'mbart50_m2m'
    translator = EasyNMT(model_name, cache_folder='./pretrained_models', load_translator=True)
    zh_sent = list(zh2en.keys())
    en_sent = translator.translate(zh_sent, source_lang='zh', target_lang='en', batch_size=50, show_progress_bar=False)
    en_sent = [s.lower() for s in en_sent]
    zh2en = dict(zip(zh_sent, en_sent))
    json.dump(zh2en, open(zh2en_path, 'w'), indent=4, ensure_ascii=False)
    print('Finishing translating zh -> en: %s' % (len(zh2en)))
    torch.cuda.empty_cache()

    en_sent = list(en2zh.keys())
    zh_sent = translator.translate(en_sent, source_lang='en', target_lang='zh', batch_size=50, show_progress_bar=False)
    zh_sent = [s.lower() for s in zh_sent]
    en2zh = dict(zip(en_sent, zh_sent))
    json.dump(en2zh, open(en2zh_path, 'w'), indent=4, ensure_ascii=False)
    print('Finishing translating en -> zh: %s' % (len(en2zh)))
