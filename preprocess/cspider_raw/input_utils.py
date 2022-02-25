#coding=utf8
import os, sys, sqlite3, re, string, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import stanza
import numpy as np
from numpy.core.fromnumeric import cumsum
from LAC import LAC
import word2number as w2n
from itertools import product
from fuzzywuzzy import process
from googletrans import Translator
from nltk.corpus import stopwords
from utils.constants import MAX_RELATIVE_DIST, DATASETS
from preprocess.graph_utils import GraphProcessor
from preprocess.process_utils import QUOTATION_MARKS, is_number


def is_word_number(w):
    try:
        num = w2n.word_to_num(w)
        return True
    except: return False

STOPWORDS = set(["的", "是", "，", "？", "有", "多少", "哪些", "我", "什么", "你", "知道", "啊", "次", "些", "一些",
            "一下", "吗", "在", "请问", "或", "或者", "想", "和", "为", "帮", "那个", "你好", "这", "上", "这边", "这些",
            "了", "并且", "都", "呢", "呀", "哪个", "还有", "这个", "-", "项目", "我查", "就是", "所有", "就", "那么",
            "它", "它们", "他", "他们", "要求", "谁", "了解", "告诉", "时候", "个", "能", "那", "人", "问", "中", "列出", "找出", "找到",
            "可以", "一共", "哪", "麻烦", "叫", "想要", "《", "》", "分别", "按", "按照", "过", "为其"])

class CachedTranslator():

    def __init__(self, cached_dir=None) -> None:
        super(CachedTranslator, self).__init__()
        self.translator = Translator(service_urls=['translate.googleapis.com'])
        self.cached_dir = cached_dir if cached_dir is not None else DATASETS['cspider_raw']['cached_dir']
        self.zh2en, self.en2zh = {}, {}
        zh2en_path = os.path.join(self.cached_dir, 'translation.zh2en')
        self.zh2en = json.load(open(zh2en_path, 'r')) if os.path.exists(zh2en_path) else {}
        en2zh_path = os.path.join(cached_dir, 'translation.en2zh')
        self.en2zh = json.load(open(en2zh_path, 'r')) if os.path.exists(en2zh_path) else {}

    def translate(self, query: str, tgt_lang: str = 'en'):
        try:
            if tgt_lang == 'en':
                res = self.zh2en.get(query, None)
                if res is None:
                    res = self.translator.translate(query, dest='en').text.lower()
                    self.zh2en[query] = res
            else:
                res = self.en2zh.get(query, None)
                if res is None:
                    res = self.translator.translate(query, dest='zh-cn').text.lower()
                    self.en2zh[query] = res
            return res
        except: # return itself if RuntimeError occurs during calling API
            print('[ERROR]: calling google translation API failed!')
            return query
    
    def batched_translate(self, queries: list = [], tgt_lang: str = 'en'):
        return [self.translate(q, tgt_lang=tgt_lang) for q in queries]

    def save_translation_memory(self, save_dir: str = None):
        save_dir = self.cached_dir if save_dir is None else save_dir
        with open(os.path.join(save_dir, 'translations.zh2en'), 'w') as of:
            json.dump(self.zh2en, of, indent=4, ensure_ascii=False)
        with open(os.path.join(save_dir, 'translations.en2zh'), 'w') as of:
            json.dump(self.en2zh, of, indent=4, ensure_ascii=False)
        return


class InputProcessor():

    def __init__(self, encode_method='lgesql', db_dir='data/cspider_raw/database', db_content=True, bridge=False, **kargs):
        super(InputProcessor, self).__init__()
        self.db_dir = db_dir
        self.db_content, self.bridge = db_content, bridge
        self.nlp_en = stanza.Pipeline('en', processors='tokenize,pos,lemma')#, use_gpu=False)
        tools = LAC(mode='lac')
        self.nlp_zh = lambda s: tools.run(s)
        self.stopwords_en = set(stopwords.words("english")) - {'no'}
        self.stopwords_zh = set(STOPWORDS)
        self.punctuations = QUOTATION_MARKS + list('，。！￥？（）《》、；·…' + string.punctuation)
        # to reduce the number of API calls
        self.translator = CachedTranslator()
        self.graph_processor = GraphProcessor(encode_method)

    def pipeline(self, entry: dict, db: dict, verbose: bool = False):
        """ db should be preprocessed """
        entry = self.preprocess_question(entry, verbose=verbose)
        entry = self.schema_linking(entry, db, verbose=verbose)
        entry = self.graph_processor.process_graph_utils(entry, db)
        return entry

    def preprocess_database(self, db: dict, verbose: bool = False):
        """ Tokenize, lemmatize, lowercase table and column names for each database """
        table_toks, processed_table_toks, processed_table_names = [], [], []
        for tab in db['table_names']:
            doc = self.nlp_en(tab)
            tab = [w.text.lower() for s in doc.sentences for w in s.words]
            # ptab = [w.lemma.lower() for s in doc.sentences for w in s.words]
            table_toks.append(tab)
            # processed_table_toks.append(ptab)
            # processed_table_names.append(" ".join(ptab))
        db['table_toks'] = table_toks
        # db['processed_table_toks'] = processed_table_toks
        # db['processed_table_names'] = processed_table_names
        
        column_toks, processed_column_toks, processed_column_names = [], [], []
        for _, c in db['column_names']:
            doc = self.nlp_en(c)
            c = [w.text.lower() for s in doc.sentences for w in s.words]
            # pc = [w.lemma.lower() for s in doc.sentences for w in s.words]
            column_toks.append(c)
            # processed_column_toks.append(pc)
            # processed_column_names.append(" ".join(pc))
        db['column_toks'] = column_toks
        # db['processed_column_toks'] = processed_column_toks
        # db['processed_column_names'] =  processed_column_names

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
            # print('Lemmatized:', ', '.join(processed_table_names))
            print('Columns:', ', '.join([' '.join(c) for c in column_toks]))
            # print('Lemmatized:', ', '.join(processed_column_names), '\n')
        return db

    def question_normalization(self, question: str):
        # quote normalization
        for p in QUOTATION_MARKS:
            if p not in ['"', '“', '”']:
                if question.count(p) == 1:
                    question = question.replace(p, '')
                else:
                    question = question.replace(p, '"')
        return question.strip()

    def preprocess_question(self, entry: dict, verbose: bool = False):
        """ Tokenize, lemmatize, lowercase question"""
        # chinese sentence, use LAC tokenize
        question = self.question_normalization(entry['question'])
        # remove all blank symbols
        filtered = filter(lambda x: x[0] not in list(' \t\n\r\f\v'), zip(*self.nlp_zh(question)))
        tok_tag = list(zip(*filtered))
        cased_toks, pos_tags = list(tok_tag[0]), list(tok_tag[1])
        uncased_toks = [t.lower() for t in cased_toks]
        entry['cased_question_toks'] = cased_toks
        entry['uncased_question_toks'] = uncased_toks
        # r, p, c, u, xc, w are tags representing meaningless words
        entry['pos_tags'] = pos_tags # tags and meanings: https://github.com/baidu/lac

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

    def extract_numbers_and_entities(self, question_toks, pos_tags):
        numbers, entities = [], []
        for j, word in enumerate(question_toks):
            if is_number(word):
                entities.append((str(float(word)), (j, j + 1)))
            elif len(word) > 1 and is_number(word[:-1]): # 2个, 3种
                entities.append((str(float(word[:-1])), (j, j + 1)))
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
                elif (not wrapped) and pos_tags[j] in ['nz', 'nw', 'PER', 'LOC', 'ORG', 'TIME']:
                    entities.append(word, (j, j + 1))
                prev_is_bracket = False
        return numbers, entities

    def schema_linking(self, entry: dict, db: dict, verbose: bool = False):
        """ Perform schema linking: both question and database need to be preprocessed """
        cased_question_toks, question_toks = entry['cased_question_toks'], entry['uncased_question_toks']
        question, pos_tags = ''.join(question_toks), entry['pos_tags']
        table_toks, column_toks = db['table_toks'], db['column_toks']
        table_names, column_names = db['table_names'], list(map(lambda x: x[1], db['column_names']))
        q_num, dtype = len(question_toks), '<U100'

        def question_schema_matching(schema_toks, schema_names, category):
            assert category in ['table', 'column']
            s_num, matched_pairs = len(schema_names), {'partial': [], 'exact': []}
            q_s_mat = np.array([[f'question-{category}-nomatch'] * s_num for _ in range(q_num)], dtype=dtype)
            s_q_mat = np.array([[f'{category}-question-nomatch'] * q_num for _ in range(s_num)], dtype=dtype)
            # forward translation, schema items en -> zh
            for idx, toks in enumerate(schema_toks):
                toks_zh = self.translator.batched_translate(toks, tgt_lang='zh')
                if len(toks) > 1:
                    full_schema_zh = self.translator.translate(schema_names[idx], tgt_lang='zh')
                    toks_zh.append(full_schema_zh)
                for j, span in enumerate(toks_zh):
                    span = span.replace(' ', '') # question exclude whitespaces
                    match_type = 'partial' if j < len(toks_zh) - 1 else 'exact'
                    if span in question and span not in self.stopwords_zh:
                        start_id = question.index(span)
                        start, end = entry['char2word_id_mapping'][start_id], entry['char2word_id_mapping'][start_id + len(span) - 1] + 1
                        q_s_mat[range(start, end), idx] = f'question-{category}-{match_type}match'
                        s_q_mat[idx, range(start, end)] = f'{category}-question-{match_type}match'
                        if verbose:
                            matched_pairs[match_type].append(str((span, start, end, schema_names[idx], idx)))
            # backward translation, question_tok zh -> en, increase recall
            toks_en = self.translator.batched_translate(cased_question_toks, tgt_lang='en')
            for start, tok in enumerate(toks_en):
                tok = tok.lower()
                if tok in self.stopwords_en or tok in self.punctuations or pos_tags[start] in ['r', 'p', 'c', 'u', 'xc', 'w']: continue
                for idx, name in enumerate(schema_names):
                    if tok in name and 'exact' not in q_s_mat[start, idx]:
                        match_type = 'partial' if len(schema_toks[idx]) > 1 else 'exact'
                        q_s_mat[start, idx] = f'question-{category}-{match_type}match'
                        s_q_mat[idx, start] = f'{category}-question-{match_type}match'
                        if verbose:
                            matched_pairs[match_type].append(str((tok, start, start + 1, schema_names[idx], idx)))
            return q_s_mat, s_q_mat, matched_pairs
        
        q_tab_mat, tab_q_mat, table_matched_pairs = question_schema_matching(table_toks, table_names, 'table')
        q_col_mat, col_q_mat, column_matched_pairs = question_schema_matching(column_toks, column_names, 'column')

        if self.db_content:
            column_matched_pairs['value'] = []
            db_file = os.path.join(self.db_dir, db['db_id'], db['db_id'] + '.sqlite')
            try:
                conn = sqlite3.connect(db_file)
                conn.text_factory = lambda b: b.decode(errors='ignore')
                conn.execute('pragma foreign_keys=ON')
                numbers, entities = self.extract_numbers_and_entities(question_toks, pos_tags)
                for i, (tab_id, col_name) in enumerate(db['column_names_original']):
                    if i == 0 or 'id' in column_toks[i]: # ignore * and special token 'id'
                        continue
                    tab_name = db['table_names_original'][tab_id]
                    command = "SELECT DISTINCT \"%s\" FROM \"%s\";" % (col_name, tab_name)
                    try:
                        cursor = conn.execute(command)
                        cell_values = cursor.fetchall()
                        cell_values = [str(each[0]) for each in cell_values]
                        if len(cell_values) == 0: continue
                        for cv in cell_values:
                            if re.search(r'[a-zA-Z]', cv):
                                cv = self.translator.translate(cv, tgt_lang='zh')
                            cv = re.sub(r'["“”\'\s]', '', cv)
                            if cv in question and cv not in self.stopwords_zh:
                                start_id = question.index(cv)
                                start, end = entry['char2word_id_mapping'][start_id], entry['char2word_id_mapping'][start_id + len(cv) - 1] + 1
                                for j in range(start, end):
                                    if 'nomatch' in q_col_mat[j, i]:
                                        q_col_mat[j, i] = 'question-column-valuematch'
                                        col_q_mat[i, j] = 'column-question-valuematch'
                                if verbose:
                                    column_matched_pairs['value'].append(str((cv, start, end, column_names[i], i)))
                        # normalize numbers
                        cell_values = [str(float(cv)) if is_number(cv) else cv for cv in cell_values]
                        for num, (start, _) in numbers:
                            if num in cell_values and 'nomatch' in q_col_mat[start, i]:
                                q_col_mat[start, i] = 'question-column-valuematch'
                                col_q_mat[i, start] = 'column-question-valuematch'
                                if verbose:
                                    column_matched_pairs['value'].append(str((cv, start, start + 1, column_names[i], i)))
                        for ent, (start, end) in entities:
                            if re.search(u'[\u4e00-\u9fa5]', ent):
                                ent = self.translator.translate(ent, tgt_lang='en')
                            cv, score = process.extractOne(ent, cell_values)
                            if score >= 85:
                                for j in range(start, end):
                                    if 'nomatch' in q_col_mat[j, i]:
                                        q_col_mat[j, i] = 'question-column-valuematch'
                                        col_q_mat[i, j] = 'column-question-valuematch'
                                if verbose:
                                    column_matched_pairs['value'].append(str((cv, start, end, column_names[i], i)))                          
                    except Exception: print('Error raised while executing SQL:', command)
                conn.close()
            except:
                print('DB file not found:', db_file)

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
            print('Table matched: (question_span, start index, end index, table name, table id)')
            print('Exact match:', ', '.join(table_matched_pairs['exact']) if table_matched_pairs['exact'] else 'empty')
            print('Partial match:', ', '.join(table_matched_pairs['partial']) if table_matched_pairs['partial'] else 'empty')
            print('Column matched: (question_span, start index, end index, column name, column id)')
            print('Exact match:', ', '.join(column_matched_pairs['exact']) if column_matched_pairs['exact'] else 'empty')
            print('Partial match:', ', '.join(column_matched_pairs['partial']) if column_matched_pairs['partial'] else 'empty')
            if self.db_content:
                print('Value match:', ', '.join(column_matched_pairs['value']) if column_matched_pairs['value'] else 'empty')
            print('\n')
        return entry


if __name__ == '__main__':

    processor = InputProcessor()
    data_dir = DATASETS['cspider_raw']['data']
    tables = { db['db_id']: db for db in json.load(open(os.path.join(data_dir, 'tables.json'), 'r')) }
    data = json.load(open(os.path.join(data_dir, 'train.json'), 'r')) + json.load(open(os.path.join(data_dir, 'dev.json'), 'r'))
    for db in tables:
        db = tables[db]
        processor.preprocess_database(db)
    for ex in data:
        ex = processor.preprocess_question(ex)
        ex = processor.schema_linking(ex, tables[ex['db_id']], verbose=True)
        print('')
    processor.translator.save_translation_memory()