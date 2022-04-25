#coding=utf8
import sys, os, re, json, pickle, string
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import stanza
import numpy as np
from nltk.corpus import stopwords
from itertools import combinations
from utils.constants import DATASETS, MAX_RELATIVE_DIST
from preprocess.graph_utils import GraphProcessor
from preprocess.process_utils import is_number, is_int, QUOTATION_MARKS
from preprocess.spider.bridge_content_encoder import get_database_matches


class InputProcessor():

    def __init__(self, encode_method='lgesql', db_dir='data/wikisql', db_content=False, bridge=False, **kargs):
        super(InputProcessor, self).__init__()
        self.db_dir = db_dir
        self.db_content = db_content
        self.nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma')#, use_gpu=False)
        self.stopwords = set(stopwords.words("english")) - {'no'} | set(QUOTATION_MARKS + list('，。！￥？（）《》、；·…' + string.punctuation))
        self.bridge = bridge # whether extract candidate cell values for each column given the question
        self.graph_processor = GraphProcessor(encode_method)
        self.table_pmatch, self.table_ematch = 0, 0
        self.column_pmatch, self.column_ematch, self.column_vmatch = 0, 0, 0
        self.bridge_count = 0

    def pipeline(self, entry: dict, db: dict, verbose: bool = False):
        """ db should be preprocessed """
        entry = self.preprocess_question(entry, verbose=verbose)
        entry = self.schema_linking(entry, db, verbose=verbose)
        entry = self.bridge_content(entry, db)
        entry = self.graph_processor.process_graph_utils(entry, db)
        return entry

    def preprocess_database(self, db: dict, verbose: bool = False):
        """ Tokenize, lemmatize, lowercase table and column names for each database """
        table_toks = [['database']] # only one table, use a special name
        db['table_toks'] = table_toks
        db['processed_table_toks'] = table_toks
        column_toks, processed_column_toks = [], []
        for _, c in db['column_names']:
            doc = self.nlp(c)
            c = [w.text.lower() for s in doc.sentences for w in s.words]
            pc = [w.lemma.lower() for s in doc.sentences for w in s.words]
            column_toks.append(c)
            processed_column_toks.append(pc)
        db['column_toks'] = column_toks
        db['processed_column_toks'] = processed_column_toks
        
        column2table = list(map(lambda x: x[0], db['column_names'])) # from column id to table id
        table2columns = [list(range(len(db['column_names'])))] # from table id to column ids list
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
            col_types = db['column_types']
            db['processed_cells'] = [[re.sub(r'\s+', ' ', c.lower()) if col_types[cid] == 'text' else str(int(float(c))) if is_int(c) else str(float(c)) if is_number(c) else str(c)
                for c in cvs] for cid, cvs in enumerate(db['cells'])]
        if verbose:
            print('Tokenized tables:', ', '.join(['|'.join(tab) for tab in table_toks]))
            print('Tokenized columns:', ', '.join(['|'.join(col) for col in column_toks]), '\n')
        return db

    def preprocess_question(self, entry: dict, verbose: bool = False):
        """ Tokenize, lemmatize, lowercase question"""
        # error fixing
        if entry['question'] == "What is the Gecko value for the item that has a Prince XML value of 'no' and a KHTML value of 'yes'?":
            entry['question'] = "What is the Gecko value for the item that has a Prince XML value of 'yes' and a KHTML value of 'yes'?"
        elif entry['question'] == 'Name the melbourne for adelaide of no with auckland of yes and gold coast of yes':
            entry['question'] = 'Name the melbourne for adelaide of no with auckland of yes and gold coast of no'
        elif entry['question'] == 'Which Adelaide has Perth no, Gold Coast no, and Sydney yes?':
            entry['question'] = 'Which Adelaide has Perth no, Gold Coast yes, and Sydney yes?'
        elif entry['question'] == 'yes or no for the melbourne that has no for adelaide, no for gold coast?':
            entry['question'] = 'yes or no for the melbourne that has yes for adelaide, no for gold coast?'
        elif entry['question'] == 'Which athlete from Germany has 2.20 of O and a 2.25 of O?':
            entry['question'] = 'Which athlete from Germany has 2.20 of o and a 2.25 of o?'
        elif entry['question'] == 'Which 2006 Tournament has a 2004, and a 2002 of not tier i?':
            entry['question'] = 'Which 2006 Tournament has a 2004 of A, and a 2002 of not tier i?'

        # LAC tokenize
        doc = self.nlp(entry['question'])
        cased_toks = [w.text for s in doc.sentences for w in s.words]
        uncased_toks = [w.text.lower() for s in doc.sentences for w in s.words]
        processed_toks = [w.lemma.lower() for s in doc.sentences for w in s.words]
        # pos_tags = [w.xpos for s in doc.sentences for w in s.words]
        if re.search(r'^(.|[\d\.]+)\?$', cased_toks[-1]): # tokenization error
            cased_toks[-1:] = [cased_toks[-1][:-1], '?']
            uncased_toks[-1:] = [uncased_toks[-1][:-1], '?']
            processed_toks[-1:] = [uncased_toks[-1][:-1], '?']
            # pos_tags[-1:] = [pos_tags[-1], '.']
        entry['cased_question_toks'] = cased_toks
        entry['uncased_question_toks'] = uncased_toks
        entry['processed_question_toks'] = processed_toks
        # entry['pos_tags'] = pos_tags

        # relations in questions, q_num * q_num
        q_num, dtype = len(cased_toks), '<U100'
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
        uncased_question_toks, question_toks = entry['uncased_question_toks'], entry['processed_question_toks']
        column_toks, column_names = db['processed_column_toks'], [' '.join(toks) for toks in db['processed_column_toks']]
        q_num, dtype = len(question_toks), '<U100'

        def question_schema_matching_method(schema_toks, schema_names, category):
            assert category in ['table', 'column']
            s_num, matched_pairs = len(schema_names), {'partial': [], 'exact': []}
            q_s_mat = np.array([[f'question-{category}-nomatch'] * s_num for _ in range(q_num)], dtype=dtype)
            s_q_mat = np.array([[f'{category}-question-nomatch'] * q_num for _ in range(s_num)], dtype=dtype)
            max_len = max([len(toks) for toks in schema_toks])
            index_pairs = sorted(filter(lambda x: 0 < x[1] - x[0] <= max_len + 1, combinations(range(q_num + 1), 2)), key=lambda x: x[1] - x[0])
            for sid, name in enumerate(schema_names):
                current_len = len(schema_toks[sid])
                for start, end in index_pairs:
                    if end - start > current_len + 1: break
                    span = ' '.join(question_toks[start:end])
                    if span in self.stopwords: continue
                    if span == name: # fully match will overwrite partial match due to sort
                        q_s_mat[range(start, end), sid] = f'question-{category}-exactmatch'
                        s_q_mat[sid, range(start, end)] = f'{category}-question-exactmatch'
                        if verbose:
                            matched_pairs['exact'].append(str((name, sid, span, start, end)))
                    elif (end - start == 1 and span in schema_toks[sid]) or (end - start > 1 and span in name):
                        q_s_mat[range(start, end), sid] = f'question-{category}-partialmatch'
                        s_q_mat[sid, range(start, end)] = f'{category}-question-partialmatch'
                        if verbose:
                            matched_pairs['partial'].append(str((name, sid, span, start, end)))
            return q_s_mat, s_q_mat, matched_pairs

        q_tab_mat = np.array([['question-table-nomatch'] for _ in range(q_num)], dtype=dtype)
        tab_q_mat = np.array([['table-question-nomatch'] * q_num], dtype=dtype)
        q_col_mat, col_q_mat, column_matched_pairs = question_schema_matching_method(column_toks, column_names, 'column')
        self.column_pmatch += np.sum(q_col_mat == 'question-column-partialmatch')
        self.column_ematch += np.sum(q_col_mat == 'question-column-exactmatch')

        if self.db_content:
            column_matched_pairs['value'] = []
            for qid, word in enumerate(uncased_question_toks):
                if word in self.stopwords: continue
                word = str(int(float(word))) if is_int(word) else str(float(word)) if is_number(word) else str(word)
                for cid, col_name in enumerate(column_names):
                    if 'nomatch' in q_col_mat[qid, cid]:
                        col_cells, col_type = db['processed_cells'][cid], db['column_types'][cid]
                        for c in col_cells:
                            if (word in c and col_type == 'text') or (word == c and col_type == 'real'):
                                q_col_mat[qid, cid] = 'question-column-valuematch'
                                col_q_mat[cid, qid] = 'column-question-valuematch'
                                if verbose:
                                    column_matched_pairs['value'].append(str((col_name, cid, c, word, qid, qid + 1)))
                                break
            self.column_vmatch += np.sum(q_col_mat == 'question-column-valuematch')

        # two symmetric schema linking matrix: q_num x (t_num + c_num), (t_num + c_num) x q_num
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

    def bridge_content(self, entry: dict, db: dict):
        """ Return chosen cell values (at most MAX_CELL_NUM) for each column
        """
        if not (self.bridge and self.db_content):
            entry['cells'] = entry['processed_cells'] = [[] for _ in range(len(db['column_names']))]
            return entry
        col_types, col_names = db['column_types'], list(map(lambda x: x[1], db['column_names_original']))
        matched_cells, matched_processed_cells = [], []
        for cid, col_name in enumerate(col_names):
            if col_types[cid] == 'real':
                matched_cells.append([])
                matched_processed_cells.append([])
                continue
            col_cells = db['cells'][cid]
            candidates = get_database_matches(entry['question'], 'database', 'column', None, cells=col_cells)
            self.bridge_count += len(candidates)
            processed_candidates = []
            if candidates:
                candidates = [self.nlp(c) for c in candidates]
                processed_candidates = ['='] + sum([[w.lemma.lower() for s in c.sentences for w in s.words] + [','] for c in candidates], [])[:-1]
                candidates = ['='] + sum([[w.text.lower() for s in c.sentences for w in s.words] + [','] for c in candidates], [])[:-1]
                matched_cells.append(candidates)
                matched_processed_cells.append(processed_candidates)
            else:
                matched_cells.append([])
                matched_processed_cells.append([])
        entry['cells'], entry['processed_cells'] = matched_cells, matched_processed_cells
        return entry


if __name__ == '__main__':

    data_dir = DATASETS['wikisql']['data']
    dataset = pickle.load(open(os.path.join(data_dir, 'train.lgesql.bin'), 'rb'))
    tables = pickle.load(open(os.path.join(data_dir, 'tables.bin'), 'rb'))
    import time
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('./pretrained_models/electra-large-discriminator')
    processor = InputProcessor(db_content=True, bridge=True)
    bridge_count, col_count, true_bridge_count, val_count = 0, 0, 0, 0
    input_lens = []
    start_time = time.time()
    for jdx, ex in enumerate(dataset):
        if (jdx + 1) % 1000 == 0:
            print('Processing %d-th example ...' % (jdx + 1))
        db = tables[ex['db_id']]
        ex = processor.bridge_content(ex, db)
        col_types, processed_cells = db['column_types'], ex['cells']
        values = [(cond[0], cond[2]) for cond in ex['sql']['conds']]
        for cid, cv in values:
            ct = col_types[cid]
            if ct == 'text':
                val_count += 1
                if len(processed_cells[cid]) > 0:
                    true_bridge_count += 1
        bridge_count += sum([1 for cvs in processed_cells if len(cvs) > 0])
        col_count += col_types.count('text')
        table = [[DATASETS['wikisql']['schema_types']['table']] + t for t in db['table_toks']]
        column = [[DATASETS['wikisql']['schema_types'][db['column_types'][idx]]] + c + processed_cells[idx]
                for idx, c in enumerate(db['column_toks'])]
        toks = sum([ex['uncased_question_toks']] + table + column, [])
        input_len = len([tokenizer.tokenize(w) for w in toks]) + 3 # plus 3, CLS SEP SEP
        input_lens.append(input_len)
    
    print('In total, %d columns with type text, among them %d columns have %d bridge content' % (col_count, bridge_count, processor.bridge_count))
    print('In total, %d columns have text values in SQL, among them %d columns have bridge count' % (val_count, true_bridge_count))
    print('MAX/MIN/AVG input len with PLM is %s/%s/%.2f' % (max(input_lens), min(input_lens), sum(input_lens) / float(len(dataset))))
    print('Cost %.2fs .' % (time.time() - start_time))

# train dataset
# In total, 277248 columns with type text, among them 62319 columns have 67485 bridge content
# In total, 56566 columns have text values in SQL, among them 50676 columns have bridge count
# MAX/MIN/AVG input len with PLM is 262/20/39.24

# dev dataset
# In total, 40584 columns with type text, among them 9120 columns have 9914 bridge content
# In total, 8407 columns have text values in SQL, among them 7537 columns have bridge count
# MAX/MIN/AVG input len with PLM is 153/20/39.07

# test dataset
# In total, 77222 columns with type text, among them 17536 columns have 18996 bridge content
# In total, 16010 columns have text values in SQL, among them 14365 columns have bridge count
# MAX/MIN/AVG input len with PLM is 147/20/39.13