#coding=utf8
import sys, os, re, json, pickle, string
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from LAC import LAC
from rouge import Rouge
from itertools import combinations
from numpy.core.fromnumeric import cumsum
from utils.constants import DATASETS, MAX_CELL_NUM, MAX_RELATIVE_DIST
from preprocess.graph_utils import GraphProcessor
from preprocess.process_utils import float_equal, is_number, is_int, quote_normalization, QUOTATION_MARKS, load_db_contents, extract_db_contents, ZH_NUMBER, ZH_WORD2NUM

REPLACEMENT = dict(zip('０１２３４５６７８９％：．～幺（）：％‘’\'`—', '0123456789%:.~一():%“”""-'))

NORM = lambda s: re.sub(r'[０１２３４５６７８９％：．～幺（）：％‘’\'`—]', lambda c: REPLACEMENT[c.group(0)], s)

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

    def __init__(self, encode_method='lgesql', db_dir='data/nl2sql/db_content.json', db_content=True, bridge=True, **kargs):
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
            cells = self.bridge_content(entry['uncased_question_toks'], db)
            entry['cells'] = [['='] + sum([self.nlp(c) + ['，'] for c in candidates], [])[:-1] if candidates else [] for candidates in cells]
        else: entry['cells'] = [[] for _ in range(len(db['column_names']))]
        entry = self.graph_processor.process_graph_utils(entry, db)
        return entry

    def preprocess_database(self, db: dict, verbose: bool = False):
        """ Tokenize, lemmatize, lowercase table and column names for each database """
        table_toks = [['数据库']] # only one table, use a special name
        # normalize brackets
        column_toks = [self.nlp(NORM(col.lower())) for _, col in db['column_names']]
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
            col_types = db['column_types']
            db['cells'] = extract_db_contents(self.contents, db, strip=False)
            db['processed_cells'] = [[normalize_cell_value(c) if col_types[cid] == 'text' else transform_word_to_number(normalize_cell_value(c))
                for c in filter(lambda x: x.strip().lower() not in ['nan', '-', '--', 'none', 'null', ''], cvs)] for cid, cvs in enumerate(db['cells'])]
        if verbose:
            print('Tokenized tables:', ', '.join(['|'.join(tab) for tab in table_toks]))
            print('Tokenized columns:', ', '.join(['|'.join(col) for col in column_toks]), '\n')
        return db

    def normalize_question(self, entry):
        question = entry['question']
        # quote and numbers normalization
        question = quote_normalization(question)
        question = NORM(question)
        question = re.sub(r'(那个|那儿)', '', question).replace('一平', '每平') # influence value recognition
        entry['question'] = re.sub(r'\s+', ' ', question).strip(' \t\n!&,.:;=?@^_`~，。？！；、…')
        return entry

    def preprocess_question(self, entry: dict, verbose: bool = False):
        """ Tokenize, lemmatize, lowercase question"""
        # LAC tokenize
        question = self.normalize_question(entry)['question'].replace('丰和支行', ' 丰和支行 ').replace('上海联合', ' 上海联合 ')
        cased_toks = re.sub(r'\s+', ' ', ' '.join(self.nlp(question))).split(' ')

        # some tokenizeation error may affect value recognition
        if '威海市文登分局法医' in cased_toks:
            idx = cased_toks.index('威海市文登分局法医')
            cased_toks[idx:idx+1] = ['威海市文登分局', '法医']
        elif '川南幼专语言' in cased_toks:
            idx = cased_toks.index('川南幼专语言')
            cased_toks[idx:idx+1] = ['川南幼专', '语言']
        elif '川南幼专现代汉语' in cased_toks:
            idx = cased_toks.index('川南幼专现代汉语')
            cased_toks[idx:idx+1] = ['川南幼专', '现代汉语']
        elif '南团岛' in cased_toks:
            idx = cased_toks.index('南团岛')
            cased_toks[idx-1:idx+1] = ['市南', '团岛']
        elif '黄浦区华融证券股份有限公司' in cased_toks:
            idx = cased_toks.index('黄浦区华融证券股份有限公司')
            cased_toks[idx:idx+1] = ['黄浦区', '华融证券股份有限公司']
        elif '人民北路鑫威尼斯西' in cased_toks:
            idx = cased_toks.index('人民北路鑫威尼斯西')
            cased_toks[idx:idx+1] = ['人民北路', '鑫威尼斯西']
        elif '鲁迅美术学院视觉传达设计学院' in cased_toks:
            idx = cased_toks.index('鲁迅美术学院视觉传达设计学院')
            cased_toks[idx:idx+1] = ['鲁迅美术学院', '视觉传达设计学院']
        elif '宁阳县伊发机箱' in cased_toks:
            idx = cased_toks.index('宁阳县伊发机箱')
            cased_toks[idx:idx+2] = ['宁阳县', '伊发机箱制造厂']
        elif '广州东海堂' in cased_toks:
            idx = cased_toks.index('广州东海堂')
            cased_toks[idx:idx+1] = ['广州', '东海堂']
        else:
            for idx, tok in enumerate(cased_toks):
                match = re.search(r'^(小学|初中|高中|初一|初二|初三|高一|高二|高三)(数学|语文|英语|美术|音乐|体育|物理|化学|地理|生物|历史|政治)$', tok)
                if match: cased_toks[idx] = match.group(1) + ' ' + match.group(2)
            cased_toks = ' '.join(cased_toks).split(' ')

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
        question_toks, column_toks = entry['uncased_question_toks'], db['column_toks']
        question, column_names = ''.join(question_toks), [''.join(toks) for toks in column_toks]
        q_num, dtype = len(question_toks), '<U100'

        def question_schema_matching_method(schema_toks, schema_names, category):
            assert category in ['table', 'column']
            s_num, matched_pairs = len(schema_names), {'partial': [], 'exact': []}
            q_s_mat = np.array([[f'question-{category}-nomatch'] * s_num for _ in range(q_num)], dtype=dtype)
            s_q_mat = np.array([[f'{category}-question-nomatch'] * q_num for _ in range(s_num)], dtype=dtype)
            max_len = max([len(toks) for toks in schema_toks])
            index_pairs = sorted(filter(lambda x: 0 < x[1] - x[0] <= max_len + 1, combinations(range(q_num + 1), 2)), key=lambda x: x[1] - x[0])
            for sid, name in enumerate(schema_names):
                if category == 'column' and sid == 0: continue
                current_len = len(schema_toks[sid])
                for start, end in index_pairs:
                    if end - start > current_len + 1: break
                    span = ''.join(question_toks[start:end])
                    if span in self.stopwords: continue
                    if (end - start == 1 and span in schema_toks[sid]) or (end - start > 1 and span in name):
                        # tradeoff between precision and recall
                        q_s_mat[range(start, end), sid] = f'question-{category}-partialmatch'
                        s_q_mat[sid, range(start, end)] = f'{category}-question-partialmatch'
                        if verbose:
                            matched_pairs['partial'].append(str((schema_names[sid], sid, span, start, end)))
                # exact match, considering tokenization errors
                idx, name = 0, re.sub(r'\(.*?\)', '', name).strip() # remove context in brackets
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
            num_toks = [transform_word_to_number(w) for w in question_toks]
            for cid, col_name in enumerate(column_names):
                if cid == 0: continue
                col_cells, col_type = db['processed_cells'][cid], db['column_types'][cid]
                for qid, (word, num_word) in enumerate(zip(question_toks, num_toks)):
                    if 'nomatch' in q_col_mat[qid, cid] and word not in self.stopwords:
                        for c in col_cells:
                            if (word in c and col_type == 'text') or (num_word == c and col_type == 'real'):
                                q_col_mat[qid, cid] = 'question-column-valuematch'
                                col_q_mat[cid, qid] = 'column-question-valuematch'
                                if verbose:
                                    column_matched_pairs['value'].append(str((col_name, cid, c, word, qid, qid + 1)))
                                break
            self.column_vmatch += np.sum(q_col_mat == 'question-column-valuematch')

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

    def bridge_content(self, question_toks: list, db: dict):
        """ Return chosen cell values (at most MAX_CELL_NUM) for each column, e.g.
        [ ['数学老师', '英语老师'] , ['100%'] , ['10', '20'] , ... ]
        """
        cells, col_types = db['processed_cells'], db['column_types']
        numbers = extract_numbers_in_question(''.join(question_toks))
        question = ''.join(filter(lambda s: s not in self.stopwords, question_toks))
        candidates = [[]]

        def number_score(c, numbers):
            if (not is_number(c)) or len(numbers) == 0: return 0.
            return max([1. if float_equal(c, r) or float_equal(c, r, 100) else 0. for r in numbers])

        for cid, col_cells in enumerate(cells):
            if cid == 0: continue
            if col_types[cid] == 'text':
                scores = [(c, self.rouge_score(c, question)) for c in col_cells if 0 < len(c) < 80 and c != '.']
                scores = sorted(filter(lambda x: x[1]['rouge-l']['f'] > 0, scores), key=lambda x: - x[1]['rouge-l']['f'])[:MAX_CELL_NUM]
                if len(scores) > 1: # at most two cells but the second one must have high rouge-1 precision
                    scores = scores[:1] + list(filter(lambda x: x[1]['rouge-1']['p'] >= 0.6, scores[1:]))
            else:
                scores = sorted(filter(lambda x: x[1] > 0, [(c, number_score(c, numbers)) for c in col_cells]), key=lambda x: - x[1])[:MAX_CELL_NUM]
                # scores = [] # do not use cell values for real numbers
            candidates.append(list(map(lambda x: str(int(float(x[0]))) if col_types[cid] == 'real' and is_int(x[0]) else x[0], scores)))
        return candidates

def normalize_cell_value(c):
    return re.sub(r'\s+', ' ', NORM(c.strip().lower()))

def extract_numbers_in_question(question):
    candidates = []
    for span in re.finditer(r'([0-9\.点负%s十百千万亿]+)' % (ZH_NUMBER), question):
        s, e = span.start(), span.end()
        word = question[s: e]
        if s > 0 and re.search(r'[\-周/e]', question[s - 1]): continue
        if e < len(question) and re.search(r'[些批部层楼下手共月周日号股线ae]', question[e]): continue
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
    match_obj = re.search(r'[0-9\.点负%s十百千万亿]+' % (ZH_NUMBER), word)
    if match_obj:
        span = match_obj.group(0)
        s, e = match_obj.start(), match_obj.end()
        if not (s > 0 and word[s - 1] == '/') and not (e < len(word) and re.search(r'[些批部层楼下手共月日号股线ae\-/]', word[e])):
            if is_number(span): return str(float(span))
            try:
                parsed_num = ZH_WORD2NUM(word.rstrip('万亿'))
                return str(float(parsed_num))
            except: pass
    return word


if __name__ == '__main__':
    import time
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('./pretrained_models/chinese-macbert-large')
    # check BRIDGE function
    processor = InputProcessor()
    data_dir = DATASETS['nl2sql']['data']
    tables = pickle.load(open(os.path.join(data_dir, 'tables.bin'), 'rb'))
    train, dev = pickle.load(open(os.path.join(data_dir, 'train.lgesql.bin'), 'rb')), pickle.load(open(os.path.join(data_dir, 'dev.lgesql.bin'), 'rb'))
    real_bridge, text_bridge, all_bridge, text_tp, real_tp, text_count, real_count = 0, 0, 0, 0, 0, 0, 0
    input_lens, dataset = [], dev
    start_time = time.time()
    for jdx, ex in enumerate(dataset):
        if (jdx + 1) % 1000 == 0:
            print('Processing %d-th example ...' % (jdx + 1))
        db = tables[ex['db_id']]
        question_toks, col_types = ex['uncased_question_toks'], db['column_types']
        cells = processor.bridge_content(question_toks, db)
        processed_cells = [['='] + sum([processor.nlp(c) + ['，'] for c in candidates], [])[:-1] if candidates else [] for candidates in cells]
        values = [(cond[0], cond[2]) for cond in ex['sql']['conds']]
        for cid, cv in values:
            ct = col_types[cid]
            cv = normalize_cell_value(cv)
            cv = str(int(float(cv))) if ct == 'real' and is_int(cv) else cv
            if cv in cells[cid]:
                if ct == 'text': text_tp += 1
                else: real_tp += 1
            if ct == 'text': text_count += 1
            else: real_count += 1
        real_bridge += sum([len(cv) for cid, cv in enumerate(cells) if col_types[cid] == 'real'])
        text_bridge += sum([len(cv) for cid, cv in enumerate(cells) if col_types[cid] == 'text'])
        all_bridge += sum([len(cv) for cv in cells])

        table = [[DATASETS['nl2sql']['schema_types']['table']] + t for t in db['table_toks']]
        column = [[DATASETS['nl2sql']['schema_types'][db['column_types'][idx]]] + c + processed_cells[idx]
                for idx, c in enumerate(db['column_toks'])]

        if False:
            print(' '.join(question_toks))
            print(ex['query'])
            print('\n'.join([' '.join(db['column_toks'][cid]) + '[%s] ' % (col_types[cid]) + ' '.join(processed_cells[cid]) for cid, _ in values]))
            print('\n')

        toks = sum([question_toks] + table + column, []) # ensure that input_length < 512
        input_len = len([tokenizer.tokenize(w) for w in toks]) + 3 # plus 3, CLS SEP SEP
        input_lens.append(input_len)

    print('In total, true/all real SQL value count: %s/%s' % (real_tp, real_count))
    print('In total, true/all text SQL value count: %s/%s' % (text_tp, text_count))
    print('In total, bridge value count real/text/all %s/%s/%s' % (real_bridge, text_bridge, all_bridge))
    print('MAX/MIN/AVG input len with PLM is %s/%s/%.2f' % (max(input_lens), min(input_lens), sum(input_lens) / float(len(dataset))))
    print('Cost %.2fs .' % (time.time() - start_time))

    # train:
    # In total, true/all real SQL value count: 3591/19867
    # In total, true/all text SQL value count: 45253/47556
    # In total, bridge value count real/text/all 15502/151289/166791
    # MAX/MIN/AVG input len 241/20/62.86

    # dev:
    # In total, true/all real SQL value count: 420/2576
    # In total, true/all text SQL value count: 4496/4723
    # In total, bridge value count real/text/all 1802/15040/16842
    # MAX/MIN/AVG input len with PLM is 209/23/62.69
