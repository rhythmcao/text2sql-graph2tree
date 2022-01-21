#coding=utf8
import re, json, os, sys, copy, math, pickle, datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import editdistance as edt
import cn2an, traceback
import jionlp as jio
from decimal import Decimal
from fuzzywuzzy import process
from itertools import combinations
from asdl.transition_system import SelectValueAction
from utils.constants import DATASETS
from preprocess.process_utils import is_number, is_int, ValueCandidate, State, SQLValue
from preprocess.dusql.input_utils import quote_normalization, load_db_contents, extract_db_cells

SYNONYM_SETS = [
    set(['杜,郭', '姓杜和姓郭']),
    set(['汉语', '中文']),
    set(['aaaa级', '4a级']),
    set(['aaaaa级', '5a级']),
    set(['扌', '提手旁']),
    set(['第28届台湾金曲奖', '第28届']),
    set(['北京电影学院', '北电']),
    set(['诺贝尔化学奖', '化学奖']),
    set(['诺贝尔文学奖', '文学奖']),
    set(['第二届中国国际进口博览会', '第二届进博会']),
    set(['第一届中国国际进口博览会', '第一届']),
    set(['韩国最大的汽车企业', '韩国最大的汽车公司']),
    set(['皇帝老师', '皇上的老师']),
    set(['食草', '吃草']),
    set(['世界一流大学', '世界一流']),
    set(['金融与会计专业', '"金融会计"专业']),
    set(['ios系统', 'ios操作系统']),
    set(['"1+2"是哥德巴赫猜想研究的丰碑', '哥德巴赫猜想']),
    set(['微软游戏工作制', '微软游戏']),
    set(['北京世界园艺博览会', '北京园博会']),
    set(['2017世博会', '2017届']),
    set(['穿过爱情的漫长旅程:萧红传', '萧红传'])
]

def float_equal(val1, val2, base=1):
    val1, val2 = float(val1), float(val2)
    if math.fabs(val1 - val2) < 1e-6: return True
    elif math.fabs(val1 * base - val2) < 1e-6 or math.fabs(val1 - val2 * base) < 1e-6: return True
    return False

AGG_OP = ('none', 'max', 'min', 'count', 'sum', 'avg')
CMP_OP = ('not_in', 'between', '==', '>', '<', '>=', '<=', '!=', 'in', 'like')
UNIT_OP = ('none', '-', '+', "*", '/')
PLACEHOLDER = '@' # @ never appears in the dataset
CN_NUMBER_1 = '零一二三四五六七八九'
CN_NUMBER_2 = '〇壹贰叁肆伍陆柒捌玖'
CN_TWO_ALIAS = '两'
CN_UNIT = '十拾百佰千仟万萬亿億兆点．'
RESERVE_CHARS = CN_NUMBER_1 + CN_NUMBER_2 + CN_TWO_ALIAS + CN_UNIT
NUM_ALIAS = lambda num: [CN_NUMBER_1[num]] + [CN_NUMBER_2[num]] if num != 2 else \
    [CN_TWO_ALIAS] + [CN_NUMBER_1[num]] + [CN_NUMBER_2[num]]
word2num = lambda s: cn2an.cn2an(s, mode='smart')
num2word = lambda s, m: cn2an.an2cn(s, m)

def compare_and_extract_date(t1, t2):
    y1, m1, d1 = t1.split('-')
    y2, m2, d2 = t2.split('-')
    if y1 == y2 and m1 == m2:
        return y1 + '-' + m1
    if m1 == m2 and d1 == d2:
        return m1 + '-' + d1
    if y1 == y2: return y1
    if m1 == m2: return m1
    if d1 == d2: return d1
    return None

def compare_and_extract_time(t1, t2):
    h1, m1, s1 = t1.split(':')
    h2, m2, s2 = t2.split(':')
    if h1 == h2 and m1 == m2:
        return h1.lstrip('0') + ':' + m1
        # return h1 + ':' + m1 + ':00'
    if h1 == h2:
        return h1.lstrip('0') + ':00'
        # return h1 + ':00:00'
    if m1 == m2 and s1 == s2:
        return m1.lstrip('0') + ':' + s1
    if m1 == m2:
        return m1.lstrip('0') + ':00'
    return None

def extract_time(result, val):
    # extract date or time string from jio parsed result
    tt = result['type']
    obj = result['time']
    if tt in ['time_span', 'time_point']: # compare list
        t1, t2 = obj[0], obj[1]
        t1_date, t1_time = t1.split(' ')
        t2_date, t2_time = t2.split(' ')
        if t1_time == '00:00:00' and t2_time == '23:59:59': # ignore time
            if t1_date == t2_date:
                y = str(datetime.datetime.today().year)
                if y == t1_date[:4]: # ignore current year prepended
                    return t1_date[5:]
                return t1_date
            else: # find the common part
                return compare_and_extract_date(t1_date, t2_date)
        else: # ignore date
            if '到' not in val:
                return compare_and_extract_time(t1_time, t2_time)
            else:
                return '-'.join(t1_time, t2_time)
    elif tt == 'time_delta':
        v = [obj[k] for k in ['year', 'month', 'day', 'hour'] if k in obj]
        if len(v) > 0:
            return v[0] # only use one value
    return None

class ValueProcessor():

    def __init__(self, table_path, db_dir) -> None:
        self.db_dir = db_dir
        if type(table_path) == str and os.path.splitext(table_path)[-1] == '.json':
            tables_list = json.load(open(table_path, 'r'))
            self.tables = {}
            for db in tables_list:
                self.tables[db['db_id']] = db
        elif type(table_path) == str:
            self.tables = pickle.load(open(table_path, 'rb'))
        else:
            self.tables = table_path
        self.contents = self._load_db_cells()

    def _load_db_cells(self):
        contents = load_db_contents(self.db_dir)
        self.contents = {}
        for db_id in self.tables:
            db = self.tables[db_id]
            if 'cells' in db:
                self.contents[db_id] = db['cells']
            self.contents[db_id] = extract_db_cells(contents, db)
        return self.contents

    def postprocess_value(self, value_id, value_candidates, db, state, entry):
        """ Retrieve DB cell values for reference
        @params:
            value_id: int for SelectAction
            value_candidates: list of ValueCandidate for matched value
            db: database schema dict
            state: namedtuple of (track, agg_op, cmp_op, unit_op, column_id/TIME_NOW)
        @return: value_str
        """
        raw_value = SelectValueAction.reserved_dusql.id2word[value_id] if value_id < SelectValueAction.size('dusql') else \
            value_candidates[value_id - SelectValueAction.size('dusql')].matched_cased_value
        value = self.obtain_possible_value(raw_value, db, state, entry)
        return value

    def obtain_possible_value(self, raw_value, db, state, entry):
        value = ''
        clause, col_id = state.track.split('->')[-1], state.col_id
        if col_id == 'TIME_NOW':
            col_name, col_type, cell_values = col_id, 'time', []
        else:
            col_name = db['column_names_original'][col_id][1]
            col_type = db['column_types'][col_id]
            cell_values = self.contents[db['db_id']][col_id]

        def word_to_number(val, name, force=False):
            if re.search(r'item', val): return False
            if not force and (':' in val or '/' in val or '_' in val \
                or re.search(r'(上午|下午|晚上|时|点|分|秒|月|日|号)', val) or val.count('-') > 1 or val.count('.') > 1):
                return False
            if '-' in val and val.count('-') == 1:
                split_val = val.split('-')
                split_val = [v for v in split_val if v.strip()]
                # determine negative or between x and y
                val = split_val if len(split_val) > 1 else [val]
            elif '~' in val and val.count('~') == 1:
                val = val.split('~')
            elif '到' in val and val.count('到') == 1:
                val = val.split('到')
            else:
                val = [val]
            values = []
            for v in val:
                percent = False
                if v.startswith('百分之'):
                    percent = True
                    v = v[3:]
                if v.endswith('%'):
                    percent = True
                    v = v.rstrip('%')
                height = '米' in v and '千米' not in v and '平米' not in v
                v = v.replace('k', '千').replace('w', '万').replace('米', '点')
                v = re.sub(r'[^0-9\.\-{}]'.format(RESERVE_CHARS), '', v)
                if is_number(v):
                    v = float(v)
                else:
                    try:
                        v = word2num(v)
                    except: return False
                v = float(Decimal(str(v)) * Decimal('100')) if height else float(Decimal(str(v)) * Decimal('0.01')) if percent else float(v)
                v = int(v) if is_int(v) else float(v)
                if '(' in name:
                    # determine the metric in the column name wrapped by (万亿)
                    metric = re.search(r'\((.*?)\)', name).group(1)
                    if '亿' in metric or '万' in metric:
                        factor = 1e8 if '亿' in metric else 1e4
                        if is_int(v) and v % factor == 0:
                            v //= factor
                values.append(str(v))
            nonlocal value
            value = '-'.join(values)
            return True

        def word_to_time(val):
            nonlocal value
            try:
                result = jio.parse_time(val)
                v = extract_time(result, val)
                if v is None:
                    return False
                else:
                    value = v
                return True
            except: return False

        def try_mapping_items(val):
            nonlocal value
            if 'item' in val:
                idx = re.sub(r'[^0-9]', '', val[val.index('item') + 4:])
                if not is_int(idx): return False
                idx = int(idx)
                if 0 <= idx < len(entry['item_mapping_reverse']):
                    value = entry['item_mapping_reverse'][idx]
                    return True
            return False

        if clause == 'limit': # value should be integers
            if is_number(raw_value):
                value = int(float(raw_value))
            elif word_to_number(raw_value, col_name, force=True):
                value = int(float(value))
            else: value = 1 # for all other cases, use 1
        elif clause == 'having': # value should be numbers
            if is_number(raw_value):
                value = int(float(raw_value)) if is_int(raw_value) else float(raw_value)
            elif word_to_number(raw_value, col_name, force=True): pass
            else: value = 1
        else: # WHERE clause

            def try_binary_values(val):
                nonlocal value
                val = '否' if is_number(val) and float_equal(0, val) else '是'
                if '是' in cell_values or '否' in cell_values:
                    value = val if val in ['是', '否'] else '是'
                    return True
                elif '有' in cell_values or '无' in cell_values:
                    value = {'是': '有', '否': '无'}[val] if val in ['是', '否'] else '有'
                    return True
                return False

            if col_type == 'number':
                if word_to_number(raw_value, col_name, force=True): pass
                elif try_mapping_items(raw_value): pass
                else: value = 1
            elif col_type == 'time':
                if word_to_number(raw_value, col_name, force=False): pass
                elif word_to_time(raw_value): pass
                else: value = 1
            elif col_type == 'binary' and try_binary_values(raw_value): pass
            else:
                like_op = 'like' in state.cmp_op
                if len(cell_values) > 0 and not like_op:
                    most_similar = process.extractOne(raw_value, cell_values)
                    value = most_similar[0] if most_similar[1] > 50 else raw_value
                else: value = raw_value
        return str(value) if is_number(value) else "'" + value + "'"

    def extract_values(self, entry: dict, db: dict, verbose=False):
        """ Extract values(class SQLValue) which will be used in AST construction,
        we need question_toks for comparison, entry for char/word idx mapping.
        The matched_index field in the ValueCandidate stores the index_pair of question_toks, not question.
        Because BIO labeling is performed at word-level instead of char-level.
        `track` is used to record the traverse clause path.
        """
        try:
            value_set = set()
            question_toks = copy.deepcopy(entry['uncased_question_toks'])
            sqlvalues, question_toks = self.extract_values_from_sql(entry['sql'], value_set, question_toks, entry, track='')
            entry = self.assign_values(entry, sqlvalues)
            if verbose and len(entry['values']) > 0:
                print('Question:', ' '.join(entry['uncased_question_toks']))
                print('SQL:', entry['query'])
                print('Values:', ' ; '.join([repr(val) for val in entry['values']]), '\n')
        except Exception as e:
            print('ID:', entry['question_id'])
            print('Question:', ' '.join(entry['uncased_question_toks']))
            print('SQL:', entry['query'])
            print('sql:', json.dumps(entry['sql'], ensure_ascii=False))
            print(e)
            exc_type, exc_value, exc_traceback_obj = sys.exc_info()
            traceback.print_tb(exc_traceback_obj)
            exit(1)
        return entry

    def assign_values(self, entry, values):
        # take set because some SQLValue may use the same ValueCandidate
        cased_question_toks = entry['cased_question_toks']
        candidates = set([val.candidate for val in values if isinstance(val.candidate, ValueCandidate)])
        candidates = sorted(candidates)
        offset = SelectValueAction.size('dusql')
        for idx, val in enumerate(candidates):
            val.set_value_id(idx + offset)
            cased_value = ''.join(cased_question_toks[val.matched_index[0]:val.matched_index[1]])
            val.set_matched_cased_value(cased_value)
        entry['values'], entry['candidates'] = values, candidates
        return entry

    def use_extracted_values(self, sqlvalue, values):
        """ When value can not be found in the question,
        the last chance is to resort to extracted values for reference
        """
        old_candidates = set([v.candidate for v in values if v.real_value == sqlvalue.real_value and v.candidate is not None])
        if len(old_candidates) > 0:
            if len(old_candidates) > 1:
                print('Multiple candidate values which can be used for %s ...' % (sqlvalue.real_value))
            sqlvalue.add_candidate(list(old_candidates)[0])
            return True
        return False

    def extract_values_from_sql(self, sql: dict, values: set, question_toks: list, entry: dict, track: str = ''):
        values, question_toks = self.extract_values_from_sql_unit(sql, values, question_toks, entry, track)
        if sql['intersect']:
            values, question_toks = self.extract_values_from_sql_unit(sql['intersect'], values, question_toks, entry, track + '->intersect')
        if sql['union']:
            values, question_toks = self.extract_values_from_sql_unit(sql['union'], values, question_toks, entry, track + '->union')
        if sql['except']:
            values, question_toks = self.extract_values_from_sql_unit(sql['except'], values, question_toks, entry, track + '->except')
        return values, question_toks

    def extract_values_from_sql_unit(self, sql: dict, values: set, question_toks: list, entry: dict, track: str = ''):
        """ Each entry in values is an object of type SQLValue
        """
        table_units = sql['from']['table_units']
        for t in table_units:
            if t[0] == 'sql':
                values, question_toks = self.extract_values_from_sql(t[1], values, question_toks, entry, track + '->from')
        if sql['where']:
            for cond in sql['where']:
                if cond in ['and', 'or']: continue
                values, question_toks = self.extract_values_from_cond(cond, values, question_toks, entry, track + '->where')
        if sql['having']:
            for cond in sql['having']:
                if cond in ['and', 'or']: continue
                values, question_toks = self.extract_values_from_cond(cond, values, question_toks, entry, track + '->having')
        if sql['limit']:
            values, question_toks = self.extract_limit_val(sql, values, question_toks, entry, track + '->limit')
        return values, question_toks

    def extract_values_from_cond(self, cond, values, question_toks, entry, track):
        agg_op, cmp_op, val_unit, val1, _ = cond
        unit_op, col_id = val_unit[0], val_unit[1][1]
        state = State(track, AGG_OP[agg_op], CMP_OP[cmp_op], UNIT_OP[unit_op], col_id)
        func = self.extract_where_val if track.split('->')[-1] == 'where' else self.extract_having_val

        def extract_val(val):
            if type(val) in [list, tuple]: # value is column, do nothing
                return values, question_toks
            elif type(val) == dict: # value is SQL
                return self.extract_values_from_sql(val, values, question_toks, entry, track)
            else:
                return func(val, values, question_toks, entry, state)

        return extract_val(val1)

    def extract_where_val(self, val, *args):
        func = {int: self.extract_where_int_val, str: self.extract_where_string_val, float: self.extract_where_float_val}
        return func[type(val)](val, *args)

    def extract_where_int_val(self, val, values, question_toks, entry, state):
        sqlvalue = SQLValue(str(val), state)
        if sqlvalue in values: return values, question_toks
        else: values.add(sqlvalue)
        val, question = int(val), ''.join(question_toks)

        def try_num2char(val):
            for char in NUM_ALIAS(val):
                if char in question and question.count(char) == 1:
                    # actually no cases where question.count(char) > 1
                    char_id = question.index(char)
                    add_value_from_char_idx((char_id, char_id + 1), question_toks, sqlvalue, entry)
                    return True
            return False

        def try_num2word(val):
            # try to parse each word or word pairs in question_toks into number
            valid_toks = []
            index_pairs = filter(lambda x: x[1] - x[0] in [1, 2], combinations(range(len(question_toks) + 1), 2))
            for i, j in sorted(index_pairs, key=lambda x: x[1] - x[0]):
                try:
                    w = ''.join(question_toks[i: j])
                    num = word2num(w.replace('个', '')) # 一个亿 -> 一亿
                    if num == val:
                        valid_toks.append((i, j))
                except Exception as e: pass
            if len(valid_toks) > 0:
                start, end = valid_toks[0]
                add_value_from_token_idx((start, end), question_toks, sqlvalue)
                return True
            # try to parse the sql value to chinese words and check if it appears in the question
            chars = list(zip('零一二三四五六七八九', '0123456789'))
            for mode in ['low', 'direct']:
                word = num2word(val, mode)
                if word in question_toks and question_toks.count(word) == 1:
                    start = question_toks.index(word)
                    add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
                    return True
                elif word in question and question.count(word) == 1:
                    start_id = question.index(word)
                    add_value_from_char_idx((start_id, start_id + len(word)), question_toks, sqlvalue, entry)
                    return True
                else:
                    for old, new in chars:
                        word = word.replace(old, new) # 7亿8千万
                    if not is_number(word) and word in question and question.count(word) == 1:
                        # only deal with mixture of digits and chars
                        start_id = question.index(word)
                        add_value_from_char_idx((start_id, start_id + len(word)), question_toks, sqlvalue, entry)
                        return True
            # the 3rd cases: cn2an fails to parse 8.5万亿 and 4k, 3w
            valid_toks = []
            for idx, tok in enumerate(question_toks):
                match_obj = re.search(r'([0-9\.]+)([kw千万亿]+)', tok)
                if match_obj:
                    digits, chars = match_obj.groups()
                    try:
                        digits = float(digits)
                        chars = 1e3 ** chars.count('k') * 1e3 ** chars.count('千') * 1e4 ** chars.count('万') * 1e4 ** chars.count('w') * 1e8 ** chars.count('亿')
                        num = int(digits * chars)
                        if num == val:
                            valid_toks.append(idx)
                    except: pass
            if len(valid_toks) == 1:
                start = valid_toks[0]
                add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
                return True
            return False

        def try_parsing_percentage(val):
            for idx, w in enumerate(question_toks):
                w_norm = w.rstrip('%')
                if is_number(w_norm) and float_equal(val, w_norm, base=100) and question_toks.count(w) == 1:
                    prefix = ''.join(question_toks[:idx])
                    if prefix.endswith('百分之'):
                        char_id = entry['word2char_id_mapping'][idx]
                        start_char_id = char_id - 3
                        start_idx = entry['char2word_id_mapping'][start_char_id]
                    else: start_idx = idx
                    add_value_from_token_idx((start_idx, idx + 1), question_toks, sqlvalue)
                    return True
            return False

        def try_possible_suffix(val):
            # due to metrics such as km/km^2, there may be missing or redundant 0s
            valid_toks, val = [], str(val)
            index_pairs = filter(lambda x: x[1] - x[0] in [1, 2], combinations(range(len(question_toks) + 1), 2))
            for i, j in sorted(index_pairs, key=lambda x: x[1] - x[0]):
                try:
                    w = ''.join(question_toks[i: j])
                    num = str(word2num(w))
                    if val.startswith(num) and val == num + '0' * (len(val) - len(num)):
                        valid_toks.append((i, j, len(val) - len(num)))
                    if num.startswith(val) and num == val + '0' * (len(num) - len(val)):
                        valid_toks.append((i, j, len(num) - len(val)))
                except: pass
            if len(valid_toks) == 1:
                start, end = valid_toks[0][0], valid_toks[0][1]
                add_value_from_token_idx((start, end), question_toks, sqlvalue)
                return True
            elif len(valid_toks) == 0: return False
            # resolve ambiguity by choosing the one with smallest 0 difference
            smallest = min(valid_toks, key=lambda x: x[2])
            smallest = [i for i in valid_toks if i[2] == smallest[2]]
            if len(smallest) == 1:
                start, end = smallest[0][0], smallest[0][1]
                add_value_from_token_idx((start, end), question_toks, sqlvalue)
                return True
            smallest = [i for i in smallest if i[1] <= len(question_toks) - 1 and question_toks[i[1]] in ['公里', '千米']]
            if len(smallest) == 1:
                start, end = smallest[0][0], smallest[0][1]
                add_value_from_token_idx((start, end), question_toks, sqlvalue)
                return True
            return False

        def try_parsing_year(val):
            for idx, tok in enumerate(question_toks):
                if is_int(tok) and ((int(tok) < 30 and int(tok) + 2000 == val) or (int(tok) >= 30 and int(tok) + 1900 == val)):
                    # one sample with ambiguity, ignore by directly using the first one
                    add_value_from_token_idx((idx, idx + 1), question_toks, sqlvalue)
                    return True
            return False

        def try_parsing_height(val):
            match_objs = re.finditer(r'[12一二两]米[1-9一二三四五六七八九]+', question)
            valid_spans = []
            for match_obj in match_objs:
                matched_span = match_obj.group(0).replace('米', '点')
                try:
                    num = int(word2num(matched_span) * 100) # must be int, thus times 100
                    if val == num:
                        valid_spans.append((match_obj.start(), match_obj.end()))
                except: pass
            if len(valid_spans) == 1:
                start_id, end_id = valid_spans[0]
                add_value_from_char_idx((start_id, end_id), question_toks, sqlvalue, entry)
                return True
            return False

        if str(val) in question_toks:
            val = str(val)
            if question_toks.count(val) == 1:
                index = question_toks.index(val)
                add_value_from_token_idx((index, index + 1), question_toks, sqlvalue)
            else: # resolve ambiguity
                start = [i for i in range(len(question_toks)) if question_toks[i] == val and \
                    (i == 0 or question_toks[i - 1] not in ['哪', '前', '后', '至少']) and \
                        (i < 2 or '最' not in question_toks[i - 2])]
                add_value_from_token_idx((start[0], start[0] + 1), question_toks, sqlvalue)
        elif val < 10 and try_num2char(val): pass
        elif try_num2word(val): pass
        elif try_parsing_percentage(val): pass
        elif try_possible_suffix(val): pass
        elif try_parsing_height(val): pass
        elif try_parsing_year(val): pass
        else:
            raise ValueError('WHERE int value %d is not recognized in the question: %s' % (val, ' '.join(entry['uncased_question_toks'])))
        return values, question_toks

    def extract_where_float_val(self, val, values, question_toks, entry, state):
        sqlvalue = SQLValue(str(val), state)
        if sqlvalue in values: return values, question_toks
        else: values.add(sqlvalue)
        question = ''.join(question_toks)

        def try_num2char(val):
            for char in NUM_ALIAS(val):
                if char in question and question.count(char) == 1:
                    char_id = question.index(char)
                    add_value_from_char_idx((char_id, char_id + 1), question_toks, sqlvalue, entry)
                    return True
            return False

        def try_num2word(val):
            val = int(val) if is_int(val) else val
            # try to parse each word or word pairs in question_toks into number
            index_pairs = filter(lambda x: x[1] - x[0] in [1, 2], combinations(range(len(question_toks) + 1), 2))
            for i, j in sorted(index_pairs, key=lambda x: x[1] - x[0]):
                try:
                    w = ''.join(question_toks[i: j])
                    num = word2num(w.replace('个', '')) # 一个亿 -> 一亿
                    if float_equal(num, val, 1): # no ambiguity in the dataset
                        add_value_from_token_idx((i, j), question_toks, sqlvalue)
                        return True
                except Exception as e: pass
            # try to parse the sql value to chinese words and check if it appears in the question
            chars = list(zip('零一二三四五六七八九', '0123456789'))
            for mode in ['low', 'direct']:
                word = num2word(val, mode)
                if word in question_toks and question_toks.count(word) == 1:
                    start = question_toks.index(word)
                    add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
                    return True
                elif word in question and question.count(word) == 1:
                    start_id = question.index(word)
                    add_value_from_char_idx((start_id, start_id + len(word)), question_toks, sqlvalue, entry)
                    return True
                else:
                    for old, new in chars:
                        word = word.replace(old, new) # 7亿8千万
                    if not is_number(word) and word in question and question.count(word) == 1:
                        # only deal with mixture of digits and chars
                        start_id = question.index(word)
                        add_value_from_char_idx((start_id, start_id + len(word)), question_toks, sqlvalue, entry)
                        return True
            # 3rd cases: cn2an fails to parse 8.5万亿 and 4k, 3w
            for idx, tok in enumerate(question_toks):
                match_obj = re.search(r'([0-9\.]+)([kw千万亿]+)', tok)
                if match_obj:
                    digits, chars = match_obj.groups()
                    try:
                        digits = float(digits)
                        chars = 1e3 ** chars.count('k') * 1e3 ** chars.count('千') * 1e4 ** chars.count('万') * 1e4 ** chars.count('w') * 1e8 ** chars.count('亿')
                        num = digits * chars
                        if float_equal(num, val, 1): # also no ambiguity
                            add_value_from_token_idx((idx, idx + 1), question_toks, sqlvalue)
                            return True
                    except: pass
            return False

        def try_parsing_percentage(val):
            # no ambiguity problem in the dataset, if find candidate, return
            negative = val < 0 - 1e-6
            val_pos = - val if negative else val
            for idx, w in enumerate(question_toks):
                # - will be tokenized in preprocessing
                w_norm = w.rstrip('%')
                if is_number(w_norm) and float_equal(w_norm, val_pos, 100):
                    prefix = ''.join(question_toks[:idx])
                    if negative: # a few cases
                        char_id = entry['word2char_id_mapping'][idx]
                        if char_id > 0 and question[char_id - 1] in ['负', '-']:
                            if prefix[:-1].endswith('百分之'):
                                start_char_id = char_id - 4
                            else: start_char_id = char_id - 1
                            start_idx = entry['char2word_id_mapping'][start_char_id]
                            add_value_from_token_idx((start_idx, idx + 1), question_toks, sqlvalue)
                            return True
                        else: continue
                    else:
                        if prefix.endswith('百分之'):
                            char_id = entry['word2char_id_mapping'][idx]
                            start_char_id = char_id - 3
                            start_idx = entry['char2word_id_mapping'][start_char_id]
                        else: start_idx = idx
                        add_value_from_token_idx((start_idx, idx + 1), question_toks, sqlvalue)
                        return True
            # some cases, 百分之4点25, 百分之二
            match_obj = re.search(r'百分之([0-9{}负点十]+)'.format(CN_NUMBER_1), question)
            if match_obj:
                span, start_id, end_id = match_obj.group(1), match_obj.start(), match_obj.end()
                try:
                    num = word2num(span)
                    if float_equal(num, val, 100):
                        add_value_from_char_idx((start_id, end_id), question_toks, sqlvalue, entry)
                        return True
                except: pass
            return False

        def try_ignoring_suffix(val):
            for idx, w in enumerate(question_toks):
                w = w.rstrip('千万亿')
                if is_number(w) and float_equal(w, val, 1):
                    add_value_from_token_idx((idx, idx + 1), question_toks, sqlvalue)
                    return True
            return False

        def try_parsing_height(val):
            match_objs = re.finditer(r'[1-9一二三四五六七八九][米|点][0-9零一二三四五六七八九]+', question)
            for match_obj in match_objs:
                span = match_obj.group(0)
                try:
                    num = word2num(span.replace('米', '点'))
                    if float_equal(num, val, 100):
                        start_id, end_id = match_obj.start(), match_obj.end()
                        add_value_from_char_idx((start_id, end_id), question_toks, sqlvalue, entry)
                        return True
                except: pass
            return False

        def try_parsing_date(val):
            if is_int(val): # year
                val = val - 2000 if val - 2000 > 0 else val - 1900
                val = str(int(val))
                if val in question_toks and question_toks.count(val) == 1:
                    start = question_toks.index(val)
                    add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
                    return True
            else: # time or date
                match_obj = re.search(r'([0-9]+)(小时|时|点|月)([0-9]+)(号|日|分钟|分)?', question)
                if match_obj:
                    first, second = match_obj.group(1), match_obj.group(3)
                    s = str(first) + '.' + str(second)
                    if is_number(s) and float_equal(s, val, 1):
                        start_id, end_id = match_obj.start(), match_obj.end()
                        add_value_from_char_idx((start_id, end_id), question_toks, sqlvalue, entry)
                        return True
            return False

        def process_outliers(val):
            if entry['question_id'] in ['qid012363', 'qid012364']:
                value = '1万8千九'
            elif str(val) == '7.32':
                value = '7.32.12'
            elif int(val) == 15000 and entry['question_id'] == 'qid001798':
                value = '1万五千块'
            else: raise ValueError('WHERE float value %s is not recognized in the question: %s' % (val, ' '.join(entry['uncased_question_toks'])))
            start_id = question.index(value)
            add_value_from_char_idx((start_id, start_id + len(value)), question_toks, sqlvalue, entry)

        if str(val) in question_toks: # no ambiguity
            w = str(val)
            start = question_toks.index(w)
            add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
        elif is_int(val) and str(int(val)) in question_toks:
            val = str(int(val))
            if question_toks.count(val) == 1:
                start = question_toks.index(val)
            else: # choose the first one except some cases like: 哪些话剧188元和100元票价的门票最少剩余100张？
                start = [idx for idx, w in enumerate(question_toks) if w == val and (idx == len(question_toks) - 1 or question_toks[idx + 1] != '元')][0]
            add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
        elif is_int(val) and int(val) == 0:
            add_value_from_reserved("0", sqlvalue)
        elif is_int(val) and int(val) < 10 and try_num2char(int(val)): pass
        elif try_num2word(val): pass
        elif try_parsing_percentage(val): pass
        elif try_ignoring_suffix(val): pass
        elif try_parsing_height(val): pass
        elif try_parsing_date(val): pass
        else: process_outliers(val)
        return values, question_toks

    def extract_where_string_val(self, val, values, question_toks, entry, state):
        sqlvalue = SQLValue(str(val), state)
        if sqlvalue in values: return values, question_toks
        else: values.add(sqlvalue)
        question = ''.join(question_toks)

        def parsing_date(val):
            date1 = re.match(r'([0-9]{4})\-([0-9]{2})\-([0-9]{2})', val)
            date2 = re.match(r'([0-9]{4})\-([0-9]{2})', val)
            date3 = re.match(r'([0-9]{2})\-([0-9]{2})', val)
            if date1:
                y, m, d = date1.group(1), date1.group(2).lstrip('0'), date1.group(3).lstrip('0')
                y_cn, m_cn, d_cn = num2word(y, 'direct'), num2word(m, 'low'), num2word(d, 'low')
                y, m, d = y + '|' + y_cn, m + '|' + m_cn, d + '|' + d_cn
                patt1 = r'({})[年\.\-/](0?{})[月\.\-/](0?{})[日号]?'.format(y, m, d)
                patt2 = r'({})[年\.\-/]({})({})'.format(y, m_cn, d_cn)
                m1, m2 = re.search(patt1, question), re.search(patt2, question)
                if m1:
                    add_value_from_char_idx((m1.start(), m1.end()), question_toks, sqlvalue, entry)
                    return True
                elif m2:
                    add_value_from_char_idx((m2.start(), m2.end()), question_toks, sqlvalue, entry)
                    return True
            elif date2:
                y, m = date2.group(1), date2.group(2).lstrip('0')
                patt1 = r'{}[年\.\-](0?{})月?'.format(y, m)
                m1 = re.search(patt1, question)
                if m1:
                    add_value_from_char_idx((m1.start(), m1.end()), question_toks, sqlvalue, entry)
                    return True
            elif date3:
                m, d = date3.group(1).lstrip('0'), date3.group(2).lstrip('0')
                m_cn, d_cn = num2word(m, 'low'), num2word(d, 'low')
                m, d = m + '|' + m_cn, d + '|' + d_cn
                patt1 = r'(0?{})[月\.\-](0?{})[日号]'.format(m, d)
                patt2 = r'{}{}'.format(m_cn, d_cn)
                m1, m2 = re.search(patt1, question), re.search(patt2, question_toks, sqlvalue, entry)
                if m1:
                    add_value_from_char_idx((m1.start(), m1.end()), )
                    return True
                elif m2:
                    add_value_from_char_idx((m2.start(), m2.end()),question_toks, sqlvalue, entry)
                    return True
            return False

        def utilize_synonym_set(val):
            for ss in SYNONYM_SETS:
                if val in ss:
                    for v in ss:
                        if v in question:
                            start_id = question.index(v)
                            add_value_from_char_idx((start_id, start_id + len(v)), question_toks, sqlvalue, entry)
                            return True
            return False

        def fuzzy_match(val):
            dist, length = [], len(val)
            index_pairs = list(filter(lambda x: min([1, length - 4]) <= x[1] - x[0] <= length + 4, combinations(range(len(question) + 1), 2)))
            index_pairs = sorted(index_pairs, key=lambda x: x[1] - x[0])
            for i, j in index_pairs:
                span = question[i:j]
                score = edt.eval(val, span)
                dist.append(float(score) / len(val))
            min_dist = min(dist)
            threshold = math.floor(len(val) / 2.0) / len(val) if len(val) < 6 else 0.4
            if min_dist <= threshold:
                index_pair = index_pairs[dist.index(min_dist)]
                add_value_from_char_idx((index_pair[0], index_pair[1]), question_toks, sqlvalue, entry)
                return True
            return False

        def parsing_company(val):
            if '公司' in val or '金拱门' in val:
                best = (0, 0) # record best result: start_id and end_id
                start, end = 0, 0
                while end < len(question):
                    if question[end] not in val:
                        if start != end and end - start > best[1] - best[0]:
                            best = (start, end)
                        start = end + 1
                    end += 1
                if start != end and end - start > best[1] - best[0]:
                    best = (start, end)
                if best[1] - best[0] >= 2:
                    add_value_from_char_idx((best[0], best[1]), question_toks, sqlvalue, entry)
                    return True
            return False

        def parsing_province_city(val):
            match1 = re.match(r'(.*)省(.*)市', val)
            if match1 and match1.group(2) in question:
                start_id = question.index(match1.group(2))
                add_value_from_char_idx((start_id, start_id + len(match1.group(2))), question_toks, sqlvalue, entry)
                return True
            match2 = re.match(r'(.*)市(.*)区', val)
            if match2 and match2.group(2) in question:
                start_id = question.index(match2.group(2))
                add_value_from_char_idx((start_id, start_id + len(match2.group(2))), question_toks, sqlvalue, entry)
                return True
            return False

        def parsing_time(val):
            if val.endswith(':00'):
                if '~' in val:
                    match = re.search(r'(上午|下午|晚上)?[0-9{}十]+点到(上午|下午|晚上)?[0-9{}十半]+点'.format(CN_NUMBER_1, CN_NUMBER_1), question)
                    if match:
                        add_value_from_char_idx((match.start(), match.end()), question_toks, sqlvalue, entry)
                        return True
                elif re.search(r'[^0]:[^0]', val):
                    match = re.search(r'(上午|下午)?[0-9{}十]+点[0-9{}十半]+分?'.format(CN_NUMBER_1, CN_NUMBER_1), question)
                    if match:
                        add_value_from_char_idx((match.start(), match.end()), question_toks, sqlvalue, entry)
                        return True
                else:
                    hour = re.search(r'([1-9]+):00', val)
                    if hour:
                        hour = hour.group(1) + '|' + num2word(int(hour.group(1)), 'low')
                        match = re.search(r'(上午|下午|晚上|周四)?({})(点|:00)'.format(hour), question)
                        if match:
                            add_value_from_char_idx((match.start(), match.end()), question_toks, sqlvalue, entry)
                            return True
            return False

        val = quote_normalization(val).lower()
        if val in ['是', '否']:
            add_value_from_reserved(val, sqlvalue)
        elif val in question:
            start_id = question.index(val)
            add_value_from_char_idx((start_id, start_id + len(val)), question_toks, sqlvalue, entry)
        elif val.startswith('item_'):
            mapped_val = entry['item_mapping'][val]
            if mapped_val in question:
                start_id = question.index(mapped_val)
                add_value_from_char_idx((start_id, start_id + len(mapped_val)), question_toks, sqlvalue, entry)
            else:
                self.use_extracted_values(sqlvalue, values)
        elif parsing_date(val): pass
        elif utilize_synonym_set(val): pass
        elif fuzzy_match(val): pass
        elif self.use_extracted_values(sqlvalue, values): pass
        elif parsing_company(val): pass
        elif parsing_time(val): pass
        elif parsing_province_city(val): pass
        else:
            raise ValueError('WHERE string value %s is not recognized in the question: %s' % (val, ' '.join(entry['uncased_question_toks'])))
        return values, question_toks

    def extract_having_val(self, num, values, question_toks, entry, state):
        """ Having values are all numbers in the dataset, decision procedure:
        1. str(val) is in question_toks, create a new entry in the set ``values``
        2. the mapped word of val is in question_toks, use the word as matched value candidate
        3. num == 1 and some word can be viewed as an evidence of value 1, use that word as candidate
        4. num == 1, directly add pre-defined value
        5. o.w. try to retrieve value candidate from already generated values
        """
        assert is_number(num)
        sqlvalue = SQLValue(str(num), state)
        if sqlvalue in values: return values, question_toks
        else: values.add(sqlvalue)
        num = int(num) if is_int(num) else float(num)
        question = ''.join(question_toks)

        def try_percentage_variants(val):
            val_100 = val * 100
            variants = [str(int(val_100)) + '%', str(float(val_100)) + '%', str(float(val)), str(val) + '%']
            for v in variants:
                if v in question_toks and question_toks.count(v) == 1:
                    start = question_toks.index(v)
                    add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
                    return True
            if is_int(val_100):
                val_100 = int(val_100)
                variants = ['百分之' + str(val_100), '百分之' + num2word(val_100, 'low'), '百分之' + num2word(val_100, 'direct')]
            else:
                variants = ['百分之' + str(val_100), '百分之' + str(val).replace('.', '点'), '百分之' + num2word(val_100, 'low'), '百分之' + num2word(val_100, 'direct')]
            for v in variants:
                if v in question and question.count(v) == 1:
                    char_id = question.index(v)
                    add_value_from_char_idx((char_id, char_id + len(v)), question_toks, sqlvalue, entry)
                    return True
            return False

        def try_digit_to_char(num):
            if num == 5 and entry['question_id'] == 'qid003923':
                start = question_toks.index('五个')
                add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
                return True
            # for integer number less than 10
            for char in NUM_ALIAS(num):
                if char in question and question.count(char) == 1:
                    char_id = question.index(char)
                    add_value_from_char_idx((char_id, char_id + 1), question_toks, sqlvalue, entry)
                    return True
            return False

        def try_num2word_parse(val):
            # from each word to number
            for start, w in enumerate(question_toks):
                try:
                    num = word2num(w)
                    if num == val:
                        add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
                        return True
                except Exception as e: pass
            # from number to word
            for mode in ['low', 'direct']:
                try:
                    word = num2word(val, mode)
                    if word in question_toks and question_toks.count(word) == 1:
                        start = question_toks.index(word)
                        add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
                        return True
                    elif word in question and question.count(word) == 1:
                        char_id = question.index(word)
                        add_value_from_char_idx((char_id, char_id + len(word)), question_toks, sqlvalue, entry)
                        return True
                except: pass
            return False

        def try_word_number_mixture(val):
            # 1米9, 1万5
            pattern = r'[一二两三四五六七八九1-9][米万][一二两三四五六七八九1-9]+'
            for match_obj in re.finditer(pattern, question):
                s = re.sub(r'[米万]', '点', match_obj.group(0))
                try:
                    s = word2num(s)
                    for num in [str(s), str(int(s * 100)), str(int(s * 10000))]:
                        if num == str(val):
                            start_id, end_id = match_obj.start(), match_obj.end()
                            add_value_from_char_idx((start_id, end_id), question_toks, sqlvalue, entry)
                            return True
                except: pass
            return False

        def check_valid_prefix(val):
            # due to different metrics, there maybe redudant or missing 0s
            val, valid_toks = str(val), []
            for idx, w in enumerate(question_toks):
                try:
                    num = str(word2num(w))
                    if val.startswith(num) and val == num + '0' * (len(val) - len(num)):
                        valid_toks.append(idx)
                    if num.startswith(val) and num == val + '0' * (len(num) - len(val)):
                        valid_toks.append(idx)
                except: pass
            if len(valid_toks) == 1:
                start = valid_toks[0]
                add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
                return True
            return False

        if not is_int(num): # float number
            if str(num) in question_toks:
                start = question_toks.index(str(num))
                add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
            elif try_percentage_variants(num): pass
            elif num == 0.5 and '一半' in question_toks:
                start = question_toks.index('一半')
                add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
            else:
                raise ValueError('Not recognized HAVING float value %s' % (num))
        elif str(num) in question_toks:
            num = str(num)
            if question_toks.count(num) == 1:
                start = question_toks.index(num)
                add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
            else: # duplicate values
                start = [i for i in range(len(question_toks)) if question_toks[i] == num and \
                    (i == 0 or question_toks[i - 1] != '每') and \
                    (i == len(question_toks) -1 or (question_toks[i + 1] not in ['月', '日', '星']))]
                add_value_from_token_idx((start[0], start[0] + 1), question_toks, sqlvalue)
        elif num < 10 and try_digit_to_char(num): pass
        elif try_num2word_parse(num): pass
        elif try_percentage_variants(num): pass
        elif try_word_number_mixture(num): pass
        elif check_valid_prefix(num): pass
        else:
            if num == 1:
                add_value_from_reserved("1", sqlvalue)
            else: raise ValueError('Unresolved HAVING value %s' % (num))
        return values, question_toks

    def extract_limit_val(self, limit, values, question_toks, entry, track):
        """ Decision procedure for LIMIT value (type int): luckily, only integers less or equal than 10
        1. num != 1 and str(num) directly appears in question_toks:
            a. no duplicate occurrence, add a new entry in the set ``values``
            b. try to resolve the ambiguity by prompt words such as '哪', '前', '后', '最'
            c. use the first one, o.w.
        2. num != 1 and the mapped char of num in question_toks/question, use that char as matched value candidate
        3. some annotation errors in the dataset, try fixing: 3 -> 5, 10 ; 5 -> 3
        4. directly add pre-defined vocab value, if num != 1, remember to fix the annotation error
        Each time when some span is matched, update question_toks to prevent further match
        """
        num = limit['limit']
        state = State(track, 'none', '==', 'none', 0)
        sqlvalue = SQLValue(str(num), state)
        if sqlvalue in values: return values, question_toks
        else: values.add(sqlvalue)
        num, question = int(num), ''.join(question_toks)

        def resolve_ambiguity(num):
            word_idxs = [i for i in range(len(question_toks)) if question_toks[i] == str(num)]
            char_idxs = [entry['word2char_id_mapping'][i] for i in word_idxs]
            prev_chars = [question[i - 1] if i >= 1 else '' for i in char_idxs]
            for prompt in ['哪', '前', '后']:
                if prompt in prev_chars:
                    index = prev_chars.index(prompt)
                    break
            else: # the first occurrence after '最'
                if '最' in question:
                    prompt_idx = question.index('最')
                    indexes = [idx for idx in range(len(char_idxs)) if char_idxs[idx] > prompt_idx]
                    index = indexes[0] if len(indexes) > 0 else 0
                else:
                    index = 0
            return word_idxs[index]

        def try_number_alias(num):
            if num >= 10: return False
            for char in NUM_ALIAS(num):
                count = question.count(char)
                if count == 0: continue
                if count == 1:
                    char_id = question.index(char)
                    word_id = entry['char2word_id_mapping'][char_id]
                    add_value_from_token_idx((word_id, word_id + 1), question_toks, sqlvalue)
                    return True
                else:
                    char_idxs = [idx for idx in range(len(question)) if question[idx] == char]
                    if '最' in question: # 最多的三个，最少的两个
                        prompt_idx = question.index('最')
                        indexes = [idx for idx in range(len(char_idxs)) if char_idxs[idx] > prompt_idx]
                        index = indexes[0] if indexes else 0
                    else: index = 0
                    word_id = entry['char2word_id_mapping'][char_idxs[index]]
                    add_value_from_token_idx((word_id, word_id + 1), question_toks, sqlvalue)
                    return True
            return False

        def try_fixing_error(num):
            # some annotation errors in the dataset
            if num == 3: # try 5 and 10
                if '5' in question_toks:
                    print('Fix LIMIT annotation error 3->5:', entry['question'])
                    limit['limit'] = 5
                    entry['query'] = re.sub(r'limit {}'.format(num), 'limit 5', entry['query'], flags=re.I)
                    sqlvalue.real_value = str(5)
                    index = question_toks.index('5') if question_toks.count('5') == 1 else resolve_ambiguity(5)
                    add_value_from_token_idx((index, index + 1), question_toks, sqlvalue)
                    return True
                elif '10' in question_toks:
                    print('Fix LIMIT annotation error 3->10:', entry['question'])
                    limit['limit'] = 10
                    entry['query'] = re.sub(r'limit {}'.format(num), 'limit 10', entry['query'], flags=re.I)
                    sqlvalue.real_value = str(10)
                    index = question_toks.index('10') if question_toks.count('10') == 1 else resolve_ambiguity(10)
                    add_value_from_token_idx((index, index + 1), question_toks, sqlvalue)
                    return True
            elif num == 5: # try 3
                if '3' in question_toks:
                    print('Fix LIMIT annotation error 5->3:', entry['question'])
                    limit['limit'] = 3
                    entry['query'] = re.sub(r'limit {}'.format(num), 'limit 3', entry['query'], flags=re.I)
                    sqlvalue.real_value = str(3)
                    index = question_toks.index('3') if question_toks.count('3') == 1 else resolve_ambiguity(3)
                    add_value_from_token_idx((index, index + 1), question_toks, sqlvalue)
                    return True
            return False

        if num != 1 and str(num) in question_toks:
            num = str(num)
            if question_toks.count(num) == 1:
                start = question_toks.index(num)
                add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
            else: # duplicate occurrences
                start = resolve_ambiguity(num)
                add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
        elif num != 1 and try_number_alias(num): pass
        elif num != 1 and try_fixing_error(num): pass
        else:
            if num != 1: # need to modify the annotation, including "sql" and "query" keys
                limit['limit'] = 1
                entry['query'] = re.sub(r'limit {}'.format(num), 'limit 1', entry['query'], flags=re.I)
                sqlvalue.real_value = str(1) # potential error: revise value already in the value set
            add_value_from_reserved("1", sqlvalue)
        return values, question_toks

def add_value_from_reserved(val, sqlvalue):
    sqlvalue.add_candidate(SelectValueAction.reserved_dusql[val])

def add_value_from_token_idx(index_pairs, question_toks, sqlvalue):
    value = ''.join(question_toks[index_pairs[0]: index_pairs[1]])
    candidate = ValueCandidate(matched_index=tuple(index_pairs), matched_value=value)
    sqlvalue.add_candidate(candidate)
    question_toks[index_pairs[0]: index_pairs[1]] = [PLACEHOLDER * len(question_toks[idx]) for idx in range(index_pairs[0], index_pairs[1])]
    return question_toks

def add_value_from_char_idx(index_pairs, question_toks, sqlvalue, entry):
    start_id, end_id = index_pairs
    start = entry['char2word_id_mapping'][start_id]
    end = entry['char2word_id_mapping'][end_id - 1] + 1
    value = ''.join(question_toks[start: end])
    candidate = ValueCandidate(matched_index=(start, end), matched_value=value)
    sqlvalue.add_candidate(candidate)
    question_toks[start: end] = [PLACEHOLDER * len(question_toks[idx]) for idx in range(start, end)]
    return question_toks

def generate_test_pairs(dataset):
    samples = []
    for ex in dataset:
        values = ex['values']
        cur_pairs = []
        for v in values:
            # state: (track, agg_op, col_id/str, cmp_op)
            match_value = v.candidate.matched_cased_value if isinstance(v.candidate, ValueCandidate) else SelectValueAction.reserved_dusql.id2word[v.candidate]
            cur_pairs.append((v.real_value, v.state, match_value))
        samples.append(cur_pairs)
    return samples


if __name__ == '__main__':

    from eval.evaluation import is_float
    def _equal(p, g): # compare value script from DuSQL official script
        p = p.strip('"\'') if type(p) is str else p
        g = g.strip('"\'') if type(g) is str else g
        if str(p) == str(g):
            return True
        if is_float(p) and is_float(g) and float(p) == float(g):
            return True
        return False

    # try to normalize the value mentioned in the question into the SQL values in the query
    data_dir = DATASETS['dusql']['data']
    table_path = os.path.join(data_dir, 'tables.bin')
    db_dir = DATASETS['dusql']['database']
    processor = ValueProcessor(table_path=table_path, db_dir=db_dir)

    dataset = pickle.load(open(os.path.join(data_dir, 'train.bin'), 'rb'))
    # dataset = pickle.load(open(os.path.join(data_dir, 'dev.bin'), 'rb'))

    test_pairs = generate_test_pairs(dataset)
    ex_correct, count, correct = 0, 0, 0
    for ex, pairs in zip(dataset, test_pairs):
        db = processor.tables[ex['db_id']]
        flag = True
        for pair in pairs:
            gold_value, state, match_value = pair
            pred_value = processor.obtain_possible_value(match_value, db, state, ex)
            count += 1
            if _equal(pred_value, gold_value):
                correct += 1
            else:
                flag = False
                clause, col_id = state.track.split('->')[-1], state.col_id
                if col_id == 'TIME_NOW':
                    col_name = col_id
                    col_type = 'time'
                else:
                    col_name = processor.tables[ex['db_id']]['column_names_original'][col_id][1]
                    col_type = processor.tables[ex['db_id']]['column_types'][col_id]
                print('Clause %s Column %s[%s]: Gold/Match/Pred value: %s/%s/%s' % (clause, col_name, col_type, gold_value, match_value, pred_value))
        if flag:
            ex_correct += 1
        else:
            print('SQL:', ex['query'])
            print('Id: %s Question:' % (ex['question_id']), ' '.join(ex['cased_question_toks']), '\n')
    print('Values postprocess accuracy is %.4f' % (correct / float(count)))
    print('Samples postprocess accuracy is %.4f' % (ex_correct / float(len(dataset))))
