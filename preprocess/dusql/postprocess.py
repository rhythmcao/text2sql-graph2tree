#coding=utf8
import re, json, os, sys, pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import datetime
import numpy as np
import jionlp as jio
from decimal import Decimal
from fuzzywuzzy import process
from collections import Counter
from utils.constants import DATASETS
from asdl.transition_system import SelectValueAction
from preprocess.process_utils import is_number, is_int, BOOL_TRUE_ZH, BOOL_FALSE_ZH, ZH_NUMBER, ZH_UNIT, ZH_UNIT_MAPPING, ZH_RESERVE_CHARS, ZH_NUM2WORD, ZH_WORD2NUM
from preprocess.process_utils import load_db_contents, extract_db_contents

def compare_and_extract_date(t1, t2):
    y1, m1, d1 = t1.split('-')
    y2, m2, d2 = t2.split('-')
    if y1 == y2 and m1 == m2:
        return y1 + '-' + m1
    if m1 == m2 and d1 == d2:
        return m1 + '.' + d1 # return m1 + '-' + d1
    if y1 == y2: return y1
    if m1 == m2: return m1
    if d1 == d2: return d1
    return None

def compare_and_extract_time(t1, t2, full=False):
    h1, m1, s1 = t1.split(':')
    h2, m2, s2 = t2.split(':')
    if h1 == h2 and m1 == m2: # ignore second is better
        return h1.lstrip('0') + ':' + m1 if not full else h1 + ':' + m1 + ':00'
    if h1 == h2: # no difference whether ignore second
        return h1.lstrip('0') + ':00' if not full else h1 + ':00:00'
    if m1 == m2 and s1 == s2:
        return m1.lstrip('0') + ':' + s1
    if m1 == m2:
        return m1.lstrip('0') + ':00'
    return None

def extract_time_or_date(result, val):
    # extract date or time string from jio parsed result
    tt, obj = result['type'], result['time']
    if tt in ['time_span', 'time_point']: # compare list
        t1, t2 = obj[0], obj[1]
        t1_date, t1_time = t1.split(' ')
        t2_date, t2_time = t2.split(' ')
        today = datetime.datetime.today()
        if t1_time == '00:00:00' and t2_time == '23:59:59': # ignore time
            if t1_date == t2_date:
                if str(today.year) == t1_date[:4]: # ignore current year prepended
                    return t1_date[5:].replace('-', '.') # return t1_date[5:]
                return t1_date
            else: # find the common part
                return compare_and_extract_date(t1_date, t2_date)
        elif t1_date == t2_date == today.strftime("%Y-%m-%d"): # ignore date
            if '到' not in val: return compare_and_extract_time(t1_time, t2_time)
            else: return '-'.join(t1_time, t2_time)
        else: # preserve both date and time
            date_part = t1_date
            time_part = compare_and_extract_time(t1_time, t2_time, full=True)
            return date_part + '-' + time_part
    elif tt == 'time_delta':
        v = [obj[k] for k in ['year', 'month', 'day', 'hour'] if k in obj]
        if len(v) > 0: # only use one value is enough in DuSQL
            return v[0]
    return None


def parse_mixed_float_and_metric(val):
    match_obj = re.search(fr'([0-9\.{ZH_NUMBER}]+)([{ZH_UNIT}]+)', val)
    if match_obj:
        float_num, metric = match_obj.group(1), match_obj.group(2)
        if is_number(float_num): num = float(float_num)
        else:
            try: num = ZH_WORD2NUM(float_num)
            except: return None
        metric = np.prod([ZH_UNIT_MAPPING[m] for m in list(metric)])
        num = float(Decimal(str(num)) * Decimal(str(metric)))
        return num
    return None


def process_hyphen(val):   
    if '-' in val and val.count('-') == 1:
        split_vals = [v for v in val.split('-') if v.strip()]
        # determine negative number or num1-num2
        all_vals = split_vals if len(split_vals) > 1 else [val.strip()]
    elif '~' in val and val.count('~') == 1:
        all_vals = [v for v in val.split('~') if v.strip()]
    elif '到' in val and val.count('到') == 1:
        all_vals = [v for v in val.split('到') if v.strip()]
    else: all_vals = [val.strip()]
    return all_vals


class ValueProcessor():

    def __init__(self, table_path, db_dir) -> None:
        self.db_dir = db_dir
        if type(table_path) == str and os.path.splitext(table_path)[-1] == '.json':
            tables_list = json.load(open(table_path, 'r'))
            self.tables = { db['db_id']: db for db in tables_list }
        elif type(table_path) == str:
            self.tables = pickle.load(open(table_path, 'rb'))
        else: self.tables = table_path
        self.contents = self._load_db_contents()

    def _load_db_contents(self):
        contents = load_db_contents(self.db_dir)
        self.contents = {}
        for db_id in self.tables:
            db = self.tables[db_id]
            self.contents[db_id] = db['cells'] if 'cells' in db else extract_db_contents(contents, db)
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
        # chinese chars do not have whitespace, while whitespace is needed between english words
        # when ValueCandidate is constructed in models/encoder/auxiliary.py, whitespaces are inserted in matched_value
        raw_value = re.sub(r'([^a-zA-Z])(\s+)([^a-zA-Z])', lambda match_obj: match_obj.group(1) + match_obj.group(3), raw_value)
        value = self.postprocess_raw_value_string(raw_value, db, state, entry)
        return value

    def postprocess_raw_value_string(self, raw_value, db, state, entry):
        value = ''
        clause, col_id = state.track.split('->')[-1], state.col_id
        if col_id == 'TIME_NOW':
            col_name, col_type, cell_values = col_id, 'time', []
        else:
            col_name = db['column_names_original'][col_id][1]
            col_type = db['column_types'][col_id]
            cell_values = self.contents[db['db_id']][col_id]

        def word_to_number(val, name, time_return=True):
            if re.search(r'item', val): return False
            if time_return and (':' in val or '/' in val or '_' in val \
                or re.search(r'(上午|下午|晚上|时|点|分|秒|月|日|号)', val) \
                    or val.count('-') > 1 or val.count('.') > 1): return False
            all_vals = process_hyphen(val)
            parsed_nums = []
            for v in all_vals:
                # percentage all changes to float number
                is_percent = re.search(r'百分之|%', v) is not None
                v = re.sub(r'百分之|%', '', v)
                # height all use cm as the metric, multiply 100
                is_height = re.search(fr'[1-9{ZH_NUMBER}]米[0-9{ZH_NUMBER}]+', v) is not None
                v = v.replace('米', '点') if is_height else v
                # multiplier coefficient if represents area or distance
                is_area = re.search(r'平方公里|平方千米', v) is not None
                is_distance = re.search(r'公里', v) is not None and (not is_area)
                # ignore these metrics containing ambiguous char 千 or k
                v = re.sub(r'千米|千瓦|千克|千斤|千卡|kg|km', '', v, flags=re.I)
                v = re.sub(r'w', '万', re.sub(r'k', '千', v, flags=re.I), flags=re.I)
                v = re.sub(r'[^0-9\.\-{}]'.format(ZH_RESERVE_CHARS), '', v)
                if is_number(v): v = float(v)
                else:
                    try: v = ZH_WORD2NUM(v)
                    except: 
                        v = parse_mixed_float_and_metric(v)
                        if v is None: return False
                    
                v = float(Decimal(str(v)) * Decimal('100')) if is_height else \
                        float(Decimal(str(v)) * Decimal('0.01')) if is_percent else \
                            float(Decimal(str(v)) * Decimal(str('1000000'))) if is_area else \
                                float(Decimal(str(v)) * Decimal(str('1000'))) if is_distance else float(v)

                num = int(v) if is_int(v) else v
                if '(' in name: # metric exist in the column name, e.g. 投资额(万亿)
                    metric = re.search(r'\((.*?)\)', name).group(1)
                    metric = re.sub(r'千米|千瓦|千克|千斤|千卡|kg|km', '', metric, flags=re.I)
                    metric = [ZH_UNIT_MAPPING[m] for m in list(metric) if m in ZH_UNIT_MAPPING]
                    if len(metric) > 0:
                        factor = np.prod(metric)
                        if is_int(num) and num % factor == 0:
                            num = int(num // factor)
                elif '届' in name and cell_values: # e.g. 第十八届, 第? num/word 届?
                    templates = [tuple([cv.startswith('第'), re.search(r'\d+', cv) is not None, '届' in cv]) for cv in cell_values]
                    counter = Counter(templates)
                    flags = counter.most_common(1)[0][0]
                    num = str(num) if flags[1] else ZH_NUM2WORD(num, 'low')
                    num = '第' + num if flags[0] else num
                    num = num + '届' if flags[2] else num
                
                parsed_nums.append(str(num))

            nonlocal value
            value = '-'.join(parsed_nums)
            return True

        def word_to_time_or_date(val):
            try:
                if re.search(r'[\d%s]{1,4}年[\d%s十]{1,2}月[\d%s十]{1,3}$' % (ZH_NUMBER, ZH_NUMBER, ZH_NUMBER), val): val += '日'
                match_obj = re.search(r'(\d{1,4}[-/]\d{1,2}[-/]\d{1,4})( \d{1,2}:\d{1,2}:\d{1,2})?', val)
                if match_obj: val = match_obj.group(0)
                result = jio.parse_time(val)
                result = extract_time_or_date(result, val)
                if result is not None:
                    nonlocal value
                    value = result
                    return True
            except: pass
            return False

        def try_item_mappings(val):
            if 'item' in val:
                idx = re.sub(r'[^0-9]', '', val[val.index('item') + 4:])
                if is_int(idx):
                    idx = int(idx)
                    if 0 <= idx < len(entry['item_mapping_reverse']):
                        nonlocal value
                        value = entry['item_mapping_reverse'][idx]
                        return True
            return False

        if state.cmp_op == '!=' and raw_value.startswith('非'):
            raw_value = raw_value.lstrip('非')
        if clause == 'limit': # value should be integers
            if is_number(raw_value):
                value = int(float(raw_value))
            elif word_to_number(raw_value, col_name, time_return=False):
                value = int(float(value))
            else: value = 1 # for all other cases, use 1
        elif clause == 'having': # value should be numbers
            if is_number(raw_value):
                value = int(float(raw_value)) if is_int(raw_value) else float(raw_value)
            elif word_to_number(raw_value, col_name, time_return=False): pass
            else: value = 1
        else: # WHERE clause
            def try_binary_values(val):
                if val in ['是', '否']:
                    evidence = []
                    for cv in cell_values:
                        for idx, (t, f) in enumerate(zip(BOOL_TRUE_ZH, BOOL_FALSE_ZH)):
                            if cv == t or cv == f:
                                evidence.append(idx)
                                break
                    nonlocal value
                    if len(evidence) > 0:
                        bool_idx = Counter(evidence).most_common(1)[0][0]
                        value = BOOL_TRUE_ZH[bool_idx] if val == '是' else BOOL_FALSE_ZH[bool_idx]
                    else: value = val
                    return True
                return False

            if col_type == 'number':
                if try_item_mappings(raw_value): pass
                elif word_to_number(raw_value, col_name, time_return=False): pass
                else: value = str(raw_value)
            elif col_type == 'time':
                if word_to_number(raw_value, col_name, time_return=True): pass
                elif word_to_time_or_date(raw_value): pass
                else: value = str(raw_value)
            elif col_type == 'binary' and try_binary_values(raw_value): pass
            else:
                like_op = 'like' in state.cmp_op
                if len(cell_values) > 0 and not like_op:
                    most_similar = process.extractOne(raw_value, cell_values)
                    value = most_similar[0] if most_similar[1] > 50 else raw_value
                    # additional attention: the retrieved value must have the same number within
                    if re.sub(r'[^\d]', '', value) != re.sub(r'[^\d]', '', raw_value): value = raw_value
                else: value = raw_value.lstrip('姓').rstrip('类').rstrip('型')
        return str(value) if is_number(value) else "'" + value + "'"


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

    from preprocess.process_utils import ValueCandidate
    def generate_test_samples(dataset):
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

    # try to normalize the value mentioned in the question into the SQL values in the query
    data_dir = DATASETS['dusql']['data']
    table_path = os.path.join(data_dir, 'tables.bin')
    db_dir = DATASETS['dusql']['database']
    processor = ValueProcessor(table_path=table_path, db_dir=db_dir)

    dataset = pickle.load(open(os.path.join(data_dir, 'train.lgesql.bin'), 'rb'))
    # dataset = pickle.load(open(os.path.join(data_dir, 'dev.lgesql.bin'), 'rb'))

    test_samples = generate_test_samples(dataset)
    instance_correct, count, correct = 0, 0, 0
    for entry, sample in zip(dataset, test_samples):
        db = processor.tables[entry['db_id']]
        flag, count = True, count + len(sample)
        for gold_value, state, match_value in sample:
            pred_value = processor.postprocess_raw_value_string(match_value, db, state, entry)
            if _equal(pred_value, gold_value): correct += 1
            else:
                flag = False
                clause, col_id = state.track.split('->')[-1], state.col_id
                if col_id == 'TIME_NOW':
                    col_name = col_id
                    col_type = 'time'
                else:
                    col_name = processor.tables[entry['db_id']]['column_names_original'][col_id][1]
                    col_type = processor.tables[entry['db_id']]['column_types'][col_id]
                print('Clause %s Column %s[%s]: Gold/Match/Pred value: %s/%s/%s' % (clause, col_name, col_type, gold_value, match_value, pred_value))
        if flag: instance_correct += 1
        else:
            print('Question[%s]: [%s]' % (entry['question_id'], '|'.join(entry['cased_question_toks'])))
            print('SQL: %s\n' % (entry['query']))
    print('Values postprocess accuracy is %.4f' % (correct / float(count)))
    print('Samples postprocess accuracy is %.4f' % (instance_correct / float(len(dataset))))