#coding=utf8
import re, os, sys, copy, math, json, pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import string
import numpy as np
from decimal import Decimal
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from rouge import Rouge
from asdl.transition_system import SelectValueAction
from preprocess.process_utils import ValueCandidate, float_equal, load_db_contents, extract_db_contents
from preprocess.process_utils import is_number, is_int
from preprocess.process_utils import QUOTATION_MARKS, ZH_WORD2NUM, ZH_NUMBER, ZH_UNIT, ZH_UNIT_MAPPING
from preprocess.nl2sql.input_utils import NORM, STOPWORDS
from preprocess.nl2sql.value_utils import PLACEHOLDER, EN_NE, ZH_NE, EN2ZH_NE, ZH2EN_NE

mappings = dict(zip('零一二三四五六七八九两点', '01234567892.'))
NUM_NORM = lambda s: re.sub('[' + ''.join(mappings.keys()) + ']', lambda m: mappings[m.group(0)], NORM(s.strip().lower()))

EN_NE_PATTERN = '(' + '|'.join(sorted(EN_NE, key=lambda s: -len(s))) + ')'
ZH_NE_PATTERN = '(' + '|'.join(sorted(ZH_NE, key=lambda s: -len(s))) + ')'

ABBREV_SET = [
    set({'湖南卫视', '湖南台', '芒果台', '芒果tv', '芒果', '马桶台'}), set({'初一', '七年级'}), set({'初二', '八年级'}), set({'腾讯', '企鹅公司', '鹅厂'}), set({'新办', '第一次', '首次'}),
    set({'嘉应学院', '嘉大'}), set({'攀枝花学院', '攀大'}), set({'一次性', '一次缴', '趸缴'}), set({'船用燃油硫排放', 'seca', '硫氧化物排放控制区'}), set({'春城', '昆明'}), set({'鹿城', '三亚'}),
    set({'液晶电视-海尔', '海尔牌液晶电视'}), set({'粤语', '广东话'}), set({'卫生间', '洗手间'}), set({'3a', 'aaa', '三a', '三个a', '3个a'}), set({'5a', 'aaaaa', '五a', '五个a', '5个a'}),
    set({'蓝莓台', '蓝鲸台', '浙江台', '浙江卫视'}), set({'江苏卫视', '红台', '江苏台', '荔枝台'}), set({'北京卫视', '北京台', '北京电视台', 'btv'}), set({'流通', '销售', '售卖'}),
    set({'红利分发', '分红型'}), set({'不合格', '不达标', '不符合标准', '不符合规定', '达不到标准', '没达到标准'}), set({'合格', '合格品', '及格', '达标', '符合规定', '达到标准'}),
] + [set({k, v}) for k, v in EN2ZH_NE.items()]

def translate_possible_entity(s):
    if re.search(EN_NE_PATTERN, s, flags=re.I):
        s = re.sub(EN_NE_PATTERN, lambda m: EN2ZH_NE[m.group(0)], s)
    elif re.search(ZH_NE_PATTERN, s, flags=re.I):
        s = re.sub(ZH_NE_PATTERN, lambda m: ZH2EN_NE[m.group(0)], s)
    return s


def parse_number2word(span):
    try:
        span = ZH_WORD2NUM(span)
        span = int(float(span)) if is_int(span) else float(span)
        return span
    except:
        try:
            maps = dict(zip('0123456789.-', '零一二三四五六七八九点负'))
            span = re.sub(r'[0123456789\.\-]', lambda m: maps[m.group()], span)
            span = ZH_WORD2NUM(span)
            span = int(float(span)) if is_int(span) else float(span)
            return span
        except: return None


def parse_date(value_str):
    year_month_day_match = re.search(rf'^([\d{ZH_NUMBER}]' + r'{2,4})年.*?' + rf'([\d{ZH_NUMBER}十]+)月.*?' + rf'([\d{ZH_NUMBER}十]+)[日号].*?', value_str)
    if year_month_day_match: # 19年3月21日 -> 20190321
        y, m, d = year_month_day_match.groups()
        try:
            y, m, d = ZH_WORD2NUM(y), ZH_WORD2NUM(m), ZH_WORD2NUM(d)
            if y < 30: y = 2000 + y
            elif y < 100: y = 1900 + y
            if len(str(m)) == 1: m = '0' + str(m)
            if len(str(d)) == 1: d = '0' + str(d)
            ymd = str(y) + str(m) + str(d)
            return ymd
        except: pass

    year_month_match = re.search(rf'^([\d{ZH_NUMBER}]' + r'{2,4})年.*?' + rf'([\d{ZH_NUMBER}十]+)月', value_str)
    if year_month_match: # 19年4月 -> 2019.04
        y, m = year_month_match.groups()
        try:
            y, m = ZH_WORD2NUM(y), ZH_WORD2NUM(m)
            if y < 30: y = 2000 + y
            elif y < 100: y = 1900 + y
            if len(str(m)) == 1: m = '0' + str(m)
            ym = str(y) + '.' + str(m)
            return ym
        except: pass

    year_match = re.search(rf'^([\d{ZH_NUMBER}]' + r'{2})[年级]', value_str)
    if year_match:
        y = year_match.group(1)
        try:
            y = ZH_WORD2NUM(y)
            if y < 30: y = 2000 + y
            elif y < 100: y = 1900 + y
            return str(y)
        except: pass
    return None


def jaccard_score(s1, s2):
    return len(set(s1) & set(s2)) * 1.0 / len(set(s1) | set(s2))


class ValueProcessor():

    def __init__(self, table_path, db_dir) -> None:
        self.db_dir = db_dir
        if type(table_path) == str and os.path.splitext(table_path)[-1] == '.json':
            tables_list = json.load(open(table_path, 'r'))
            self.tables = { db['db_id']: db for db in tables_list }
        elif type(table_path) == str:
            self.tables = pickle.load(open(table_path, 'rb'))
        else: self.tables = table_path
        db_contents = load_db_contents(self.db_dir)
        self.entries = self._load_db_entries(db_contents)
        self.contents = self._load_db_contents(db_contents)
        rouge = Rouge(metrics=["rouge-1"])
        self.rouge_score = lambda pred, ref: rouge.get_scores(' '.join(list(pred)), ' '.join(list(ref)))[0]['rouge-1']
        self.stopwords = STOPWORDS | set(QUOTATION_MARKS + list('，。！￥？（）《》、；·…' + string.punctuation))

    def _load_db_entries(self, db_contents):
        """ Given db_id, list of reference (column id, value) pairs, e.g. [ (1, '24'), (4, '36'), ... ]
        extract the list of all distinct entries (rows)
        """
        entries = {}
        for db_id in self.tables:
            db = self.tables[db_id]
            entries[db_id] = db_contents[db['db_id']][db['table_names'][0]]['cell']
        return entries

    def _load_db_contents(self, db_contents):
        """ Given db_id and column id, extract the list of all distinct cell values, e.g. self.contents[db_id][col_id]
        """
        contents = {}
        for db_id in self.tables:
            db = self.tables[db_id]
            contents[db_id] = db['cells'] if 'cells' in db else extract_db_contents(db_contents, db, strip=False)
        return contents
    
    def extract_values_with_constraints(self, db_id, constraints: list, target_id):
        entries = self.entries[db_id]
        for col_id, value in constraints:
            entries = list(filter(lambda row: row[col_id] == value, entries))
        return set([row[target_id] for row in entries])

    def postprocess_value(self, sqlvalue, db, entry):
        """
        sqlvalue: class SQLValue, the current condition value to be resolved
        db: dict, tables
        entry: Example of the current sample
        """
        vc, state = sqlvalue.candidate, sqlvalue.state
        db_id, col_id = db['db_id'], state.col_id
        cmp_op, col_name, col_type = state.cmp_op, db['column_names'][col_id][1], db['column_types'][col_id]
        cell_values = self.contents[db_id][col_id]

        if type(vc) == int: # reserved vocab
            reserved_vocab = SelectValueAction.vocab('nl2sql').id2word
            value_str = reserved_vocab[vc]
            if value_str in ['0', '1']:
                output_str = str(value_str)
            elif col_type == 'real':
                output_str = '1' if value_str == '是' else '0'
            elif value_str == '是':
                filtered_values = [c for c in cell_values if '是' in c or '有' in c or '√' in c]
                output_str = filtered_values[0] if len(filtered_values) > 0 else '是'
            else:
                filtered_values = [c for c in cell_values if '否' in c or '不' in c or '无' in c or '未' in c or '免' in c or '没有' in c or 'no' in c or c == '/']
                output_str = filtered_values[0] if len(filtered_values) > 0 else value_str
            return '"' + output_str + '"'

        assert isinstance(vc, ValueCandidate)
        value_str = vc.matched_cased_value
        value_str= re.sub(r'([a-zA-Z0-9])\s+([a-zA-Z0-9])', lambda match_obj: match_obj.group(1) + PLACEHOLDER + match_obj.group(2), value_str)
        value_str = re.sub(r'\s+', '', value_str).replace(PLACEHOLDER, ' ')
        output_str = ''

        if col_type == 'real': # numbers
            value_str = re.sub(r'(千米|千瓦|千克|千斤|千卡)', '', value_str, flags=re.I) # remove some metrics
            date_num = parse_date(value_str)
            if date_num is not None: return '"' + date_num + '"'

            plus_one = 1 if '排名' in col_name and cmp_op == '<' else 0
            number_match = re.search(rf'(百分之?|[0-9\.点元块毛角\-负{ZH_NUMBER}{ZH_UNIT}%])+', value_str)
            if number_match:
                span = number_match.group().rstrip('.')
                span = re.sub(r'(百分之?|%)', '', span) # directly remove percentage
                if re.search(r'^[百千万亿]+$', span): span = '一' + span
                is_rmb_jiao = (re.search(r'[角毛]', span) is not None and re.search(r'[元块]', span) is None)
                if re.search(r'[元块角毛]', span): span = re.sub(r'[元块角毛]', '点', span.rstrip('元块角毛'))

                metric_in_brackets = re.search(r'([\(（](.+?)/.*?[\)）]|[\(（](.+?)[\)）])', col_name)
                divider = None
                if metric_in_brackets:
                    metric = re.search(r'(万亿|千亿|百亿|十亿|千万|百万|十万|亿|万|千字)', metric_in_brackets.group(1))
                    if metric:
                        divider = np.prod([ZH_UNIT_MAPPING[c] for c in metric.group(1) if c in ZH_UNIT_MAPPING])

                if is_number(span):
                    span = int(float(span)) if is_int(span) else float(span)
                    if divider and is_int(span) and span % divider == 0:
                        span = int(span // divider)
                    span = span + plus_one
                    span = str(float(Decimal(span) * Decimal('0.1'))) if is_rmb_jiao else str(span)
                    return '"' + span + '"'

                number_word_pattern = re.search(rf'([\d\.]+)([百千万亿]+)', span)
                if number_word_pattern:
                    number, unit = number_word_pattern.groups()
                    if is_number(number):
                        number = int(float(number)) if is_int(number) else float(number)
                        metric = np.prod([ZH_UNIT_MAPPING[c] for c in unit if c in ZH_UNIT_MAPPING])
                        if metric and divider:
                            number = Decimal(str(number)) * Decimal(str(metric)) / Decimal(str(divider))
                            number = int(float(number)) if is_int(number) else float(number)
                            return '"' + str(number) + '"'
                        elif metric:
                            full_number = float(Decimal(str(number)) * Decimal(str(metric)))
                            if full_number <= 1e4 and is_int(full_number):
                                return '"' + str(int(full_number)) + '"'
                        return '"' + str(number) + '"'

                try: # try parsing words into number
                    number = parse_number2word(span)
                    number = int(float(number)) if is_int(number) else float(number)
                    if divider and is_int(number) and number % divider == 0:
                        number = int(number // divider)
                    if is_int(number) and number < 1e5: span = number
                    else:
                        number_ = parse_number2word(span.rstrip('万亿'))
                        span = int(float(number_)) if is_int(number_) else float(number_)
                    span = span + plus_one
                    span = str(float(Decimal(span) * Decimal('0.1'))) if is_rmb_jiao else str(span)
                    return '"' + str(span) + '"'
                except: pass
            return '"' + value_str + '"'
        else: # string text
            normed_cell_values = [(cvid, re.sub(r'\s+', ' ', NUM_NORM(c))) for cvid, c in enumerate(cell_values) if c.lower().strip() != '.' and 0 < len(c.lower().strip()) < 80]
            if len(normed_cell_values) == 0: return '"' + value_str + '"'
            if len(normed_cell_values) == 1: return '"' + cell_values[normed_cell_values[0][0]] + '"'

            def try_abbreviation(val):
                val = val.lower()
                for s in ABBREV_SET:
                    if val in s:
                        for c in cell_values:
                            if c.lower() in s:
                                nonlocal output_str
                                output_str = c
                                return True
                return False

            def try_parsing_date(val):
                result = None
                val = val.lower().replace('情人节', '2月14日').replace('元旦', '1月1日')
                match = re.search(rf'([\d{ZH_NUMBER}]+)[年/\.\-].*?([\d{ZH_NUMBER}]+)[月/\.\-].*?([\d{ZH_NUMBER}]+)', val)
                if match:
                    try:
                        y, m, d = match.groups()
                        y, m, d = ZH_WORD2NUM(y), ZH_WORD2NUM(m), ZH_WORD2NUM(d)
                        if y < 30: y = 2000 + y
                        elif y < 100: y = 1900 + y
                        if len(str(m)) == 1: m = '0?' + str(m)
                        if len(str(d)) == 1: d = '0?' + str(d)
                        ymd = str(y) + r'[年/\.\-]' + str(m) + r'[月/\.\-]' + str(d)
                        for c in cell_values:
                            if re.search(ymd, c.strip().lower()):
                                result = c
                                break
                    except: return False
                match = re.search(rf'([\d{ZH_NUMBER}]+)年.*?([\d{ZH_NUMBER}]+)月', val)
                if result is None and match:
                    try:
                        y, m = match.groups()
                        y, m = ZH_WORD2NUM(y), ZH_WORD2NUM(m)
                        if y < 30: y = 2000 + y
                        elif y < 100: y = 1900 + y
                        if len(str(m)) == 1: m = '0?' + str(m)
                        ym = r'^' + str(y) + r'[年/\.\-]' + str(m) + r'$'
                        for c in cell_values:
                            if re.search(ym, c.strip().lower()):
                                result = c
                                break
                    except: return False
                if result is not None:
                    nonlocal output_str
                    output_str = result
                    return True
                return False

            def try_parsing_digits_sensitive(val):
                if re.search(r'[^\d\.+\-×/xy]', value_str, flags=re.I): return False
                nonlocal output_str
                output_str = val
                return True

            def try_search_database(val):
                candidates = []
                for c in cell_values:
                    if val.strip() == c.strip():
                        candidates.append((c, 1.0))
                    elif val.strip() == c.strip():
                        candidates.append((c, 0.75))
                    elif val.strip().lower() == c.strip().lower():
                        candidates.append((c, 0.5))
                    elif val.strip().lower().replace('-', '') == c.strip().lower().replace('-', ''):
                        candidates.append((c, 0.25))
                if len(candidates) > 0:
                    c = sorted(candidates, key=lambda x: - x[1])[0][0]
                    nonlocal output_str
                    output_str = c
                    return True
                return False

            if try_search_database(value_str): return '"' + output_str + '"'
            elif try_abbreviation(value_str): return '"' + output_str + '"'
            elif try_parsing_digits_sensitive(value_str): return '"' + output_str + '"'
            elif try_parsing_date(value_str): return '"' + output_str + '"'

            idx = self.select_cell_idx_with_heuristic_rules(value_str, normed_cell_values)
            if idx is not None: return '"' + cell_values[idx] + '"'

        return '"' + value_str + '"'


    def select_cell_idx_with_heuristic_rules(self, value_str, cell_values):
        value_str = NUM_NORM(value_str)
        scores = [self.rouge_score(value_str, c) for _, c in cell_values]
        max_score = max(scores, key=lambda s: (s['f'], s['p']))
        max_score = (max_score['f'], max_score['p'])

        if max_score[0] > 0: # at least one char matching
            max_idxs = [idx for idx, score in enumerate(scores) if float_equal(score['f'], max_score[0]) and float_equal(score['p'], max_score[1])]
            if len(max_idxs) == 1:
                return cell_values[max_idxs[0]][0]

            # use editdistance to resolve ambiguity
            candidates = [cell_values[idx][1] for idx in max_idxs]
            most_similar = process.extractOne(value_str, candidates)
            candidate = most_similar[0]
            return cell_values[max_idxs[candidates.index(candidate)]][0]
        return None


    def select_cell_idx_with_bert_similarity(self, value_str, cells):
        raise NotImplementedError


if __name__ == '__main__':
    from utils.constants import DATASETS

    def _equal(p, g):
        p, g = p.strip('"'), g.strip('"')
        if is_number(p) and is_number(g):
            return float(p) == float(g)
        return p == g

    data_dir = DATASETS['nl2sql']['data']
    table_path = os.path.join(data_dir, 'tables.bin')
    db_dir = DATASETS['nl2sql']['database']
    processor = ValueProcessor(table_path=table_path, db_dir=db_dir)

    dataset = pickle.load(open(os.path.join(data_dir, 'train.lgesql.bin'), 'rb'))
    # dataset = pickle.load(open(os.path.join(data_dir, 'dev.lgesql.bin'), 'rb'))
    test_samples = [ex['values'] for ex in dataset]

    instance_correct, real_count, real_correct, text_count, text_correct = 0, 0, 0, 0, 0
    for entry, sqlvalues in zip(dataset, test_samples):
        db, flag = processor.tables[entry['db_id']], True
        for sqlvalue in sqlvalues:
            col_id = sqlvalue.state.col_id
            col_type = db['column_types'][col_id]
            if col_type == 'real': real_count += 1
            else: text_count += 1
            pred_value = processor.postprocess_value(sqlvalue, db, entry)
            if _equal(pred_value, sqlvalue.real_value):
                if col_type == 'real': real_correct += 1
                else: text_correct += 1
            else:
                flag, col_name, candidate = False, db['column_names'][col_id][1], sqlvalue.candidate
                matched_value = candidate.matched_cased_value if isinstance(candidate, ValueCandidate) else SelectValueAction.vocab('nl2sql').id2word[candidate]
                print('Column %s[%s]: Gold/Match/Pred value: %s/%s/%s' % (col_name, col_type, sqlvalue.real_value, matched_value, pred_value))
        if flag: instance_correct += 1
        else:
            print('Question[%s]: [%s]' % (entry['question_id'], '|'.join(entry['cased_question_toks'])))
            print('SQL: %s\n' % (entry['query']))

    print('Real values postprocess accuracy is %.4f' % (real_correct / float(real_count)))
    print('Text values postprocess accuracy is %.4f' % (text_correct / float(text_count)))
    print('Samples postprocess accuracy is %.4f' % (instance_correct / float(len(dataset))))
