#coding=utf8
import re, os, sys, copy, math
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import editdistance as edt
from decimal import Decimal
from itertools import combinations
from asdl.transition_system import SelectValueAction
from preprocess.process_utils import ValueCandidate, State, SQLValue
from preprocess.process_utils import is_number, is_int, float_equal, quote_normalization
from preprocess.process_utils import ZH_NUM2WORD, ZH_WORD2NUM, ZH_NUMBER, ZH_UNIT, DIGIT_ALIAS, UNIT_OP, AGG_OP

ABBREV_MAPPING = {
    '扌': '提手旁',
    'aaaa级': '4a级',
    'aaaaa级': '5a级',
    '汉语': '中文',
    '食草': '吃草',
    '杜,郭': '姓杜和姓郭',
    '皇帝老师': '皇上的老师',
    '世界一流大学': '世界一流',
    '微软游戏工作制': '微软游戏',
    'ios系统': 'ios操作系统',
    '韩国最大的汽车企业': '韩国最大的汽车公司',
    '“1+2”是哥德巴赫猜想研究的丰碑': '哥德巴赫猜想',
    '北京世界园艺博览会': '北京园博会',
    '2017世博会': '2017届',
    '第28届台湾金曲奖': '第28届',
    '第一届中国国际进口博览会': '第一届',
    '第二届中国国际进口博览会': '第二届进博会',
    '诺贝尔化学奖': '化学奖',
    '诺贝尔文学奖': '文学奖',
    '穿过爱情的漫长旅程:萧红传': '萧红传',
    '北京电影学院': '北电'
}

CMP_OP = ('not_in', 'between', '==', '>', '<', '>=', '<=', '!=', 'in', 'like')
PLACEHOLDER = '|' # | never appears in the dataset

class ValueExtractor():

    def extract_values(self, entry: dict, db: dict, verbose=False):
        """ Extract values(class SQLValue) which will be used in AST construction,
        we need question_toks for comparison, entry for char/word idx mapping.
        The matched_index field in the ValueCandidate stores the index_pair of uncased_question_toks, not question.
        `track` is used to record the traverse clause path for disambiguation.
        """
        result = { 'values': set(), 'entry': entry, 'question_toks': copy.deepcopy(entry['uncased_question_toks']) , 'db': db}
        result = self.extract_values_from_sql(entry['sql'], result, '')
        entry = self.assign_values(entry, result['values'])
        if verbose and len(entry['values']) > 0:
            print('Question:', ' '.join(entry['uncased_question_toks']))
            print('SQL:', entry['query'])
            print('Values:', ' ; '.join([repr(val) for val in entry['values']]), '\n')
        return entry

    def assign_values(self, entry, values):
        # take set because some SQLValue may use the same ValueCandidate
        cased_question_toks = entry['cased_question_toks']
        candidates = set([val.candidate for val in values if isinstance(val.candidate, ValueCandidate)])
        candidates = sorted(candidates)
        offset = SelectValueAction.size('dusql')
        for idx, val in enumerate(candidates):
            val.set_value_id(idx + offset)
            cased_value = ' '.join(cased_question_toks[val.matched_index[0]:val.matched_index[1]])
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

    def extract_values_from_sql(self, sql: dict, result: dict, track: str):
        result = self.extract_values_from_sql_unit(sql, result, track)
        if sql['intersect']:
            result = self.extract_values_from_sql_unit(sql['intersect'], result, track + '->intersect')
        if sql['union']:
            result = self.extract_values_from_sql_unit(sql['union'], result, track + '->union')
        if sql['except']:
            result = self.extract_values_from_sql_unit(sql['except'], result, track + '->except')
        return result

    def extract_values_from_sql_unit(self, sql: dict, result: dict, track: str):
        """ Each entry in values is an object of type SQLValue
        """
        table_units = sql['from']['table_units']
        for t in table_units:
            if t[0] == 'sql':
                result = self.extract_values_from_sql(t[1], result, track + '->from')
        if sql['where']:
            for cond in sql['where']:
                if cond in ['and', 'or']: continue
                result = self.extract_values_from_cond(cond, result, track + '->where')
        if sql['having']:
            for cond in sql['having']:
                if cond in ['and', 'or']: continue
                result = self.extract_values_from_cond(cond, result, track + '->having')
        if sql['limit']:
            result = self.extract_limit_val(sql, result, track + '->limit')
        return result

    def extract_values_from_cond(self, cond, result, track):
        agg_op, cmp_op, val_unit, val1, val2 = cond # no BETWEEN operator, thus no val2
        unit_op, col_id = val_unit[0], val_unit[1][1]
        state = State(track, AGG_OP[agg_op], CMP_OP[cmp_op], UNIT_OP[unit_op], col_id)
        func = self.extract_where_val if track.split('->')[-1] == 'where' else self.extract_having_val

        def extract_val(val):
            if type(val) in [list, tuple]: # value is column, do nothing
                return result
            elif type(val) == dict: # value is SQL
                return self.extract_values_from_sql(val, result, track)
            else: return func(val, result, state)
        return extract_val(val1)

    def extract_where_val(self, val, result, state):
        func = {int: self.extract_where_int_val, str: self.extract_where_string_val, float: self.extract_where_float_val}
        sqlvalue = SQLValue(str(val), state)
        if sqlvalue in result['values']: return result
        else: result['values'].add(sqlvalue)
        val = val if type(val) == str else int(val) if is_int(val) else val
        return func[type(val)](val, result, sqlvalue)


    def extract_where_int_val(self, num, result, sqlvalue):
        question_toks, entry = result['question_toks'], result['entry']
        num, question = int(num), ''.join(question_toks)

        def try_digit_alias(num):
            if num >= 10: return False
            for char in DIGIT_ALIAS(num):
                start_ids = extract_number_occurrences(char, question)
                if len(start_ids) > 0:
                    start_id = start_ids[0]
                    add_value_from_char_idx((start_id, start_id + 1), question_toks, sqlvalue, entry)
                    return True
            return False

        def try_parsing_year(num):
            if not re.search(r'[12]\d\d\d', str(num)): return False
            if num > 2000: num -= 2000
            else: num -= 1900
            match_obj = re.search(fr'{num:d}年?', question)
            if match_obj:
                add_value_from_char_idx((match_obj.start(), match_obj.end()), question_toks, sqlvalue, entry)
                return True
            return False

        def try_parsing_outliers(num):
            if '8.5万亿' in question and float_equal(num, 8.5*1e12):
                span = '8.5万亿'
            elif '一个亿' in question and float_equal(num, 1e8):
                span = '一个亿'
            else: return False
            start_id = question.index(span)
            add_value_from_char_idx((start_id, start_id + len(span)), question_toks, sqlvalue, entry)
            return True

        start_ids = extract_number_occurrences(num, question)
        if len(start_ids) > 0: # need to resolve ambiguity due to multiple matches
            if len(start_ids) > 1 and entry['question_id'] in ['qid009904', 'qid009905', 'qid009906', 'qid014302', 'qid020478']: start_id = start_ids[1]
            else: start_id = start_ids[0]
            add_value_from_char_idx((start_id, start_id + len(str(num))), question_toks, sqlvalue, entry)
        elif try_digit_alias(num): pass
        elif try_number_to_word(num, question, question_toks, sqlvalue, entry): pass
        elif num == 0: add_value_from_reserved("0", sqlvalue)
        elif str(num) in question and question.count(str(num)) == 1:
            num = str(num)
            start_id = question.index(num)
            add_value_from_char_idx((start_id, start_id + len(num)), question_toks, sqlvalue, entry)
        elif try_parsing_height(num, question, question_toks, sqlvalue, entry): pass
        elif try_parsing_year(num): pass
        elif try_parsing_outliers(num): pass
        else:
            raise ValueError('WHERE int value %d is not recognized in the question: %s' % (num, ' '.join(entry['uncased_question_toks'])))
        return result


    def extract_where_float_val(self, num, result, sqlvalue):
        question_toks, entry = result['question_toks'], result['entry']
        question = ''.join(question_toks)

        def try_parsing_outliers(num):
            if float_equal(num, 1.91) and '191' in question: matched_span = '191'
            elif float_equal(num, 1.91) and '一米九一' in question: matched_span = '一米九一'
            elif float_equal(num, 11.2) and '11月20号' in question: matched_span = '11月20号'
            elif float_equal(num, 3.4) and '3小时40分钟' in question: matched_span = '3小时40分钟'
            else: return False
            start_id = question.index(matched_span)
            add_value_from_char_idx((start_id, start_id + len(matched_span)), question_toks, sqlvalue, entry)
            return True

        start_ids = extract_number_occurrences(num, question)
        if len(start_ids) > 0: # luckily, no need to resolve ambiguity
            start_id = start_ids[0]
            add_value_from_char_idx((start_id, start_id + len(str(num))), question_toks, sqlvalue, entry)
        elif try_percentage_variants(num, question, question_toks, sqlvalue, entry): pass
        elif str(num) in question and question.count(str(num)) == 1:
            num = str(num)
            start_id = question.index(num)
            add_value_from_char_idx((start_id, start_id + len(num)), question_toks, sqlvalue, entry)
        elif try_parsing_outliers(num): pass
        else: raise ValueError('WHERE float value %s is not recognized !' % (str(num)))
        return result


    def extract_where_string_val(self, val, result, sqlvalue):
        values, question_toks, entry = result['values'], result['question_toks'], result['entry']
        question = ''.join(question_toks)

        def try_parsing_date(val):
            if val == '2016-09-15-20:00:00':
                match_obj = re.search(r'2016年9月15[号日]晚上?[八8]点', question)
                if match_obj:
                    start_id, end_id = match_obj.start(), match_obj.end()
                    add_value_from_char_idx((start_id, end_id), question_toks, sqlvalue, entry)
                    return True
            ymd = re.match(r'(\d{4})\-(\d{2})\-(\d{2})', val) # year-month-day
            ym = re.match(r'(\d{4})\-(\d{2})', val) # year-month, in dev dataset, 2018年7月 -> 2018-07
            if ymd:
                y, m, d = ymd.group(1), ymd.group(2).lstrip('0'), ymd.group(3).lstrip('0')
                y_cn, m_cn, d_cn = ZH_NUM2WORD(y, 'direct'), ZH_NUM2WORD(m, 'low'), ZH_NUM2WORD(d, 'low')
                y, m, d = y + '|' + y_cn, m + '|' + m_cn, d + '|' + d_cn
                pattern1 = r'({})[年\.\-/](0?{})[月\.\-/](0?{})[日号]?'.format(y, m, d)
                match_obj = re.search(pattern1, question)
                if match_obj:
                    add_value_from_char_idx((match_obj.start(), match_obj.end()), question_toks, sqlvalue, entry)
                    return True
                pattern2 = r'({})[年\.\-/]({})({})'.format(y, m_cn, d_cn)
                match_obj = re.search(pattern2, question)
                if match_obj:
                    add_value_from_char_idx((match_obj.start(), match_obj.end()), question_toks, sqlvalue, entry)
                    return True
            elif ym:
                pattern = r'{}[年\.\-](0?{})月?'.format(ym.group(1), ym.group(2).lstrip('0'))
                match_obj = re.search(pattern, question)
                if match_obj:
                    add_value_from_char_idx((match_obj.start(), match_obj.end()), question_toks, sqlvalue, entry)
                    return True
            return False

        def try_abbreviation_mapping(val):
            if val in ABBREV_MAPPING:
                mapped_val = ABBREV_MAPPING[val]
                if mapped_val in question:
                    start_id = question.index(mapped_val)
                    add_value_from_char_idx((start_id, start_id + len(mapped_val)), question_toks, sqlvalue, entry)
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

        def try_parsing_company(val):
            # companies often use abbreviations in questions, e.g. 深圳腾讯科技有限公司 -> 腾讯公司
            if '公司' in val or '金拱门' in val:
                pos = search_for_longest_substring(question, val)
                if pos[1] - pos[0] >= 2:
                    add_value_from_char_idx((pos[0], pos[1]), question_toks, sqlvalue, entry)
                    return True
            return False

        def try_parsing_province(val):
            # mentioning the city and region may be enough
            match_obj = re.match(r'(.*)省(.*)市', val)
            if match_obj and match_obj.group(2) in question:
                start_id = question.index(match_obj.group(2))
                add_value_from_char_idx((start_id, start_id + len(match_obj.group(2))), question_toks, sqlvalue, entry)
                return True
            match_obj = re.match(r'(.*)市(.*)区', val)
            if match_obj and match_obj.group(2) in question:
                start_id = question.index(match_obj.group(2))
                add_value_from_char_idx((start_id, start_id + len(match_obj.group(2))), question_toks, sqlvalue, entry)
                return True
            return False

        def try_parsing_time(val):
            if val.endswith(':00'):
                time_pattern = fr'(上午|下午|晚上|周[一二三四五六七日])?[0-9{ZH_NUMBER}十]+点(半|[0-9{ZH_NUMBER}十]+分?)?'
                if '~' in val: match_obj = re.search(fr'{time_pattern}[至到~\-]{time_pattern}', question)
                else: match_obj = re.search(time_pattern, question)
                if match_obj:
                    add_value_from_char_idx((match_obj.start(), match_obj.end()), question_toks, sqlvalue, entry)
                    return True
                if val == '09:00:00' and '上午9:00' in question: # two outliers
                    start_id = question.index('上午9:00')
                    add_value_from_char_idx((start_id, start_id + len('上午9:00')), question_toks, sqlvalue, entry)
                    return True
            return False

        val = quote_normalization(val).lower()
        if val in ['是', '否']: add_value_from_reserved(val, sqlvalue)
        elif val in question: # luckily, no need for disambiguation
            start_id = question.index(val)
            add_value_from_char_idx((start_id, start_id + len(val)), question_toks, sqlvalue, entry)
        elif val.startswith('item_'):
            mapped_val = entry['item_mapping'][val]
            if mapped_val in question:
                start_id = question.index(mapped_val)
                add_value_from_char_idx((start_id, start_id + len(mapped_val)), question_toks, sqlvalue, entry)
            else:
                flag = self.use_extracted_values(sqlvalue, values)
                if not flag: raise ValueError('Item_xxx not found in WHERE clause !')
        elif try_parsing_date(val): pass
        elif try_abbreviation_mapping(val): pass
        elif fuzzy_match(val): pass
        elif try_parsing_company(val): pass
        elif try_parsing_time(val): pass
        elif try_parsing_province(val): pass
        else: raise ValueError('WHERE string value %s is not recognized in the question: %s' % (val, ' '.join(entry['uncased_question_toks'])))
        return result


    def extract_having_val(self, num, result, state):
        """ Having values are all numbers in the dataset, decision procedure:
        1. str(val) is in question, create a new entry in the set ``values``
        2. the mapped word of val is in question_toks, use the word as matched value candidate
        3. try parsing percentage and height
        4. num == 1, directly add pre-defined value
        """
        assert is_number(num)
        sqlvalue = SQLValue(str(num), state)
        values, question_toks, entry = result['values'], result['question_toks'], result['entry']
        if sqlvalue in values: return values, question_toks
        else: values.add(sqlvalue)
        num = int(num) if is_int(num) else float(num)
        question = ''.join(question_toks)

        def try_digit_alias(num):
            if num >= 10: return False
            # for integer number less than 10
            for char in DIGIT_ALIAS(int(num)):
                match_obj = re.search(r'(%s)[^十百千万亿]' % (char), question)
                if match_obj: # no need to resolve ambiguity
                    char_id = match_obj.start(1)
                    add_value_from_char_idx((char_id, char_id + 1), question_toks, sqlvalue, entry)
                    return True
            return False

        # deal with float number
        if not is_int(num):
            num = str(num)
            start_ids = extract_number_occurrences(num, question)
            if len(start_ids) == 1:
                start_id = start_ids[0]
                add_value_from_char_idx((start_id, start_id + len(num)), question_toks, sqlvalue, entry)
            elif try_percentage_variants(num, question, question_toks, sqlvalue, entry): pass
            elif float_equal(float(num), 0.5) and '一半' in question:
                start_id = question.index('一半')
                add_value_from_char_idx((start_id, start_id + 2), question_toks, sqlvalue, entry)
            else: raise ValueError('Unresolved HAVING float value %s !' % (num))
            return result

        # deal with integers
        start_ids = extract_number_occurrences(num, question, exclude_prefix='A-Za-z', exclude_suffix='A-Za-z')
        if len(start_ids) > 0: # for multiple occurrences, directly use the first one is ok
            start_id = start_ids[0]
            add_value_from_char_idx((start_id, start_id + len(str(num))), question_toks, sqlvalue, entry)
        elif try_digit_alias(num): pass
        elif try_number_to_word(num, question, question_toks, sqlvalue, entry): pass
        elif try_percentage_variants(num, question, question_toks, sqlvalue, entry): pass
        elif str(num) in question and question.count(str(num)) == 1:
            num = str(num)
            start_id = question.index(num)
            add_value_from_char_idx((start_id, start_id + len(num)), question_toks, sqlvalue, entry)
        elif try_parsing_height(num, question, question_toks, sqlvalue, entry): pass
        elif num == 1: add_value_from_reserved("1", sqlvalue)
        else: raise ValueError('Unresolved HAVING int value %s !' % (num))
        return result


    def extract_limit_val(self, limit, result, track):
        """ Decision procedure for LIMIT value (type int): luckily, only integers less or equal than 10
        1. num != 1 and str(num) directly appears in question_toks:
            a. no duplicate occurrence, add a new entry in the set ``values``
            b. try to resolve the ambiguity by prompt words such as '哪', '前', '后', '最'
            c. use the first one, o.w.
        2. num != 1 and the mapped char of num in question_toks/question, use that char as matched value candidate
        3. num == 1, directly use reserved vocabulary
        Each time when some span is matched, update question_toks to prevent further match
        """
        num = limit['limit']
        state = State(track, 'none', '==', 'none', 0)
        sqlvalue = SQLValue(str(num), state)
        values, question_toks, entry = result['values'], result['question_toks'], result['entry']
        if sqlvalue in values: return result
        else: values.add(sqlvalue)
        num, question = int(num), ''.join(question_toks)

        def resolve_limit_ambiguity(start_ids):
            prev_chars = [question[i - 1] if i >= 1 else '' for i in start_ids]
            for prompt in ['哪', '前', '后']:
                if prompt in prev_chars:
                    start_id = start_ids[prev_chars.index(prompt)]
                    break
            else: # use the first occurrence after '最'
                if '最' in question:
                    prompt_idx = question.index('最')
                    indexes = [idx for idx in start_ids if idx > prompt_idx]
                    start_id = indexes[0] if indexes else start_ids[0]
                else: raise ValueError('Cannot resolve ambiguity in LIMIT clause.')
            return start_id

        def try_digit_alias(num):
            if num >= 10: return False
            for char in DIGIT_ALIAS(num):
                start_ids = extract_number_occurrences(char, question)
                if len(start_ids) == 0: continue
                elif len(start_ids) == 1:
                    start_id = start_ids[0]
                    add_value_from_char_idx((start_id, start_id + 1), question_toks, sqlvalue, entry)
                    return True
                else: # resolve ambiguity
                    if '最' in question: # 最多的三个，最少的两个
                        prompt_idx = question.index('最')
                        indexes = [idx for idx in start_ids if idx > prompt_idx]
                        start_id = indexes[0] if indexes else start_ids[0]
                    else: start_id = start_ids[0]
                    add_value_from_char_idx((start_id, start_id + 1), question_toks, sqlvalue, entry)
                    return True
            return False

        if num == 1: # special value LIMIT 1, directly use pre-defined vocabulary
            add_value_from_reserved("1", sqlvalue)
            return result
        start_ids = extract_number_occurrences(num, question)
        if len(start_ids) > 0:
            if len(start_ids) == 1: start_id = start_ids[0]
            else: start_id = resolve_limit_ambiguity(start_ids)
            add_value_from_char_idx((start_id, start_id + len(str(num))), question_toks, sqlvalue, entry)
        elif try_digit_alias(num): pass
        else: raise ValueError('Unresolved LIMIT value %d !' % (num))
        return result


def search_for_longest_substring(question, val):
    # record longest "substring": start and end position
    # "substring" means chars in the question follow the asc order in val, but not necessarily adjacent in val
    longest, start, end = (0, 0), 0, 0
    while end < len(question):
        if question[end] not in val:
            if start != end and end - start > longest[1] - longest[0]:
                longest = (start, end)
            start = end + 1
        end += 1
    if start != end and end - start > longest[1] - longest[0]:
        longest = (start, end)
        return longest
    return longest # if no char in question exists in the val, return (0, 0)


def try_percentage_variants(num, question, question_toks, sqlvalue, entry):
    num_100 = float(Decimal(str(num)) * Decimal('100'))
    if is_int(num_100):
        num_100 = int(num_100)
        nums = [str(num_100), ZH_NUM2WORD(num_100, 'low'), ZH_NUM2WORD(num_100, 'direct')]
    else:
        nums = [str(num_100), str(num_100).replace('.', '点'), ZH_NUM2WORD(num_100, 'low'), ZH_NUM2WORD(num_100, 'direct')]
    num_pattern = '|'.join(sorted(nums, key=lambda x: - len(x))).replace('.', '\.')
    percentage = f'(({num_pattern})%|百分之({num_pattern}))'
    match_obj = re.search(percentage, question)
    if match_obj:
        start_id, end_id = match_obj.start(0), match_obj.end(0)
        add_value_from_char_idx((start_id, end_id), question_toks, sqlvalue, entry)
        return True
    return False


def try_number_to_word(num, question, question_toks, sqlvalue, entry):
    for mode in ['low', 'direct']:
        try:
            word = ZH_NUM2WORD(num, mode)
            if word in question:
                if question.count(word) == 1:
                    start_id = question.index(word)
                    add_value_from_char_idx((start_id, start_id + len(word)), question_toks, sqlvalue, entry)
                    return True
        except: pass
    candidates = []
    pos = [(span.start(0), span.end(0)) for span in re.finditer(r'([0-9\.点%s十百千万亿kw]+)' % (ZH_NUMBER), question)]
    for s, e in pos:
        try:
            word = question[s: e].replace('k', '千').replace('w', '万')
            parsed_num = ZH_WORD2NUM(word)
            if parsed_num == num: # luckily, no need to resolve ambiguity in DuSQL
                candidates.append((s, e))
        except: pass
    if len(candidates) > 0:
        add_value_from_char_idx(candidates[0], question_toks, sqlvalue, entry)
        return True
    candidates = []
    metric_pos = [(span.start(1), span.end(1), span.end(2))
        for span in re.finditer(r'([0-9\.点%s十百千万亿]+)(平方公里|平方千米|千米|千克|千瓦|千卡|公里|公斤|千斤|km|kg)' % (ZH_NUMBER), question)]
    for s, e, m in metric_pos:
        try:
            word = question[s: e]
            parsed_num = ZH_WORD2NUM(word)
            if parsed_num == num:
                candidates.append((s, e))
            elif '平方' in question[e: m] and float_equal(parsed_num * 1e6, num):
                candidates.append((s, m))
            elif float_equal(parsed_num * 1e3, num):
                candidates.append((s, m))
        except: pass
    if len(candidates) > 0: # luckily, no need to resolve ambiguity in DuSQL
        add_value_from_char_idx(candidates[0], question_toks, sqlvalue, entry)
        return True
    return False


def try_parsing_height(num, question, question_toks, sqlvalue, entry):
    pos = [(span.group(0).replace('米', '点'), span.start(), span.end()) for span in re.finditer(fr'[1-9{ZH_NUMBER}]米[0-9{ZH_NUMBER}]+', question)]
    candidates = []
    for val, s, e in pos:
        try:
            val = ZH_WORD2NUM(val)
            if float_equal(num, val * 100) or float_equal(num, val):
                candidates.append((s, e))
        except: pass
    if len(candidates) == 1:
        add_value_from_char_idx(candidates[0], question_toks, sqlvalue, entry)
        return True
    return False


def extract_number_occurrences(num, question, exclude_prefix='', exclude_suffix=''):
    """ Extract all occurrences of num in the questioin, return the start char ids.
    But exclude those with specified prefixes or suffixes. In these cases, num may only be part of the exact number, e.g. 10 in 100.
    """
    pos = [(span.start(0), span.end(0)) for span in re.finditer(str(num), question)]
    char_ids = []
    for s, e in pos:
        if s > 0 and re.search(fr'[0-9\._{ZH_NUMBER}{ZH_UNIT}{exclude_prefix}每年月]', question[s - 1]): continue
        if e < len(question) - 1 and re.search(fr'千米|千克|千瓦|千卡|千斤', question[e:e+2]):
            char_ids.append(s)
            continue
        if e < len(question) and re.search(fr'[0-9\._%{ZH_NUMBER}{ZH_UNIT}{exclude_suffix}%星月日]', question[e]): continue
        char_ids.append(s)
    return char_ids


def add_value_from_reserved(val, sqlvalue):
    sqlvalue.add_candidate(SelectValueAction.reserved_dusql[val])


def add_value_from_char_idx(index_pairs, question_toks, sqlvalue, entry):
    start_id, end_id = index_pairs
    start = entry['char2word_id_mapping'][start_id]
    end = entry['char2word_id_mapping'][end_id - 1] + 1
    value = ' '.join(question_toks[start: end])
    candidate = ValueCandidate(matched_index=(start, end), matched_value=value)
    sqlvalue.add_candidate(candidate)
    question_toks[start: end] = [PLACEHOLDER * len(question_toks[idx]) for idx in range(start, end)]
    return question_toks