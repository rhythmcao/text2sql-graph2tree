#coding=utf8
import re, copy, math
import editdistance as edt
from itertools import combinations
from asdl.transition_system import SelectValueAction
from preprocess.process_utils import is_number, is_int, UNIT_OP, AGG_OP
from preprocess.process_utils import ValueCandidate, State, SQLValue, ZH_NUM2WORD, DIGIT_ALIAS
from preprocess.process_utils import float_equal, ZH_NUM2WORD, ZH_WORD2NUM, ZH_NUMBER, ZH_UNIT

CMP_OP = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')

PLACEHOLDER = '|'

ABBREV_SET = [
    set({'spring', '春天', '春季'}),
    set({'fall', 'autumn', '秋天', '秋季'}),
    set({'f', 'female', 'miss', '女性', '女士', '女人', '女'}),
    set({'m', 'male', '男性', '男士', '男人', '男'}),
    set({'本科', '学士'}),
    set({'运维职员', 'it员工'}),
    set({'199', '90年代'}),
    set({'english', '英文', '英语'}),
    set({'bangla', '孟加拉语'}),
    set({'indiana', '印第安纳州'}),
    set({'march', '三月'}),
    set({'dvd驱动器', 'dvd播放机'})
]

def normalize_string_val(v):
    return v.strip('"').strip("'").strip('%').strip(':').strip('/')


class ValueExtractor():

    def extract_values(self, entry: dict, db: dict, verbose=False):
        """ Extract values(class SQLValue) which will be used in AST construction
        """
        result = {
            'question_toks': copy.deepcopy(entry['uncased_question_toks']),
            'values': set(), 'db': db, 'entry': entry
        }
        result = self.extract_values_from_sql(entry['sql'], result, track='')
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
        offset = SelectValueAction.size('cspider')
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
        old_candidates = set([v.candidate for v in values if v.real_value.replace('%', '') == sqlvalue.real_value.replace('%', '')
            and v.candidate is not None])
        if len(old_candidates) > 0:
            if len(old_candidates) > 1:
                print('Multiple candidate values which can be used for %s ...' % (sqlvalue.real_value))
            sqlvalue.add_candidate(list(old_candidates)[0])
            return True
        return False

    def extract_values_from_sql(self, sql: dict, result: dict, track: str = ''):
        result = self.extract_values_from_sql_unit(sql, result, track)
        if sql['intersect']:
            result = self.extract_values_from_sql_unit(sql['intersect'], result, track + '->intersect')
        if sql['union']:
            result = self.extract_values_from_sql_unit(sql['union'], result, track + '->union')
        if sql['except']:
            result = self.extract_values_from_sql_unit(sql['except'], result, track + '->except')
        return result

    def extract_values_from_sql_unit(self, sql: dict, result: dict, track: str = ''):
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
            result = self.extract_limit_val(sql['limit'], result, track + '->limit')
        return result

    def extract_values_from_cond(self, cond, result, track):
        _, cmp_op, val_unit, val1, val2 = cond
        func = self.extract_where_val if track.split('->')[-1] == 'where' else self.extract_having_val

        def extract_val(val):
            if type(val) in [list, tuple]: # value is column, do nothing
                return result
            elif type(val) == dict: # value is SQL
                return self.extract_values_from_sql(val, result, track)
            else:
                # column1 +/-/* column2, ignore column2
                agg_op, unit_op, col_id = val_unit[1][0], val_unit[0], val_unit[1][1]
                state = State(track, AGG_OP[agg_op], CMP_OP[cmp_op], UNIT_OP[unit_op], col_id)
                return func(val, result, state)

        result = extract_val(val1)
        if val2 is None: return result
        return extract_val(val2)

    def extract_where_val(self, val, result, state):
        func = {int: self.extract_where_int_val, str: self.extract_where_string_val, float: self.extract_where_float_val}
        sqlvalue = SQLValue(str(val), state)
        if sqlvalue in result['values']: return result
        else: result['values'].add(sqlvalue)
        val = str(val) if type(val) == str else int(float(val)) if is_int(val) else float(val)
        return func[type(val)](val, result, sqlvalue)


    def extract_where_int_val(self, num, result, sqlvalue):
        values, question_toks, entry = result['values'], result['question_toks'], result['entry']
        num, question = int(num), ''.join(question_toks)
        start_ids = extract_number_occurrences(num, question)

        def try_digit_alias(num):
            if num >= 10: return False
            for char in DIGIT_ALIAS(num):
                start_ids = extract_number_occurrences(char, question)
                if len(start_ids) > 0:
                    start_id = start_ids[0]
                    add_value_from_char_idx((start_id, start_id + 1), question_toks, sqlvalue, entry)
                    return True
            return False

        if len(start_ids) > 0:
            start_id = start_ids[0] # for multiple occurrences, use the first one according to position order
            add_value_from_char_idx((start_id, start_id + len(str(num))), question_toks, sqlvalue, entry)
        elif try_digit_alias(num): pass
        elif try_number_to_word(num, question, question_toks, sqlvalue, entry): pass
        elif num in [0, 1]:
            col_id = sqlvalue.state.col_id
            if result['db']['column_types'][col_id] == 'boolean':
                val = 'true' if num == 1 else 'false'
                add_value_from_reserved(val, sqlvalue)
            else: add_value_from_reserved(str(num), sqlvalue)
        elif not self.use_extracted_values(sqlvalue, values):
            raise ValueError('WHERE int value %s not found !' % (num))
        return result


    def extract_where_float_val(self, num, result, sqlvalue):
        question_toks, entry = result['question_toks'], result['entry']
        num, question = float(num), ''.join(question_toks)
        start_ids = extract_number_occurrences(num, question)
        if len(start_ids) > 0:
            start_id = start_ids[0]
            add_value_from_char_idx((start_id, start_id + len(str(num))), question_toks, sqlvalue, entry)
        else: raise ValueError('WHERE float value %s not found !' % (num))
        return result


    def extract_where_string_val(self, val, result, sqlvalue):
        values, question_toks, entry = result['values'], result['question_toks'], result['entry']
        val, question = normalize_string_val(val.lower()), ''.join(question_toks)

        def process_special_string(val):
            if val in ['null', 'none', '空']:
                add_value_from_reserved("null", sqlvalue)
            elif val in ['y', 'yes', 't', 'true', '确定']:
                add_value_from_reserved("true", sqlvalue)
            elif val in ['n', 'no', 'false', '否定']: # not include 'f' -> female in some examples
                add_value_from_reserved("false", sqlvalue)
            else: return False
            return True

        def process_date(val):
            match_obj = re.search(r'(\d{4})-(\d{2})-(\d{2})', val)
            if match_obj:
                year, month, day = match_obj.group(1), match_obj.group(2), match_obj.group(3)
                span = year + '年' + month.lstrip('0') + '月' + day.lstrip('0') + '日'
                if span in question:
                    start_id = question.index(span)
                    add_value_from_char_idx((start_id, start_id + len(span)), question_toks, sqlvalue, entry)
                    return True
            return False

        def process_abbreviation(val):
            for s in ABBREV_SET:
                if val in s:
                    for cand in s:
                        if cand in question:
                            start_id = question.index(cand)
                            add_value_from_char_idx((start_id, start_id + len(cand)), question_toks, sqlvalue, entry)
                            return True
                    break
            return False

        def process_numbers(val):
            if not is_number(val): return False
            num = int(val) if is_int(val) else float(val)
            return try_number_to_word(num, question, question_toks, sqlvalue, entry)

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

        if val in question:
            start_id = question.index(val)
            add_value_from_char_idx((start_id, start_id + len(val)), question_toks, sqlvalue, entry)
        elif process_special_string(val): pass
        elif process_date(val): pass
        elif process_abbreviation(val): pass
        elif process_numbers(val): pass
        elif fuzzy_match(val): pass
        elif not self.use_extracted_values(sqlvalue, values):
            raise ValueError('WHERE string value %s not found in %s!' % (val, question))
        return result


    def extract_having_val(self, num, result, state):
        """ Having values are all numbers in the training dataset, decision procedure:
        1. str(val) is in question_toks, create a new entry in the set ``values``
        2. the mapped word of val is in question_toks, use the word as matched value candidate
        3. num == 1 and some word can be viewed as an evidence of value 1, use that word as candidate
        4. num == 1, directly add pre-defined value
        5. o.w. try to retrieve value candidate from already generated values
        """
        sqlvalue = SQLValue(str(num), state)
        values, question_toks, entry = result['values'], result['question_toks'], result['entry']
        assert is_number(num)
        num = int(num) if is_int(num) else float(num)
        question = ''.join(question_toks)
        if sqlvalue in values: return result
        else: values.add(sqlvalue)
        
        def try_digit_alias(num):
            if num >= 10: return False
            for char in DIGIT_ALIAS(num):
                start_ids = extract_number_occurrences(char, question)
                if len(start_ids) == 0: continue
                elif len(start_ids) == 1:
                    start_id = start_ids[0]
                    add_value_from_char_idx((start_id, start_id + 1), question_toks, sqlvalue, entry)
                    return True
                else: # resolve ambiguity, directly use the last one after analyze the outliers
                    start_id = start_ids[-1]
                    add_value_from_char_idx((start_id, start_id + 1), question_toks, sqlvalue, entry)
                    return True
            return False

        start_ids = extract_number_occurrences(num, question)
        if len(start_ids) > 0:
            start_id = start_ids[-1]
            add_value_from_char_idx((start_id, start_id + len(str(num))), question_toks, sqlvalue, entry)
        elif is_int(num) and try_digit_alias(num): pass
        elif try_number_to_word(num, question, question_toks, sqlvalue, entry): pass
        elif num == 0 or num == 1: add_value_from_reserved(str(num), sqlvalue)
        else:
            if not self.use_extracted_values(sqlvalue, values):
                raise ValueError('HAVING value %s can not be recognized!' % (num))
        return result

    def extract_limit_val(self, num, result, track):
        """ Decision procedure for LIMIT value (type int):
        1. num == 1, directly add pre-defined value
        2. directly in question, add a new entry in the set ``values``
        3. the mapped word of num in question, use the word as matched value candidate
        4. try to retrieve value candidate from already generated values
        Each time when some span is matched, update question_toks to prevent further match
        """
        state = State(track, 'none', '=', 'none', 0)
        values, question_toks, entry = result['values'], result['question_toks'], result['entry']
        sqlvalue = SQLValue(str(num), state)
        num, question = str(num), ''.join(question_toks)
        if sqlvalue in values: return result
        else: values.add(sqlvalue)

        if num == '1':
            add_value_from_reserved("1", sqlvalue)
        elif num in question and question.count(num) == 1:
            start_id = question.index(num)
            add_value_from_char_idx((start_id, start_id + len(num)), question_toks, sqlvalue, entry)
        elif ZH_NUM2WORD(int(num), 'low') in question:
            span = ZH_NUM2WORD(int(num), 'low')
            start_id = question.index(span)
            add_value_from_char_idx((start_id, start_id + len(span)), question_toks, sqlvalue, entry)
        elif int(num) == 2 and '两' in question:
            start_id = question.index('两')
            add_value_from_char_idx((start_id, start_id + 1), question_toks, sqlvalue, entry)
        else:
            if not self.use_extracted_values(sqlvalue, values):
                raise ValueError('LIMIT value %s can not be recognized!' % (num))
        return result


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
        if e < len(question) and re.search(fr'[0-9\._%{ZH_NUMBER}{ZH_UNIT}{exclude_suffix}%月日]', question[e]): continue
        char_ids.append(s)
    return char_ids


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


def add_value_from_reserved(val, sqlvalue):
    sqlvalue.add_candidate(SelectValueAction.reserved_cspider[val])

def add_value_from_char_idx(index_pairs, question_toks, sqlvalue, entry):
    start_id, end_id = index_pairs
    start = entry['char2word_id_mapping'][start_id]
    end = entry['char2word_id_mapping'][end_id - 1] + 1
    value = ' '.join(question_toks[start: end])
    candidate = ValueCandidate(matched_index=(start, end), matched_value=value)
    sqlvalue.add_candidate(candidate)
    question_toks[start: end] = [PLACEHOLDER * len(question_toks[idx]) for idx in range(start, end)]
    return question_toks