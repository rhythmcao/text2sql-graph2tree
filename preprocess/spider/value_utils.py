#coding=utf8
import re
import editdistance as edt
from itertools import combinations
from asdl.transition_system import SelectValueAction
from preprocess.process_utils import is_number, is_int, UNIT_OP, AGG_OP
from preprocess.process_utils import ValueCandidate, State, SQLValue

CMP_OP = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')

PLACEHOLDER = '##none##'

ABBREV_SET = [
    set({'right / r', 'right', 'r'}),
    set({'left / l', 'left', 'l'}),
    set({'uk', 'united kingdom', 'british'}),
    set({'us', 'usa', 'united states', 'united state', 'america'}),
    set({'f', 'female', 'girl', 'woman', 'females', 'girls', 'women'}),
    set({'m', 'male', 'boy', 'man', 'males', 'boys', 'men'}),
    set({'italian', 'italy'}),
    set({'french', 'france'}),
    set({'polish', 'poland'}),
    set({'la', 'los angeles', 'louisiana'}),
    set({'comp. sci.', 'computer science'}),
    set({'assistant professors', 'assistant professor', 'asstprof'}),
    set({'a puzzling pattern', 'a puzzling parallax'})
]

n2w1 = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
n2w2 = ['zeroth', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth']
n2w3 = ['', 'once', 'twice', 'thrice']
n2m1 = ['', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
n2m2 = ['', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']


def remove_names(question_toks):
    # first and last may influence the value extraction
    question = ' '.join(question_toks)
    replacement = [
        'first , middle , and last name', 'first , middle , last name', 'first , middle and last name',
        'first and last name', 'first , last name', 'first name', 'last name'
    ]
    replace_func = lambda span: ' '.join([PLACEHOLDER] * len(span.split(' ')))
    for k in replacement:
        question = question.replace(k, replace_func(k))
    return question.split(' ')


def normalize_string_val(v):
    return v.strip('"').strip("'").strip('%').strip(':').strip('/')


class ValueExtractor():

    def extract_values(self, entry: dict, db: dict, verbose=False):
        """ Extract values(class SQLValue) which will be used in AST construction
        """
        question_toks = remove_names(entry['uncased_question_toks'])
        sqlvalues, _ = self.extract_values_from_sql(entry['sql'], set(), question_toks, db, '')
        entry = self.assign_values(entry, sqlvalues)
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
        offset = SelectValueAction.size('spider')
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

    def extract_values_from_sql(self, sql: dict, values: set, question_toks: list, db: dict, track: str = ''):
        values, question_toks = self.extract_values_from_sql_unit(sql, values, question_toks, db, track)
        if sql['intersect']:
            values, question_toks = self.extract_values_from_sql_unit(sql['intersect'], values, question_toks, db, track + '->intersect')
        if sql['union']:
            values, question_toks = self.extract_values_from_sql_unit(sql['union'], values, question_toks, db, track + '->union')
        if sql['except']:
            values, question_toks = self.extract_values_from_sql_unit(sql['except'], values, question_toks, db, track + '->except')
        return values, question_toks

    def extract_values_from_sql_unit(self, sql: dict, values: set, question_toks: list, db: dict, track: str = ''):
        """ Each entry in values is an object of type SQLValue
        """
        table_units = sql['from']['table_units']
        for t in table_units:
            if t[0] == 'sql':
                values, question_toks = self.extract_values_from_sql(t[1], values, question_toks, db, track + '->from')
        if sql['where']:
            for cond in sql['where']:
                if cond in ['and', 'or']: continue
                values, question_toks = self.extract_values_from_cond(cond, values, question_toks, db, track + '->where')
        if sql['having']:
            for cond in sql['having']:
                if cond in ['and', 'or']: continue
                values, question_toks = self.extract_values_from_cond(cond, values, question_toks, db, track + '->having')
        if sql['limit']:
            values, question_toks = self.extract_limit_val(sql['limit'], values, question_toks, track + '->limit')
        return values, question_toks

    def extract_values_from_cond(self, cond, values, question_toks, db, track):
        _, cmp_op, val_unit, val1, val2 = cond
        # column1 +/-/* column2, ignore column2
        agg_op, unit_op, col_id = val_unit[1][0], val_unit[0], val_unit[1][1]
        state = State(track, AGG_OP[agg_op], CMP_OP[cmp_op], UNIT_OP[unit_op], col_id)
        func = self.extract_where_val if track.split('->')[-1] == 'where' else self.extract_having_val

        def extract_val(val):
            if type(val) in [list, tuple]: # value is column, do nothing
                return values, question_toks
            elif type(val) == dict: # value is SQL
                return self.extract_values_from_sql(val, values, question_toks, db, track)
            else:
                return func(val, values, question_toks, db, state)

        values, question_toks = extract_val(val1)
        if val2 is None: return values, question_toks
        return extract_val(val2)

    def extract_where_val(self, val, values, question_toks, db, state):
        sqlvalue = SQLValue(str(val), state)
        if sqlvalue in values: return values, question_toks
        else: values.add(sqlvalue)
        val = normalize_string_val(val) if not is_number(val) else val

        def exist_mappings(val, question_toks, mappings=[]):
            for m in mappings:
                if 0 <= val < len(m) and m[val] in question_toks:
                    start = question_toks.index(m[val])
                    add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
                    return True
            return False

        def process_possible_years(val):
            if 1900 < val < 2021:
                evidence = ['this year', 'past year', 'last year', 'today']
                question = ' '.join(question_toks)
                for e in evidence:
                    if e in question:
                        start = question[:question.index(e)].count(' ')
                        length = len(e.split())
                        add_value_from_token_idx((start, start + length), question_toks, sqlvalue)
                        return True
                val = str(val)[-2:]
                if val in question_toks: # use last two digit to represent year
                    start = question_toks.index(val)
                    add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
                    return True
                elif val + 's' in question_toks:
                    val = val + 's' # use last two digit plus 's'
                    start = question_toks.index(val)
                    add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
                    return True
            elif len(str(val)) == 3 and str(val).startswith('19'): # LIKE '19x%'
                val = str(val)[2] + '0'
                if val in question_toks:
                    start = question_toks.index(val)
                    add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
                    return True
                elif val + 's' in question_toks:
                    val = val + 's'
                    start = question_toks.index(val)
                    add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
                    return True
            return False

        def process_zero_and_one(val, col_type):
            if not (val == 0 or val == 1): return False
            if col_type == 'boolean':
                add_value_from_reserved('true' if val == 1 else 'false', sqlvalue)
            else:
                add_value_from_reserved(str(val), sqlvalue)
            return True

        def process_negative_value(val):
            if val < 0 and str(-val) in question_toks:
                start = question_toks.index(str(-val)) - 1
                if question_toks[start] == '-':
                    add_value_from_token_idx((start, start + 2), question_toks, sqlvalue)
                    return True
            return False

        def process_suffix(val):
            if len(val) > 1 and val.count(' ') == 0:
                for start, t in enumerate(question_toks):
                    if t.startswith(val):
                        add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
                        return True
            return False

        def process_special_string(val):
            if val in ['null', 'none']:
                add_value_from_reserved("null", sqlvalue)
                return True
            elif val in ['y', 'yes', 't', 'true']:
                add_value_from_reserved("true", sqlvalue)
                return True
            elif val in ['n', 'no', 'false']: # not include 'f' -> female in some examples
                add_value_from_reserved("false", sqlvalue)
                return True
            return False

        def fuzzy_match(val, threshold=0.37):
            length = len(val.split())
            index_pairs = list(filter(lambda x: min([1, length - 1]) <= x[1] - x[0] <= length + 4, combinations(range(len(question_toks) + 1), 2)))
            index_pairs = sorted(index_pairs, key=lambda x: x[1] - x[0])
            dist = []
            val = val.replace(' ', '')
            for start, end in index_pairs:
                span = ''.join(question_toks[start:end])
                score = edt.eval(val, span)
                dist.append(float(score) / len(val))
            min_dist = min(dist)
            if min_dist < threshold:
                index_pair = index_pairs[dist.index(min_dist)]
                add_value_from_token_idx(tuple(index_pair), question_toks, sqlvalue)
                return True
            return False

        def process_date(val):
            result = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', val)
            if result is None: return False
            year, month, day = result.groups()
            month, day = int(month), int(day)
            day_prefix, day_suffix = '0?', '\s?-?\s?(st|nd|rd|th)?'
            month_choice = '(%s|%s)' % (n2m1[month], n2m2[month])
            p1 = r'%s%s%s[\s,]+%s[\s,]+%s' % (day_prefix, day, day_suffix, month_choice, year)
            p2 = r'%s[\s,]+%s%s%s[\s,]+%s' % (month_choice, day_prefix, day, day_suffix, year)
            for p in [p1, p2]:
                result = re.search(p, question)
                if result is not None:
                    span = result.group(0)
                    start = question[:question.index(span)].count(' ')
                    length = len(span.split(' '))
                    add_value_from_token_idx((start, start + length), question_toks, sqlvalue)
                    return True
            return False

        def process_abbreviation(val):
            # special case in the training set
            question = ' '.join(question_toks)
            for s in ABBREV_SET:
                if val in s:
                    for cand in s:
                        if cand.count(' ') > 0 and cand in question:
                            length = len(cand.split())
                            start = question[:question.index(cand)].count(' ')
                            add_value_from_token_idx((start, start + length), question_toks, sqlvalue)
                            return True
                        elif cand.count(' ') == 0 and cand in question_toks:
                            start = question_toks.index(cand)
                            add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
                            return True
                    break
            return False
        if str(val) in question_toks:
            start = question_toks.index(str(val))
            add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
        elif is_int(val): # int
            val = int(val) # 5000.0 -> 5000
            col_type = db['column_types'][state.col_id]
            if str(val) in question_toks:
                start = question_toks.index(str(val))
                add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
            elif exist_mappings(val, question_toks, mappings=[n2w1, n2w2, n2w3, n2m1, n2m2]): pass
            elif process_zero_and_one(val, col_type): pass
            elif process_possible_years(val): pass
            elif process_negative_value(val): pass
            elif format(val, ',') in question_toks: # large numbers separated by ,
                val = format(val, ',')
                start = question_toks.index(val)
                add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
            else:
                if not self.use_extracted_values(sqlvalue, values):
                    raise ValueError('Unable to recognize int number %s in WHERE clause' % (val))
        elif is_number(val): # float
            val = str(val)
            assert val in question_toks, 'Unable to recognize float number %s in WHERE clause' % (val)
            start = question_toks.index(val)
            add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
        else: # string
            val = str(val).lower()
            question = ' '.join(question_toks)
            if val.count(' ') == 0 and val in question_toks:
                start = question_toks.index(val)
                add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
            elif val.count(' ') > 0 and val in question:
                length = len(val.split())
                start = question[:question.index(val)].count(' ')
                add_value_from_token_idx((start, start + length), question_toks, sqlvalue)
            elif process_special_string(val): pass
            elif process_date(val): pass
            elif fuzzy_match(val): pass
            elif process_abbreviation(val): pass
            elif process_suffix(val): pass
            else:
                if not self.use_extracted_values(sqlvalue, values):
                    raise ValueError('Unable to recognize string %s in WHERE clause' % (val))
        return values, question_toks

    def extract_having_val(self, num, values, question_toks, db, state):
        """ Having values are all numbers in the training dataset, decision procedure:
        1. str(val) is in question_toks, create a new entry in the set ``values``
        2. the mapped word of val is in question_toks, use the word as matched value candidate
        3. num == 1 and some word can be viewed as an evidence of value 1, use that word as candidate
        4. num == 1, directly add pre-defined value
        5. o.w. try to retrieve value candidate from already generated values
        """
        sqlvalue = SQLValue(str(num), state)
        assert is_number(num)
        num = int(num) if is_int(num) else float(num)
        if sqlvalue in values: return values, question_toks
        else: values.add(sqlvalue)

        if str(num) in question_toks:
            num = str(num)
            start = question_toks.index(num)
            add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
            return values, question_toks

        assert is_int(num)
        if num < len(n2w1) and n2w1[num] in question_toks:
            word = n2w1[num] # 2 -> two
            start = question_toks.index(word)
            add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
        elif num < len(n2w3) and n2w3[num] in question_toks:
            word = n2w3[num] # 2 -> twice
            start = question_toks.index(word)
            add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
        elif num == 1:
            evidence = ['single', 'multiple', 'ever']
            for word in evidence:
                if word in question_toks:
                    start = question_toks.index(word)
                    add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
                    break
            else:
                add_value_from_reserved("1", sqlvalue)
        else:
            if not self.use_extracted_values(sqlvalue, values):
                raise ValueError('HAVING value %s can not be recognized!' % (num))
        return values, question_toks

    def extract_limit_val(self, num, values, question_toks, track):
        """ Decision procedure for LIMIT value (type int):
        1. num != 1 and directly in question_toks, add a new entry in the set ``values``
        2. num != 1 and the mapped word of num in question_toks, use the word as matched value candidate
        3. num == 1, directly add pre-defined value, o.w.
        4. num != 1 and try to retrieve value candidate from already generated values
        Each time when some span is matched, update question_toks to prevent further match
        """
        state = State(track, 'none', '=', 'none', 0)
        sqlvalue = SQLValue(str(num), state)
        num = int(num)
        if sqlvalue in values: return values, question_toks
        else: values.add(sqlvalue)

        if num != 1 and str(num) in question_toks:
            num = str(num)
            start = question_toks.index(num)
            add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
        elif num != 1 and num < len(n2w2) and n2w2[num] in question_toks:
            word = n2w2[num] # 2 -> second
            start = question_toks.index(word)
            add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
        elif num != 1 and num < len(n2w1) and n2w1[num] in question_toks:
            word = n2w1[num] # 2 -> two
            start = question_toks.index(word)
            add_value_from_token_idx((start, start + 1), question_toks, sqlvalue)
        elif num == 1:
            add_value_from_reserved("1", sqlvalue)
        # elif num == 1 and ('JJS' in pos_tags or 'RBS' in pos_tags):
        #     # most LIMIT 1, use JJS or RBS as symbol
        #     start = pos_tags.index('JJS') if 'JJS' in pos_tags else pos_tags.index('RBS')
        #     add_value_from_token_idx((start, start + 1), question_toks, pos_tags, sqlvalue)
        # elif num == 1:
        #     evidence = ['predominantly', 'majority', 'first', 'last', 'maximum', 'minimum']
        #     for word in evidence:
        #         if word in question_toks:
        #             start = question_toks.index(word)
        #             add_value_from_token_idx((start, start + 1), question_toks, pos_tags, sqlvalue)
        #             break
        #     else:
        #         add_value_from_reserved("1", sqlvalue)
        else:
            if not self.use_extracted_values(sqlvalue, values):
                raise ValueError('LIMIT value %s can not be recognized!' % (num))
        return values, question_toks

def add_value_from_reserved(val, sqlvalue):
    sqlvalue.add_candidate(SelectValueAction.reserved_spider[val])

def add_value_from_token_idx(index_pairs, question_toks, sqlvalue):
    # use ValueCandidate to represent matched value and update toks in question_toks
    matched_value = ' '.join(question_toks[index_pairs[0]: index_pairs[1]])
    candidate = ValueCandidate(matched_index=tuple(index_pairs), matched_value=matched_value)
    sqlvalue.add_candidate(candidate)
    question_toks[index_pairs[0]: index_pairs[1]] = [PLACEHOLDER] * (index_pairs[1] - index_pairs[0])
    # pos_tags[index_pairs[0]: index_pairs[1]] = [PLACEHOLDER] * (index_pairs[1] - index_pairs[0])
