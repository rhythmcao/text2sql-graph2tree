#coding=utf8
import copy, re, json, stanza
from collections import defaultdict
from itertools import permutations
from numpy.core.fromnumeric import cumsum
from asdl.transition_system import SelectValueAction
from preprocess.process_utils import ValueCandidate, State, SQLValue
from preprocess.spider.value_utils import n2w1, n2w2

CMP_OP = ('=', '>', '<')
PLACEHOLDER = '％' # never appeared in dataset, not english %

def preprocess_toks(toks, cased_toks):
    if toks[:3] == ['yes', 'or', 'no']:
        toks[:3] = [PLACEHOLDER * len(w) for w in toks[:3]]
        cased_toks[:3] = [PLACEHOLDER * len(w) for w in cased_toks[:3]]
    return toks, cased_toks

class ValueExtractor():

    def __init__(self) -> None:
        nlp = stanza.Pipeline('en', processors='tokenize')
        self.tokenize = lambda doc: [w.text for s in nlp(doc).sentences for w in s.words]

    def extract_values(self, entry: dict, db: dict, verbose: bool = False):
        """ Extract values(class SQLValue) which will be used in AST construction,
        we need question_toks for BIO labeling. The matched_index field in the ValueCandidate stores the index_pair of uncased_question_toks, not question.
        `track` is used to record the traverse clause path for disambiguation.
        """
        char2word = [idx for idx, w in enumerate(entry['uncased_question_toks']) for _ in range(len(w))]
        question_toks, cased_question_toks = copy.deepcopy(entry['uncased_question_toks']), copy.deepcopy(entry['cased_question_toks'])
        question_toks, cased_question_toks = preprocess_toks(question_toks, cased_question_toks)
        result = { 'values': [], 'entry': entry, 'db': db, 'char2word': char2word,
            'question_toks': question_toks, 'cased_question_toks': cased_question_toks }
        result = self.extract_values_from_sql(entry['sql'], result)
        entry = self.assign_values(entry, result['values'])
        if verbose and len(entry['values']) > 0:
            print('Question:', ' '.join(entry['uncased_question_toks']))
            print('SQL:', entry['query'])
            print('Values:', ' ; '.join([repr(val) for val in entry['values']]), '\n')
        return entry

    def assign_values(self, entry, values):
        # take set because some SQLValue may use the same ValueCandidate
        candidates = set([val.candidate for val in values if isinstance(val.candidate, ValueCandidate)])
        candidates = sorted(candidates)
        offset = SelectValueAction.size('wikisql')
        for idx, val in enumerate(candidates):
            val.set_value_id(idx + offset)
        entry['values'], entry['candidates'] = set(values), candidates
        return entry

    def use_extracted_values(self, sqlvalue, values, fuzzy=False):
        """ When value can not be found in the question,
        the last chance is to resort to extracted values for reference
        """
        old_candidates = set([v.candidate for v in values if v.real_value.lower() == sqlvalue.real_value.lower() and v.candidate is not None])
        if len(old_candidates) > 0:
            if len(old_candidates) > 1:
                print('Multiple candidate values which can be used for %s ...' % (sqlvalue.real_value))
            sqlvalue.add_candidate(list(old_candidates)[0])
            return True
        if fuzzy and len(sqlvalue.real_value) > 1:
            old_candidates = set([v.candidate for v in values if sqlvalue.real_value.lower() in v.real_value.lower() and v.candidate is not None])
            if len(old_candidates) > 0:
                if len(old_candidates) > 1:
                    print('Multiple fuzzy candidate values which can be used for %s ...' % (sqlvalue.real_value))
                sqlvalue.add_candidate(list(old_candidates)[0])
                return True
        return False

    def extract_values_from_sql(self, sql: dict, result: dict):
        column_names = list(map(lambda x: x[1], result['db']['column_names_original']))
        conds = reorder_conds_based_on_length_and_column_names(sql['conds'], ''.join(result['question_toks']), column_names)
        func = { str: self.extract_string_val, float: self.extract_float_val, int: self.extract_int_val }
        for cond_id, (col_id, cmp_id, val) in enumerate(conds):
            state = State('', 'none', CMP_OP[cmp_id], 'none', col_id)
            sqlvalue = SQLValue(str(val), state)
            result['values'].append(sqlvalue)
            if type(val) == str and val.isdigit() and str(int(val)) == val: val = int(val)
            result = func[type(val)](val, result, sqlvalue, conds, cond_id)
        return result
    
    def extract_float_val(self, val: float, result: dict, sqlvalue: SQLValue, conds: list, cond_id: int):
        question_toks, cased_question_toks,  = result['question_toks'], result['cased_question_toks']
        val_str, question, char2word = str(val), ''.join(result['question_toks']), result['char2word']
        if question_toks.count(val_str) > 0: # no need to resolve ambiguity, directly use the first one
            index = question_toks.index(val_str)
            add_value_from_token_idx((index, index + 1), question_toks, cased_question_toks, sqlvalue)
        elif question.count(val_str) == 1:
            start_id = question.index(val_str)
            start, end = char2word[start_id], char2word[start_id + len(val_str) - 1] + 1
            add_value_from_token_idx((start, end), question_toks, cased_question_toks, sqlvalue)
        else: raise ValueError('The float value %s not recognized' % (val_str))
        return result

    def extract_int_val(self, val: int, result: dict, sqlvalue: SQLValue, conds: list, cond_id: int):
        question_toks, cased_question_toks, values = result['question_toks'], result['cased_question_toks'], result['values']
        val_str, question, char2word = str(val), ''.join(question_toks), result['char2word']
        word_ids = extract_int_numbers(question_toks, val)

        def resolve_ambiguity(word_ids):
            count = sum([1 for _, _, value in conds[cond_id:] if str(value) == val_str])
            if count == len(word_ids): # one for each
                index = word_ids[0]
                add_value_from_token_idx((index, index + 1), question_toks, cased_question_toks, sqlvalue)
                return True
            useful_ids = [wid for wid in word_ids if wid == 0 or question_toks[wid - 1] in ['was', 'is', 'of', 'than', 'being']]
            if len(useful_ids) == 1:
                index = useful_ids[0]
                add_value_from_token_idx((index, index + 1), question_toks, cased_question_toks, sqlvalue)
                return True
            useful_ids = [wid for wid in word_ids if wid == 0 or question_toks[wid - 1] not in ['for', 'race', 'region']]
            if len(useful_ids) == 1:
                index = useful_ids[0]
                add_value_from_token_idx((index, index + 1), question_toks, cased_question_toks, sqlvalue)
                return True
            return False

        def relax_constraints(val):
            if question_toks.count(str(val)) == 1: # a 1 xxx
                index = question_toks.index(str(val))
                add_value_from_token_idx((index, index + 1), question_toks, cased_question_toks, sqlvalue)
                return True
            word_ids = extract_int_numbers(question_toks, val, word_transform=True, ordinal=True)
            if len(word_ids) == 1:
                index = word_ids[0]
                add_value_from_token_idx((index, index + 1), question_toks, cased_question_toks, sqlvalue)
                # re-assign ValueCandidate due to order issues
                sqlvalues = [v for v in values if v.real_value == str(val)]
                vcs = sorted([v.candidate for v in sqlvalues])
                for sv, vc in zip(sqlvalues, vcs): sv.add_candidate(vc)
                return True
            return False

        if len(word_ids) == 1:
            index = word_ids[0]
            add_value_from_token_idx((index, index + 1), question_toks, cased_question_toks, sqlvalue)
        elif len(word_ids) == 0 and question.count(val_str) == 1:
            start_id = question.index(val_str)
            start, end = char2word[start_id], char2word[start_id + len(val_str) - 1] + 1
            add_value_from_token_idx((start, end), question_toks, cased_question_toks, sqlvalue)
        elif len(word_ids) > 1 and resolve_ambiguity(word_ids): pass
        elif not relax_constraints(val): raise ValueError('The int value %s can not be recognized' % (val_str))
        return result

    def extract_string_val(self, val: str, result: dict, sqlvalue: SQLValue, conds: list, cond_id: int):
        values, char2word = result['values'], result['char2word']
        question_toks, cased_question_toks = result['question_toks'], result['cased_question_toks']
        question, cased_question = ''.join(result['question_toks']), ''.join(result['cased_question_toks'])
        val_ = val.replace(' ', '')
        already_handle_a = sum([1 for v in values if v.real_value.lower() == 'a']) > 1 and sqlvalue.real_value.lower() == 'a' # avoid matching article 'a'
        only_token = len(self.tokenize(val)) == 1 and re.search(r'^[a-z]+$', val, flags=re.I)

        def deal_with_multiple_values():
            if val.lower() == 'a': return False
            count = sum([1 for _, _, value in conds[cond_id:] if str(value).lower() == val.lower()])
            # one by one
            token_val_count = question_toks.count(val.lower())
            if token_val_count == count:
                index = question_toks.index(val.lower())
                add_value_from_token_idx((index, index + 1), question_toks, cased_question_toks, sqlvalue)
                return True
            val_count = question.count(val_.lower())
            if not only_token and count == val_count:
                start_id = question.index(val_.lower())
                start, end = char2word[start_id], char2word[start_id + len(val_.lower()) - 1] + 1
                add_value_from_token_idx((start, end), question_toks, cased_question_toks, sqlvalue)
                return True
            val_count = token_val_count if only_token else val_count
            if val_count > count: # select according to prefix signals
                indexes = [(wid, wid + 1) for wid, w in enumerate(question_toks) if w == val.lower()] if only_token else extract_indexs(question, val_.lower(), char2word)
                useful_indexes = []
                for start, end in indexes:
                    if start == 0: useful_indexes.append((start, end))
                    elif start >= 1 and question_toks[start - 1] in ['"', 'is', 'of', 'was', 'being', 'class']: useful_indexes.append((start, end))
                if len(useful_indexes) == count:
                    start, end = useful_indexes[0]
                    add_value_from_token_idx((start, end), question_toks, cased_question_toks, sqlvalue)
                    return True
                useful_indexes = [(start, end) for start, end in indexes if not (end <= len(question_toks) - 1 and question_toks[end] in ['fee', 'of', '?'])]
                if len(useful_indexes) == count:
                    start, end = useful_indexes[0]
                    add_value_from_token_idx((start, end), question_toks, cased_question_toks, sqlvalue)
                    return True
                useful_indexes = []
                for start, end in indexes:
                    if end <= len(question_toks) - 1 and question_toks[end] in ['fee', 'of', '?']: continue
                    elif end <= len(question_toks) - 1 and question_toks[end] in ['was', 'as', 'number']: useful_indexes.append((start, end))
                    elif start > 0 and question_toks[start - 1] in ['called', 'from', 'team', 'shows', 'has', 'with']: useful_indexes.append((start, end))
                    elif start > 1 and ' '.join(question_toks[start - 2: start]) in ['has a', 'with a', 'with an', 'was the', 'is the', 'and a', 'for the']: useful_indexes.append((start, end))
                if len(useful_indexes) == count:
                    start, end = useful_indexes[0]
                    add_value_from_token_idx((start, end), question_toks, cased_question_toks, sqlvalue)
                    return True
                return False
            elif val_count == 0: # reuse extracted values
                # special cases
                index, length = None, 0
                if val.lower() == 'no' and 'No-' in cased_question_toks: index, length = cased_question_toks.index('No-'), 1
                elif val.lower() == 'lr' and '14' in question_toks: index, length = question_toks.index('14'), 1
                elif val.lower() == 'na' and 'ne' in question_toks: index, length = question_toks.index('ne'), 1
                elif val.lower() == 'movies' and 'moviein' in question_toks: index, length = question_toks.index('moviein'), 1
                elif val.lower() == u'951.750\u2009km' and '951.750' in question_toks: index, length = question_toks.index('951.750'), 2
                elif len(val) <= 3 and val.lower() + '?' in question_toks: index, length = question_toks.index(val.lower() + '?'), 1
                elif val.lower() + '\'s' in question_toks: index, length = question_toks.index(val.lower() + '\'s'), 1
                elif val.lower() + '´s' in question_toks: index, length = question_toks.index(val.lower()+ '´s'), 1
                elif self.use_extracted_values(sqlvalue, values, fuzzy=True): return True
                if index is not None:
                    add_value_from_token_idx((index, index + length), question_toks, cased_question_toks, sqlvalue)
                    return True
            return False

        def deal_with_special_a():
            if val.lower() != 'a': return False
            count = sum([1 for _, _, value in conds[cond_id:] if str(value).lower() == 'a'])
            A_count = cased_question_toks.count('A')
            if A_count == count:
                index = cased_question_toks.index('A')
                add_value_from_token_idx((index, index + 1), question_toks, cased_question_toks, sqlvalue)
                return True
            indexes = [(wid, wid + 1) for wid, w in enumerate(question_toks) if w == 'a']
            useful_indexes = []
            for start, end in indexes:
                if start > 0 and question_toks[start - 1] in ['is', 'of', 'was', 'were', '"']: useful_indexes.append((start, end))
                elif end <= len(question_toks) - 1 and question_toks[end] in ['value']: useful_indexes.append((start, end))
            if len(useful_indexes) > 0 and len(useful_indexes) <= count:
                add_value_from_token_idx(tuple(useful_indexes[0]), question_toks, cased_question_toks, sqlvalue)
                return True
            elif len(useful_indexes) == 0 and self.use_extracted_values(sqlvalue, values, fuzzy=False): return True
            return False

        # some value error fixing
        if val in ['a', 'nadal', 'no']: val = val.title()
        if cased_question_toks.count(val) == 1 and question_toks.count(val.lower()) > 1:
            if val == 'local' and cased_question_toks[cased_question_toks.index(val) + 1] == '/': val = 'Local'
            if val == 'total' and question_toks[question_toks.index(val) - 1] in ['by', 'the']: val = 'Total'

        if cased_question_toks.count(val) == 1: # cased value with precedence
            index = cased_question_toks.index(val)
            add_value_from_token_idx((index, index + 1), question_toks, cased_question_toks, sqlvalue)
        elif not already_handle_a and question_toks.count(val.lower()) == 1: # token with precedence
            index = question_toks.index(val.lower())
            add_value_from_token_idx((index, index + 1), question_toks, cased_question_toks, sqlvalue)
        elif not only_token and cased_question.count(val_) == 1:
            start_id = cased_question.index(val_)
            start, end = char2word[start_id], char2word[start_id + len(val_) - 1] + 1
            add_value_from_token_idx((start, end), question_toks, cased_question_toks, sqlvalue)
        elif not only_token and question.count(val_.lower()) == 1:
            start_id = question.index(val_.lower())
            start, end = char2word[start_id], char2word[start_id + len(val_.lower()) - 1] + 1
            add_value_from_token_idx((start, end), question_toks, cased_question_toks, sqlvalue)
        elif deal_with_multiple_values(): pass
        elif val == '–': add_value_from_reserved('–', sqlvalue)
        elif not deal_with_special_a():
            raise ValueError('[ERROR]: unrecongized string value %s for column %s' % (val, result['db']['column_names_original'][conds[cond_id][0]][1]))
        return result


def extract_int_numbers(question_toks, num, word_transform: bool = False, ordinal: bool = False):
    alias = [str(num)]
    if word_transform and 0 <= num <= 10:
        alias.append(n2w1[num])
        if num == 0: alias += ['no', 'none', 'multiples']
    if word_transform and ordinal and 0 <= num <= 10:
        alias.append(n2w2[num])
    suffix_pattern = r'-?(th|st|nd|rd|s|\?|\.)$' if ordinal else r'(\?|\.)$'
    occurrences = [wid for wid, w in enumerate(question_toks) if (w in alias or re.sub(suffix_pattern, '', w) in alias) and \
        (wid == 0 or question_toks[wid - 1] not in ['-', 'a']) and \
        (wid == len(question_toks) - 1 or not re.search(r'^[\-/]', question_toks[wid + 1]))]
    return occurrences


def reorder_conds_based_on_length_and_column_names(conds: list, question: str, column_names: list):
    question, column_names = question.replace(' ', '').lower(), [c.replace(' ', '').lower() for c in column_names]
    value_dict = defaultdict(list)
    for cond in conds:
        val = str(cond[2]).lower()
        value_dict[val].append(cond)
    value_keys = sorted(value_dict.keys(), key=lambda k: - len(k)) # longest value frist
    sorted_conds = []
    for key in value_keys:
        if len(value_dict[key]) == 1:
            sorted_conds.extend(value_dict[key])
            continue
        # try to resolve ambiguity according to column names occurred in the question, fix 40/2 chaos in train and dev dataset
        # if failed, use the original condition order
        conds = resolve_conds_ambiguity(value_dict[key], question, column_names)
        sorted_conds.extend(conds)
    return sorted_conds


def resolve_conds_ambiguity(conds: list, question: str, column_names: list):
    cols = [column_names[col_id] for col_id, _, _ in conds]
    for col in cols:
        if col not in question: return conds
    for order in permutations(range(len(cols))):
        sorted_conds, qid = [], 0
        for cond_id in order:
            col = cols[cond_id]
            if col in question[qid:]:
                qid = question.index(col, qid) + len(col)
                sorted_conds.append(conds[cond_id])
            else:
                qid, sorted_conds = 0, []
                break
        else: return sorted_conds
    return conds


def extract_indexs(question, val, char2word):
    question, val = question.lower(), val.lower()
    if val not in question: return []
    idx, return_indexes = 0, []
    while idx < len(question) and val in question[idx:]:
        start_id = question.index(val, idx)
        end_id = start_id + len(val)
        start, end = char2word[start_id], char2word[end_id - 1] + 1
        return_indexes.append((start, end))
        idx = end_id
    return return_indexes


def add_value_from_reserved(val, sqlvalue):
    sqlvalue.add_candidate(SelectValueAction.vocab('wikisql')[val])


def add_value_from_token_idx(index_pairs, question_toks, cased_question_toks, sqlvalue):
    # use ValueCandidate to represent matched value and update toks in question_toks
    start, end = index_pairs
    matched_value = ' '.join(question_toks[start: end])
    cased_value = ' '.join(cased_question_toks[start: end])
    candidate = ValueCandidate(matched_index=tuple(index_pairs), matched_value=matched_value, matched_cased_value=cased_value)
    sqlvalue.add_candidate(candidate)
    question_toks[start: end] = [PLACEHOLDER * len(question_toks[idx]) for idx in range(start, end)]
    cased_question_toks[start: end] = [PLACEHOLDER * len(cased_question_toks[idx]) for idx in range(start, end)]
    return