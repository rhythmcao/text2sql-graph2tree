#coding=utf8
import re, math, json
import cn2an, datetime
from itertools import chain
from typing import List, Set, Tuple
from fuzzywuzzy import process
from dateutil import parser as datetime_parser
from word2number import w2n
from collections import namedtuple

UNIT_OP = ('none', '-', '+', "*", '/')
UNIT_OP_NAME = ('', 'Minus', 'Plus', 'Times', 'Divide')
AGG_OP = ('none', 'max', 'min', 'count', 'sum', 'avg')

QUOTATION_MARKS = ["'", '"', '`', '‘', '’', '“', '”']
BOOL_TRUE = ['Y', 'y', 'T', 't', '1', 1, 'yes', 'Yes', 'true', 'True', 'YES', 'TRUE']
BOOL_FALSE = ['N', 'n', 'F', 'f', '0', 0, 'no', 'No', 'false', 'False', 'NO', 'FALSE']
BOOL_TRUE_ZH = ['1', 1, '是', '对', '有']
BOOL_FALSE_ZH = ['0', 0, '否', '错', '无']

ORDINAL = ['zeroth', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth']
FREQUENCY = ['once', 'twice', 'thrice']

MONTH = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
MONTH_ABBREV = [s[:3] for s in MONTH]
MONTH2NUMBER = dict(chain(zip(MONTH, range(1, 13)), zip(MONTH_ABBREV, range(1, 13)), [('sept', 9)]))

WEEK = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
WEEK_ABBREV = [s[:3] for s in WEEK]
WEEK2NUMBER = dict(chain(zip(WEEK, range(1, 8)), zip(WEEK_ABBREV, range(1, 8))))

ZH_NUMBER_1 = '零一二三四五六七八九'
ZH_NUMBER_2 = '〇壹贰叁肆伍陆柒捌玖'
ZH_TWO_ALIAS = '两俩'
ZH_NUMBER = ZH_NUMBER_1 + ZH_NUMBER_2 + ZH_TWO_ALIAS
ZH_UNIT = '十拾百佰千仟万萬兆亿億点'
ZH_UNIT_MAPPING = dict(zip(ZH_UNIT, [10, 10, 100, 100, 1000, 1000, 10000, 10000, 1000000, 100000000, 100000000, 1]))
ZH_RESERVE_CHARS = ZH_NUMBER + ZH_UNIT
DIGIT_ALIAS = lambda num: ZH_NUMBER_1[num] + ZH_NUMBER_2[num] if num != 2 else \
    ZH_NUMBER_1[num] + ZH_TWO_ALIAS + ZH_NUMBER_2[num]
ZH_WORD2NUM = lambda s: cn2an.cn2an(s, mode='smart')
ZH_NUM2WORD = lambda s, m: cn2an.an2cn(s, m)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_int(s):
    if is_number(s) and float(s) % 1 == 0:
        return True
    return False


def is_date(s):
    try:
        if re.search(r'(\d{1,4}[-/]\d{1,2}[-/]\d{1,4})( ?\d{1,2}:\d{1,2}:\d{1,2})?', s.strip()) is not None: return True
    except: return False


def float_equal(val1, val2, multiplier=1):
    val1, val2 = float(val1), float(val2)
    if math.fabs(val1 - val2) < 1e-5: return True
    elif math.fabs(val1 * multiplier - val2) < 1e-5 or math.fabs(val1 - val2 * multiplier) < 1e-6: return True
    return False

def load_db_contents(db_path):
    contents = json.load(open(db_path, 'r'))
    contents = {db['db_id']: db['tables'] for db in contents}
    return contents

def extract_db_contents(contents, db, strip=True):
    """ Some values in NL2SQL db_content should not strip() """
    db_cells = [[]]
    cells = contents[db['db_id']]
    for tab_id, table_name in enumerate(db['table_names']):
        all_column_cells = list(zip(*cells[table_name]['cell']))
        if all_column_cells:
            for column_cells in all_column_cells:
                column_cells = [str(cv).strip() if strip else str(cv) for cv in set(column_cells) if str(cv).strip()]
                db_cells.append(list(set(column_cells)))
        else:
            column_nums = len([c for c in db['column_names'] if c[0] == tab_id])
            db_cells.extend([[] for _ in range(column_nums)])
    return db_cells

def quote_normalization(question):
    """ Normalize quotation marks """
    if type(question) in [list, tuple]:
        new_question = []
        for idx, tok in enumerate(question):
            if len(tok) > 2 and tok[0] in QUOTATION_MARKS and tok[-1] in QUOTATION_MARKS:
                new_question += ['"', tok[1:-1], '"']
            elif len(tok) > 2 and tok[0] in QUOTATION_MARKS:
                new_question += ['"', tok[1:]]
            elif len(tok) > 2 and tok[-1] in QUOTATION_MARKS:
                new_question += [tok[:-1], '"']
            elif tok in QUOTATION_MARKS:
                new_question.append('"')
            elif len(tok) == 2 and tok[0] in QUOTATION_MARKS:
                # special case: the length of entity value is 1
                if idx + 1 < len(question) and question[idx + 1] in QUOTATION_MARKS:
                    new_question += ['"', tok[1]]
                else:
                    new_question.append(tok)
            else:
                new_question.append(tok)
        if new_question.count('"') == 1: # 's or s'
            index = new_question.index('"')
            new_question[index] = "'"
        return re.sub(r'\s+', ' ', ' '.join(new_question))
    else:
        for p in QUOTATION_MARKS:
            if p not in ['"', '“', '”']:
                if question.count(p) == 1:
                    question = question.replace(p, '')
                else:
                    question = question.replace(p, '"')
        return re.sub(r'\s+', ' ', question.strip())

def search_for_longest_substring(question, val, ignore_chars=''):
    # record longest "substring" in question: start and end position
    # "substring" means chars in the question follow the ASC order in val, but not necessarily adjacent in val
    start, end, val_ptr, candidates = 0, 0, 0, []
    while end < len(question):
        if question[end] not in val[val_ptr:] and question[end] not in ignore_chars:
            if end > start:
                candidates.append((start, end))
            val_ptr = 0
            if question[end] in val:
                start, end = end, end - 1
            else: start = end + 1
        elif question[end] in ignore_chars: pass # for longest match
        else: # move the pointer in val
            val_ptr = val.index(question[end], val_ptr)
        end += 1
    if end > start:
        candidates.append((start, end))
    if len(candidates) == 0: return (0, 0)
    # remove prefix and suffix stopwords for longest useful match
    best_match = [0, (0, 0)]
    for s, e in candidates:
        span = question[s:e]
        e -= len(span) - len(span.rstrip(ignore_chars))
        s += len(span) - len(span.lstrip(ignore_chars))
        if e - s > best_match[0]:
            best_match = [e - s, (s, e)]
    return best_match[1]


def number_string_normalization(s: str):
    # remove large number separator `,` (e.g., 10,000), suffix `s` (e.g., 1970s) or `-` (e.g., fourth- grade), and quotes
    return re.sub(r'["\',]', '', s.lower()).rstrip('-').rstrip('s').strip()


def map_en_string_to_number(s: str, month_week: bool = True):
    """ During postprocessing, try to map english word string into int number
    """
    s = re.sub(r'["\',]', '', s.lower()).rstrip('-').rstrip('.').strip()
    # try mapping to cardinal number
    try:
        negative_flag = False
        if s.startswith('-'): # negative number
            s, negative_flag = s[1:].strip(), True
        num = w2n.word_to_num(s)
        if negative_flag: return - num
        return num
    except: pass
    # try mapping to ordinal number
    if s in ORDINAL: return ORDINAL.index(s)
    suffixes = ['s', 'st', 'nd', 'rd', 'th']
    for suf in suffixes:
        try:
            s_ = s[:-len(suf)] if s.endswith(suf) else s
            num = w2n.word_to_num(s_.rstrip('-').strip())
            return num
        except: pass
    # try mapping to frequency number
    if s in FREQUENCY: return FREQUENCY.index(s) + 1
    # try month and week mapping
    if month_week:
        if s in MONTH2NUMBER: return MONTH2NUMBER[s]
        if s in WEEK2NUMBER: return WEEK2NUMBER[s]
    return None


def map_en_string_to_date(s: str):
    """ During postprocessing, try to map english word string into datetime format `2021-01-01`
    """
    try:
        s = re.sub(r'\s*(-|:|/)\s*', lambda match: match.group(1), s).strip() # remove whitespace near - and :
        if re.search(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,2} \d{1,2}:\d{1,2}:\d{1,2}$', s) or \
            re.search(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$', s) or re.search(r'^\d{1,2}[\-/][a-zA-Z]{3,4}[\-/]\d{4}$', s) or \
            re.search(r'^\d{1,2}:\d{1,2}:\d{1,2}$', s) or re.search(r'^\d+$', s): return s
        datetime_obj = datetime_parser.parse(s, fuzzy=True)
        today = datetime.datetime.today()
        if datetime_obj.year == today.year: return None
        elif datetime_obj.hour != 0 or datetime_obj.minute != 0 or datetime_obj.second != 0:
            norm_date = str(datetime_obj.strftime("%Y-%m-%d %H:%M:%S"))
        else: norm_date = str(datetime_obj.strftime("%Y-%m-%d"))
        return norm_date
    except: return None


def search_for_synonyms(value: str, cell_values: List[str], abbrev_set: List[set]):
    """ During postprocessing, try some common equivalent synonyms
    """
    value = value.lower()
    lower_cell_values = [v.lower() for v in cell_values]
    for s in abbrev_set:
        if value in s:
            if value in lower_cell_values:
                return str(cell_values[lower_cell_values.index(value)])
            for v in s:
                if v in lower_cell_values:
                    return str(cell_values[lower_cell_values.index(v)])
    return None


def extract_raw_question_span(s: str, q: str):
    """ During postprocessing, all other trials failed or LIKE operator exists, need to extract the raw span in the question,
    instead of the tokenized version which may be wrong due to tokenization error (e.g. `bob @ example . org` ).
    Notice that s and raw_q should be cased version, and if ignore whitespaces, s should occur in raw_q
    """
    if re.search(r'^[a-z0-9 ]+$', s, flags=re.I) or ' ' not in s or s in q: return s
    s_, q_ = s.replace(' ', ''), q.replace(' ', '')
    index_mapping = [idx for idx, c in enumerate(q) if c != ' ']
    try:
        start_id = q_.index(s_)
        start, end = index_mapping[start_id], index_mapping[start_id + len(s_) - 1]
        return q[start: end + 1]
    except: return s


def try_fuzzy_match(cased_value: str, cell_values: List[str], raw_question: str, abbrev_set: List[Set] = [], score: int = 60):
    """ During postprocessing, try text string fuzzy match
    """
    if len(cell_values) == 0: # no cell values available or LIKE operator avoids search for database
        cased_value = extract_raw_question_span(cased_value, raw_question)
    else:
        cell_values = [str(v) for v in cell_values]
        syn_value = search_for_synonyms(cased_value, cell_values, abbrev_set)
        if syn_value is not None: # some abbreviation
            cased_value = syn_value
        else: # fuzzy match, choose the most similar cell value
            matched_value, matched_score = process.extractOne(cased_value, cell_values)
            if matched_score >= score:
                cased_value = matched_value
            else:
                cased_value = extract_raw_question_span(cased_value, raw_question)
    return cased_value.strip()


class ValueCandidate():
    """ This class is used during decoding, where true SQL values are not available
    """
    __slots__ = ('matched_index', 'matched_value', 'matched_cased_value', 'value_id')
    def __init__(self, matched_index: Tuple[int, int], matched_value: str, matched_cased_value: str = None) -> None:
        super(ValueCandidate, self).__init__()
        self.matched_index = matched_index
        self.matched_value = matched_value
        self.matched_cased_value = matched_cased_value

    def set_value_id(self, value_id: int):
        # with pre-defined offset, used during AST construction
        self.value_id = value_id

    def set_matched_cased_value(self, matched_cased_value: str):
        self.matched_cased_value = matched_cased_value

    def __hash__(self) -> int:
        return hash(tuple(self.matched_index))

    def __eq__(self, other):
        if not isinstance(other, type(self)): return False
        return tuple(self.matched_index) == tuple(other.matched_index)

    def __lt__(self, other):
        return self.matched_index < other.matched_index

    def __str__(self) -> str:
        return "Matched Value = %s" % (self.matched_value)

    def __repr__(self) -> str:
        return "ValueCandidate[value=\"%s\", index=(%d, %d)]" % (self.matched_value, self.matched_index[0], self.matched_index[1])

State = namedtuple('State', ['track', 'agg_op', 'cmp_op', 'unit_op', 'col_id'])

class SQLValue():
    """ Encapsulate class ValueCandidate, provide extra information state, e.g. clause, column_id, type,
    for disambiguation when parsing SQL into AST. Some ValueCandidate will be used multiple times.
    """
    def __init__(self, real_value: str, state: State) -> None:
        super(SQLValue, self).__init__()
        self.real_value = real_value # real value used in sql, which is cell value
        self.state = state # namedtuple of (track, agg_op, cmp_op, unit_op, col_id)
        self.candidate = None

    def add_candidate(self, candidate):
        self.candidate = candidate # class ValueCandidate or int value representing reserved entries

    @property
    def matched_value(self):
        if isinstance(self.candidate, ValueCandidate): return self.candidate.matched_value
        return 'ReservedVocab[idx=' + str(self.candidate) + ']'

    @property
    def value_id(self):
        if isinstance(self.candidate, ValueCandidate): return self.candidate.value_id
        return self.candidate

    def __hash__(self) -> int:
        return hash((self.real_value, self.state))

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, type(self)): return False
        return self.real_value == o.real_value and self.state == o.state

    def __str__(self) -> str:
        return "Real Value = %s ; Matched Value = %s" % (self.real_value, self.matched_value)

    def __repr__(self) -> str:
        return "SQLValue[real=\"%s\", match=%s]" % (self.real_value, repr(self.candidate)) if isinstance(self.candidate, ValueCandidate) else \
            "SQLValue[real=\"%s\", match=%s]" % (self.real_value, self.matched_value)
