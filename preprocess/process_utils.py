#coding=utf8
import re
from itertools import chain
from typing import List, Set, Tuple
from fuzzywuzzy import process
from dateutil import parser as datetime_parser
from word2number import w2n
from collections import namedtuple

QUOTATION_MARKS = ["'", '"', '`', '‘', '’', '“', '”', '``', "''", "‘‘", "’’"]
BOOL_TRUE = ['Y', 'y', 'T', 't', '1', 1, '是', '对', 'yes', 'Yes', 'true', 'True', 'YES', 'TRUE']
BOOL_FALSE = ['N', 'n', 'F', 'f', '0', 0, '否', '错', 'no', 'No', 'false', 'False', 'NO', 'FALSE']

ORDINAL = ['zeroth', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth']
FREQUENCY = ['once', 'twice', 'thrice']

MONTH = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
MONTH_ABBREV = [s[:3] for s in MONTH]
MONTH2NUMBER = dict(chain(zip(MONTH, range(1, 13)), zip(MONTH_ABBREV, range(1, 13)), [('sept', 9)]))

WEEK = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
WEEK_ABBREV = [s[:3] for s in WEEK]
WEEK2NUMBER = dict(chain(zip(WEEK, range(1, 8)), zip(WEEK_ABBREV, range(1, 8))))


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
        if re.search(r'(\d{1,4}[-/]\d{1,2}[-/]\d{1,4})', s.strip()) is not None: return True
    except: return False


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
        for sym in QUOTATION_MARKS:
            question = question.replace(sym, '"')
        if question.count('"') == 1:
            question = question.replace('"', "'")
        return re.sub(r'\s+', ' ', question)


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
            num = w2n.word_to_num(s.rstrip(suf).rstrip('-').strip())
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
        datetime_obj = datetime_parser.parse(s, fuzzy=True)
        norm_date = str(datetime_obj.strftime("%Y-%m-%d"))
        return norm_date
    except: return None


def search_for_synonyms(value: str, cell_values: List[str], abbrev_set: List[set]):
    """ During postprocessing, fuzzy match failed, try some common equivalent synonyms
    """
    lower_value = value.lower()
    lower_cell_values = [str(v).lower() for v in cell_values]
    for s in abbrev_set:
        if lower_value in s:
            for v in s:
                if v in lower_cell_values:
                    index = lower_cell_values.index(v)
                    return str(cell_values[index])
    return None


def extract_raw_question_span(s: str, q: str):
    """ During postprocessing, all other trials failed or LIKE operator exists, need to extract the raw span in the question,
    instead of the tokenized version which may be wrong due to tokenization error (e.g. `bob @ example . org` ).
    Notice that s and raw_q should be cased version, and if ignore whitespaces, s should occur in raw_q
    """
    if s in q: return s
    s_ = s.replace(' ', '')
    q_ = q.replace(' ', '')
    index_mapping = [idx for idx, c in enumerate(q) if c != ' ']
    try:
        start_id = q_.index(s_)
        start, end = index_mapping[start_id], index_mapping[start_id + len(s_)]
        return q[start: end].rstrip(' ')
    except: return s


def try_fuzzy_match(cased_value: str, cell_values: List[str], raw_question: str, abbrev_set: List[Set] = [], score: int = 60):
    """ During postprocessing, try text string fuzzy match
    """
    if len(cell_values) == 0: # no cell values available or LIKE operator avoids search for database
        cased_value = extract_raw_question_span(cased_value, raw_question)
    else: # fuzzy match, choose the most similar cell value
        cell_values = [str(v) for v in cell_values]
        matched_value, matched_score = process.extractOne(cased_value, cell_values)
        if matched_score >= score:
            cased_value = matched_value
        else:
            retrieve_value = search_for_synonyms(cased_value, cell_values, abbrev_set)
            if retrieve_value is not None:
                cased_value = retrieve_value
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
