#coding=utf8
from collections import namedtuple

QUOTATION_MARKS = ["'", '"', '`', '‘', '’', '“', '”', '``', "''", "‘‘", "’’"]

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

class ValueCandidate():
    """ This class is used during decoding, where true SQL values are not available
    """
    __slots__ = ('matched_index', 'matched_value', 'matched_cased_value', 'value_id')
    def __init__(self, matched_index, matched_value, matched_cased_value=None) -> None:
        super(ValueCandidate, self).__init__()
        self.matched_index = matched_index
        self.matched_value = matched_value
        self.matched_cased_value = matched_cased_value

    def set_value_id(self, value_id):
        # with pre-defined offset, used during AST construction
        self.value_id = value_id

    def set_matched_cased_value(self, matched_cased_value):
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