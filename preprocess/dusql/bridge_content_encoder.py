#coding=utf8
import difflib
from typing import List, Optional, Tuple
from rapidfuzz import fuzz
from utils.constants import MAX_CELL_NUM

# copied from DuSQL official codes
STOPWORDS = set(["的", "是", "，", "？", "有", "多少", "哪些", "我", "什么", "你", "知道", "啊",
            "一下", "吗", "在", "请问", "或者", "想", "和", "为", "帮", "那个", "你好", "这", "上",
            "了", "并且", "都", "呢", "呀", "哪个", "还有", "这个", "-", "项目", "我查", "就是",
            "它", "要求", "谁", "了解", "告诉", "时候", "个", "能", "那", "人", "问", "中",
            "可以", "一共", "哪", "麻烦", "叫", "想要", "《", "》", "分别"])
_commonwords = {"是", "否", "有", "无"}

def is_number(s: str) -> bool:
    try:
        float(s.replace(",", ""))
        return True
    except:
        return False


def is_stopword(s: str) -> bool:
    return s.strip() in STOPWORDS


def is_commonword(s: str) -> bool:
    return s.strip() in _commonwords


def is_common_db_term(s: str) -> bool:
    return s.strip().startswith('item')


class Match(object):
    def __init__(self, start: int, size: int) -> None:
        self.start = start
        self.size = size


def is_span_separator(c: str) -> bool:
    return c in "'\"()`,.?! ，？！。（）"


def split(s: str) -> List[str]:
    return [c.lower() for c in s.strip()]


def prefix_match(s1: str, s2: str) -> bool:
    i, j = 0, 0
    for i in range(len(s1)):
        if not is_span_separator(s1[i]):
            break
    for j in range(len(s2)):
        if not is_span_separator(s2[j]):
            break
    if i < len(s1) and j < len(s2):
        return s1[i] == s2[j]
    elif i >= len(s1) and j >= len(s2):
        return True
    else:
        return False


def get_effective_match_source(s: str, start: int, end: int) -> Match:
    _start = -1

    for i in range(start, start - 2, -1):
        if i < 0:
            _start = i + 1
            break
        if is_span_separator(s[i]):
            _start = i
            break

    if _start < 0:
        return None

    _end = -1
    for i in range(end - 1, end + 3):
        if i >= len(s):
            _end = i - 1
            break
        if is_span_separator(s[i]):
            _end = i
            break

    if _end < 0:
        return None

    while _start < len(s) and is_span_separator(s[_start]):
        _start += 1
    while _end >= 0 and is_span_separator(s[_end]):
        _end -= 1

    return Match(_start, _end - _start + 1)


def get_matched_entries(
    s: str, field_values: List[str], m_theta: float = 0.85, s_theta: float = 0.85
) -> Optional[List[Tuple[str, Tuple[str, str, float, float, int]]]]:
    if not field_values:
        return None

    if isinstance(s, str):
        n_grams = split(s)
    else:
        n_grams = s

    matched = dict()
    for field_value in field_values:
        if not isinstance(field_value, str):
            continue
        fv_tokens = split(field_value)
        sm = difflib.SequenceMatcher(None, n_grams, fv_tokens)
        match = sm.find_longest_match(0, len(n_grams), 0, len(fv_tokens))
        if match.size > 0:
            source_match = get_effective_match_source(
                n_grams, match.a, match.a + match.size
            )
            if source_match and source_match.size > 1:
                match_str = field_value[match.b : match.b + match.size]
                source_match_str = s[
                    source_match.start : source_match.start + source_match.size
                ]
                c_match_str = match_str.lower().strip()
                c_source_match_str = source_match_str.lower().strip()
                c_field_value = field_value.lower().strip()
                if (
                    c_match_str
                    and not is_number(c_match_str)
                    and not is_common_db_term(c_match_str)
                ):
                    if (
                        is_stopword(c_match_str)
                        or is_stopword(c_source_match_str)
                        or is_stopword(c_field_value)
                    ):
                        continue
                    if prefix_match(c_field_value, c_source_match_str):
                        match_score = (
                            fuzz.ratio(c_field_value, c_source_match_str) / 100
                        )
                    else:
                        match_score = 0
                    if (
                        is_commonword(c_match_str)
                        or is_commonword(c_source_match_str)
                        or is_commonword(c_field_value)
                    ) and match_score < 1:
                        continue
                    s_match_score = match_score
                    if match_score >= m_theta and s_match_score >= s_theta:
                        matched[match_str] = (
                            field_value,
                            source_match_str,
                            match_score,
                            s_match_score,
                            match.size,
                        )

    if not matched:
        return None
    else:
        return sorted(
            matched.items(),
            key=lambda x: (1e16 * x[1][2] + 1e8 * x[1][3] + x[1][4]),
            reverse=True,
        )

def is_item(cells: List[str] = []):
    if cells:
        if 2 * sum([c.startswith('item') for c in cells]) > len(cells):
            return True
    return False

def get_database_matches(
    question: str,
    cells: List[str],
    col_name: str,
    col_type: str,
    match_threshold: float = 0.85,
) -> List[str]:
    matches = []
    # deal with special number id "item_xxx"
    if not cells: return matches
    if is_item(cells):
        for c in cells:
            if c in question:
                matches.append(c)
        return sorted(matches, key=lambda x: - len(x))
    # other cases
    elif col_type != 'number':
        matched_entries = get_matched_entries(
            s=question,
            field_values=cells,
            m_theta=match_threshold,
            s_theta=match_threshold,
        )
        if matched_entries:
            num_values_inserted = 0
            for _match_str, (
                field_value,
                _s_match_str,
                match_score,
                s_match_score,
                _match_size,
            ) in matched_entries:
                if "id" in col_name.lower() and match_score * s_match_score < 1:
                    continue
                matches.append(field_value)
                num_values_inserted += 1
                if num_values_inserted >= MAX_CELL_NUM:
                    break
    return matches
