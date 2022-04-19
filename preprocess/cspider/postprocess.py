#coding=utf8
import os, json, pickle, re
from collections import Counter
from asdl.transition_system import SelectValueAction
from preprocess.cspider.value_utils import ABBREV_SET, PLACEHOLDER
from preprocess.process_utils import ZH_RESERVE_CHARS, ZH_WORD2NUM, is_number, is_int
from preprocess.process_utils import BOOL_TRUE, BOOL_FALSE, load_db_contents, extract_db_contents
from preprocess.process_utils import number_string_normalization, try_fuzzy_match, extract_raw_question_span


class ValueProcessor():

    def __init__(self, table_path, db_dir) -> None:
        self.db_dir = db_dir
        if type(table_path) == str and os.path.splitext(table_path)[-1] == '.json':
            tables_list = json.load(open(table_path, 'r'))
            self.tables = {}
            for db in tables_list:
                self.tables[db['db_id']] = db
        elif type(table_path) == str:
            self.tables = pickle.load(open(table_path, 'rb'))
        else:
            self.tables = table_path
        # TEST: for submission, save time and space
        self.contents = self._load_db_contents()

    def _load_db_contents(self):
        contents = load_db_contents(self.db_dir)
        self.contents = {}
        for db_id in self.tables:
            db = self.tables[db_id]
            self.contents[db_id] = db['cells'] if 'cells' in db else extract_db_contents(contents, db)
        return self.contents

    def postprocess_value(self, sqlvalue, db, entry):
        """ Retrieve DB cell values for reference
        @params:
            value_id: int for SelectAction
            value_candidates: list of ValueCandidate for matched value
            db: database schema dict
            state: namedtuple of (track, agg_op, cmp_op, unit_op, col_id)
        @return: value_str
        """
        vc, state = sqlvalue.candidate, sqlvalue.state
        clause = state.track.split('->')[-1]
        agg_op, unit_op, col_id = state.agg_op, state.unit_op, state.col_id
        like_op = 'like' in state.cmp_op
        raw_question = entry['question']
        col_type = 'number' if agg_op != 'none' or unit_op != 'none' else db['column_types'][col_id]
        cell_values = self.contents[db['db_id']][col_id]
        output = None

        if type(vc) == int: # reserved values such as null, true, false, 0, 1
            value_str = SelectValueAction.vocab('cspider').id2word[vc]
            if clause == 'limit': # value should be integers larger than 0
                output = 1
            elif clause == 'having' or col_id == 0: # value should be integers
                output = 1 if value_str in ['1', 'true'] else 0
            elif value_str == 'null': # special value null
                output = '空'
            else: # value in WHERE clause, it depends
                if value_str in ['true', 'false']:
                    evidence = []
                    for cv in cell_values:
                        for idx, (t, f) in enumerate(zip(BOOL_TRUE, BOOL_FALSE)):
                            if cv == t or cv == f:
                                evidence.append(idx)
                                break
                    if len(evidence) > 0:
                        bool_idx = Counter(evidence).most_common(1)[0][0]
                        output = BOOL_TRUE[bool_idx] if value_str == 'true' else BOOL_FALSE[bool_idx]
                    else:
                        output = (1 if value_str == 'true' else 0) if col_type == 'number' else ('确定' if value_str == 'true' else '否定')
                else: # 0 or 1 for WHERE value
                    output = int(value_str) if col_type == 'number' else str(value_str)
        else:
            cased_value_str = vc.matched_cased_value
            cased_value_str = re.sub(r'([a-zA-Z0-9])\s+([a-zA-Z0-9])', lambda match_obj: match_obj.group(1) + PLACEHOLDER + match_obj.group(2), cased_value_str)
            cased_value_str = re.sub(r'\s+', '', cased_value_str).replace(PLACEHOLDER, ' ')
            value_str = cased_value_str.lower()

            def word_to_number(val):
                # ignore these metrics containing ambiguous char 千 or k
                val = re.sub(r'千米|千瓦|千克|千斤|千卡|kg|km', '', val, flags=re.I)
                val = re.sub(r'[^0-9\.\-{}]'.format(ZH_RESERVE_CHARS), '', val)
                if is_number(val): val = float(val)
                else:
                    try: val = ZH_WORD2NUM(val)
                    except: return False
                num = int(val) if is_int(val) else val
                nonlocal output
                output = num
                return True

            if clause == 'limit': # value should be integers
                if is_number(value_str):
                    output = int(float(value_str))
                elif word_to_number(value_str): pass
                else: output = 1
            elif clause == 'having': # value should be numbers
                if is_number(value_str):
                    output = int(float(value_str)) if is_int(value_str) else float(value_str)
                elif word_to_number(value_str): pass
                else: output = 1
            else: # WHERE clause, value can be numbers, datetime or text
                def word_to_date(val):
                    match1 = re.search(r'\d+[～-]\d+-\d+( \d+:\d+:\d+)?', val)
                    match2 = re.search(r'(\d+)年(\d+)月(\d+)日?', val)
                    if match1 or match2:
                        nonlocal output
                        output = match1.group(0) if match1 else match2.group(1) + '-' + match2.group(2) + '-' + match2.group(3)
                        return True
                    return False

                normed_value_str = number_string_normalization(value_str)
                if col_type == 'number' and is_number(normed_value_str):
                    output = int(float(normed_value_str)) if is_int(normed_value_str) else float(normed_value_str)
                elif col_type == 'number' and word_to_number(value_str): pass
                elif col_type == 'time' and word_to_date(value_str): pass
                elif col_type == 'time' and word_to_number(value_str): output = str(output)
                else: # text values
                    if is_number(normed_value_str) or re.search(u'[\u4e00-\u9fa5]', value_str):
                        output = extract_raw_question_span(cased_value_str, raw_question)
                    else: # do not have chinese chars
                        output = try_fuzzy_match(cased_value_str, ([] if like_op else cell_values), raw_question, ABBREV_SET, 85)
        # add quote and wild symbol
        if like_op: output = '"%' + str(output).strip() + '%"'
        elif type(output) != str: output = str(output)
        else: output = '"' + str(output).strip() + '"'
        return output