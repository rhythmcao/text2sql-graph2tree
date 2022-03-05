#coding=utf8
import os, json, pickle, sqlite3
from collections import Counter
from asdl.transition_system import SelectValueAction
from preprocess.spider.value_utils import ABBREV_SET
from preprocess.process_utils import is_number, is_int, is_date, BOOL_TRUE, BOOL_FALSE
from preprocess.process_utils import map_en_string_to_number, map_en_string_to_date, number_string_normalization, try_fuzzy_match, extract_raw_question_span
from utils.constants import TEST


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
        self.contents = self._load_db_contents() if not TEST else None
        self.cell_types = self._infer_cell_types() if not TEST else None

    def _load_db_contents(self):
        """ Return dict of DB contents:
        db_name [key] -> list (length num_columns) of db cells (list of cell values for each column) [value]
        """
        contents = {}
        for db in self.tables:
            db = self.tables[db]
            contents[db['db_id']] = [[]]
            db_file = os.path.join(self.db_dir, db['db_id'], db['db_id'] + '.sqlite')
            if not os.path.exists(db_file):
                contents[db['db_id']].extend([[] for _ in range(len(db['column_names_original']) - 1)])
                # raise ValueError('[ERROR]: database file %s not found ...' % (db_file))
                continue
            conn = sqlite3.connect(db_file)
            conn.text_factory = lambda b: b.decode(errors='ignore')
            for table_id, column_name in db['column_names_original'][1:]:
                table_name = db['table_names_original'][table_id]
                cursor = conn.execute("SELECT DISTINCT \"%s\" FROM \"%s\";" % (column_name, table_name))
                cell_values = cursor.fetchall()
                cell_values = [each[0] for each in cell_values if str(each[0]).strip().lower() not in ['', 'null', 'none']]
                contents[db['db_id']].append(cell_values)
            conn.close()
        return contents

    def retrieve_cell_values(self, db, col_id):
        if col_id == 0: return []
        db_file = os.path.join(self.db_dir, db['db_id'], db['db_id'] + '.sqlite')
        if not os.path.exists(db_file):
            print('Cannot find DB file:', db_file)
            return []
        conn = sqlite3.connect(db_file)
        conn.text_factory = lambda b: b.decode(errors='ignore')
        table_id, column_name = db['column_names_original'][col_id]
        table_name = db['table_names_original'][table_id]
        cursor = conn.execute("SELECT DISTINCT \"%s\" FROM \"%s\";" % (column_name, table_name))
        cell_values = cursor.fetchall()
        cell_values = [each[0] for each in cell_values if str(each[0]).strip().lower() not in ['', 'null', 'none']]
        conn.close()
        return cell_values

    def _infer_cell_types(self):
        cell_types = {}
        for db in self.tables:
            db = self.tables[db]
            cell_types[db['db_id']] = ['text']
            for col_id in range(len(db['column_names'])):
                if col_id == 0: continue
                cell_values = self.contents[db['db_id']][col_id]
                col_type = db['column_types'][col_id]
                cell_types[db['db_id']].append(self.infer_cell_type(cell_values, col_type))
                # cell_types[db['db_id']].append(col_type)
        return cell_types

    def infer_cell_type(self, cell_values, col_type):
        """ Three types: text, number, time
        Attention, boolean is merged into text
        """
        def get_type(v):
            if type(v) in [int, float]: return 'number'
            elif is_date(v) or col_type == 'time': return 'time'
            else: return 'text'

        types = [get_type(v) for v in cell_values]
        if len(types) > 0:
            counter = Counter(types)
            cell_type = counter.most_common(1)[0][0]
        else: cell_type = col_type if col_type in ['number', 'time'] else 'text'
        return cell_type

    def postprocess_value(self, value_id, value_candidates, db, state, entry):
        """ Retrieve DB cell values for reference
        @params:
            value_id: int for SelectAction
            value_candidates: list of ValueCandidate for matched value
            db: database schema dict
            state: namedtuple of (track, agg_op, cmp_op, unit_op, col_id)
        @return: value_str
        """
        clause = state.track.split('->')[-1]
        agg_op, col_id = state.agg_op, state.col_id
        like_op = 'like' in state.cmp_op
        raw_question = entry['question']
        col_type = db['column_types'][col_id]
        cell_values = self.contents[db['db_id']][col_id] if not TEST else self.retrieve_cell_values(db, col_id)
        cell_type = self.cell_types[db['db_id']][col_id] if not TEST else self.infer_cell_type(cell_values, col_type)
        output = None

        if value_id < SelectValueAction.size('spider'): # reserved values such as null, true, false, 0, 1
            value_str = SelectValueAction.reserved_spider.id2word[value_id]
            if clause == 'limit': # value should be integers larger than 0
                output = 1
            elif clause == 'having' or col_id == 0: # value should be integers
                output = 1 if value_str in ['1', 'true'] else 0
            elif value_str == 'null': # special value null
                output = 'null'
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
                        output = (1 if value_str == 'true' else 0) if cell_type == 'number' else ('T' if value_str == 'true' else 'F')
                else: # 0 or 1 for WHERE value
                    output = int(value_str) if cell_type == 'number' else str(value_str)
        else:
            vc = value_candidates[value_id - SelectValueAction.size('spider')]
            value_str, cased_value_str = vc.matched_value, vc.matched_cased_value

            def parse_number(month_week=False):
                num = map_en_string_to_number(value_str, month_week)
                if num is not None:
                    nonlocal output
                    output = num
                    return True
                return False

            if clause == 'limit': # value should be integers
                if is_number(value_str):
                    output = int(float(value_str))
                elif parse_number(False): pass
                else: output = 1
            elif clause == 'having': # value should be numbers
                if is_number(value_str):
                    output = int(float(value_str)) if is_int(value_str) else float(value_str)
                elif parse_number(False): pass
                else: output = 1
            else: # WHERE clause, value can be numbers, datetime or text
                def parse_datetime():
                    date = map_en_string_to_date(value_str)
                    if date is not None:
                        nonlocal output
                        output = date
                        return True
                    return False

                normed_value_str = number_string_normalization(value_str)
                if cell_type == 'number' and is_number(normed_value_str):
                    output = int(float(normed_value_str)) if is_int(normed_value_str) else float(normed_value_str)
                elif cell_type == 'number' and parse_number(True): pass
                elif cell_type == 'time' and parse_number(True): output = str(output)
                elif cell_type == 'time' and parse_datetime(): pass
                else: # text values
                    if is_number(normed_value_str): # some text appears like numbers such as phone number
                        output = extract_raw_question_span(cased_value_str, raw_question)
                    else:
                        output = try_fuzzy_match(cased_value_str, ([] if like_op else cell_values), raw_question, ABBREV_SET, 62)
        # add quote and wild symbol
        if like_op: output = '"%' + str(output).strip() + '%"'
        elif type(output) != str: output = str(output)
        else: output = '"' + str(output).strip() + '"'
        return output