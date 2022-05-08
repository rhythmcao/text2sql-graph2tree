#coding=utf8
import re, os, sys, json, pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from asdl.transition_system import SelectValueAction
from preprocess.process_utils import ValueCandidate, is_number, is_int, float_equal, extract_raw_question_span, ORDINAL

num2word = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']

class ValueProcessor():

    def __init__(self, table_path, db_dir) -> None:
        self.db_dir = db_dir
        if type(table_path) == str and os.path.splitext(table_path)[-1] == '.json':
            tables_list = json.load(open(table_path, 'r'))
            self.tables = { db['db_id']: db for db in tables_list }
        elif type(table_path) == str:
            self.tables = pickle.load(open(table_path, 'rb'))
        else: self.tables = table_path

    def postprocess_value(self, sqlvalue, db, entry):
        """
        sqlvalue: class SQLValue, the current condition value to be resolved
        db: dict, tables
        entry: Example of the current sample
        """
        vc, state = sqlvalue.candidate, sqlvalue.state
        db_id, col_id = db['db_id'], state.col_id
        cmp_op, col_name, col_type = state.cmp_op, db['column_names'][col_id][1], db['column_types'][col_id]

        if type(vc) == int: # reserved vocab
            reserved_vocab = SelectValueAction.vocab('wikisql').id2word
            return str(reserved_vocab[vc])

        assert isinstance(vc, ValueCandidate)
        value_str = vc.matched_cased_value
        if col_type == 'text':
            value_str = value_str.rstrip('?')
            if value_str.endswith('´s'): value_str = value_str[:-2]
            if len(value_str) < 4 and value_str.endswith('-'):
                value_str = value_str.rstrip('-')
            if not re.search(r'^[a-z ]+$', value_str, flags=re.I): # exist other symbols not letter or white space
                value_str = extract_raw_question_span(value_str, entry['question'])
            return str(value_str)
        else: # real
            value_str = value_str.replace(' ', '').rstrip('?')
            if re.search(r'\d', value_str, flags=re.I):
                value_str = re.sub(r'[a-z\'\.]+$', '', value_str, flags=re.I)
            if not is_number(value_str):
                value_str = num2word.index(value_str.lower()) if value_str.lower() in num2word else \
                    ORDINAL.index(value_str.lower()) if value_str.lower() in ORDINAL else \
                        0 if value_str.lower() in ['no', 'none'] else str(value_str)
            else: value_str = int(float(value_str)) if is_int(value_str) else float(value_str)
            return value_str


if __name__ == '__main__':

    from utils.constants import DATASETS
    data_dir = DATASETS['wikisql']['data']
    table_path = os.path.join(data_dir, 'tables.bin')
    db_dir = DATASETS['wikisql']['database']
    processor = ValueProcessor(table_path=table_path, db_dir=db_dir)

    dataset = pickle.load(open(os.path.join(data_dir, 'train.lgesql.bin'), 'rb'))
    # dataset = pickle.load(open(os.path.join(data_dir, 'dev.lgesql.bin'), 'rb'))
    test_samples = [ex['values'] for ex in dataset]

    def _equal(pred_value, real_value, col_type='text'):
        if col_type == 'real' and is_number(pred_value) and is_number(real_value):
            return float_equal(pred_value, real_value)
        return str(pred_value).lower() == str(real_value).lower()

    instance_correct, real_count, real_correct, text_count, text_correct = 0, 0, 0, 0, 0
    for entry, sqlvalues in zip(dataset, test_samples):
        db, flag = processor.tables[entry['db_id']], True
        for sqlvalue in sqlvalues:
            col_id = sqlvalue.state.col_id
            col_type = db['column_types'][col_id]
            if col_type == 'real': real_count += 1
            else: text_count += 1
            pred_value = processor.postprocess_value(sqlvalue, db, entry)
            if _equal(pred_value, sqlvalue.real_value, col_type):
                if col_type == 'real': real_correct += 1
                else: text_correct += 1
            else:
                flag, col_name, candidate = False, db['column_names'][col_id][1], sqlvalue.candidate
                matched_value = candidate.matched_cased_value if isinstance(candidate, ValueCandidate) else SelectValueAction.vocab('wikisql').id2word[candidate]
                print('Column %s[%s]: Gold/Match/Pred value: %s/%s/%s' % (col_name, col_type, sqlvalue.real_value, matched_value, pred_value))
        if flag: instance_correct += 1
        else:
            print('Question: [%s]' % (entry['question']))
            print('SQL: %s\n' % (entry['query']))

    print('Real values postprocess accuracy is %.4f' % (real_correct / float(real_count)))
    print('Text values postprocess accuracy is %.4f' % (text_correct / float(text_count)))
    print('Samples postprocess accuracy is %.4f' % (instance_correct / float(len(dataset))))

# train dataset
# Real values postprocess accuracy is 0.9999
# Text values postprocess accuracy is 0.9992
# Samples postprocess accuracy is 0.9992

# dev dataset
# Real values postprocess accuracy is 1.0000
# Text values postprocess accuracy is 0.9998
# Samples postprocess accuracy is 0.9998
