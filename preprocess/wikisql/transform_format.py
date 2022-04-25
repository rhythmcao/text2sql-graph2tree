#coding=utf8
import os, sys, json
# from nltk import word_tokenize
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def transform_table_format(*table_path):
    tables = []
    for tp, data_split in table_path:
        with open(tp, 'r') as inf:
            for line in inf:
                if not line.strip(): continue
                db = json.loads(line.strip())
                db['db_id'], db['data_split'] = db['id'], data_split
                db['table_names_original'] = [db['id']]
                db['table_names'] = db['table_names_original']
                db['column_names_original'] = [(0, c) for c in db['header']]
                db['column_names'] = [(tid, c.lower()) for tid, c in db['column_names_original']]
                db['primary_keys'] = []
                db['foreign_keys'] = []
                db['column_types'] = db['types']
                db['cells'] = [list(set(c)) for c in list(zip(*db['rows']))]
                tables.append(db)
    return tables


def transform_dataset_format(dataset_path, tables):
    """ Add fields db_id, query, question_toks and 
    """
    dataset = []
    with open(dataset_path, 'r') as inf:
        for line in inf:
            if not line.strip(): continue
            ex = json.loads(line.strip())
            ex['db_id'] = ex['table_id']
            # ex['question_toks'] = word_tokenize(ex['question'])
            ex['query'] = json.dumps({'table_id': ex['table_id'], 'sql': ex['sql']}, ensure_ascii=False)
            dataset.append(ex)
    return dataset


if __name__ == '__main__':

    from utils.constants import DATASETS
    data_dir = DATASETS['wikisql']['data']
    table_path = [(os.path.join(data_dir, data_split + '.tables.jsonl'), data_split) for data_split in ['train', 'dev', 'test']]
    tables = transform_table_format(*table_path)
    output_path = os.path.join(data_dir, 'tables.json')
    json.dump(tables, open(output_path, 'w'), indent=4, ensure_ascii=False)

    tables = {db['db_id']: db for db in tables}
    for data_split in ['train', 'dev', 'test']:
        dataset_path = os.path.join(data_dir, data_split + '.jsonl')
        dataset = transform_dataset_format(dataset_path, tables)
        output_path = os.path.join(data_dir, data_split + '.json')
        json.dump(dataset, open(output_path, 'w'), indent=4, ensure_ascii=False)