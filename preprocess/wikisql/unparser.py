#coding=utf8
import sys, os, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

AGG_MAP = {
    0: lambda s: s,
    1: lambda s: 'MAX ( ' + s + ' ) ',
    2: lambda s: 'MIN ( ' + s + ' ) ',
    3: lambda s: 'COUNT ( ' + s + ' ) ',
    4: lambda s: 'SUM ( ' + s + ' ) ',
    5: lambda s: 'AVG ( ' + s + ' ) '
}

CMP_MAP = (' = ', ' > ', ' < ')

def unparse(sql: dict, db: dict):
    column_names = db['column_names_original']
    select_str = 'SELECT ' + AGG_MAP[sql['agg']](column_names[sql['sel']][1])
    where_conds = []
    for cond in sql['conds']:
        col_id, cmp_id, val = cond
        if type(val) == str: val = '"' + val + '"'
        cond = CMP_MAP[cmp_id].join([column_names[col_id][1], str(val)])
        where_conds.append(cond)
    where_str = ' WHERE ' + ' AND '.join(where_conds) if len(where_conds) > 0 else ''
    return select_str + where_str

def reconstruct_dataset(dataset_path, output_path, tables):
    dataset = json.load(open(dataset_path, 'r'))
    output_sqls = []
    for ex in dataset:
        s = unparse(ex['sql'], tables[ex['db_id']])
        output_sqls.append((s, ex['db_id']))

    with open(output_path, 'w') as of:
        for l, d in output_sqls:
            of.write(l + '\t' + d + '\n')
    return

def load_tables(table_path):
    table_list = json.load(open(table_path, 'r'))
    tables = {}
    for table in table_list:
        tables[table['db_id']] = table
    return tables

if __name__ == '__main__':

    tables = load_tables('data/wikisql/tables.json')
    reconstruct_dataset('data/wikisql/train.json', 'data/wikisql/train_gold.sql', tables)
    reconstruct_dataset('data/wikisql/dev.json', 'data/wikisql/dev_gold.sql', tables)
    reconstruct_dataset('data/wikisql/test.json', 'data/wikisql/test_gold.sql', tables)