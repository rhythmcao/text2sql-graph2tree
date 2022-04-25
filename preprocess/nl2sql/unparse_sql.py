#coding=utf8
import sys, os, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

AGG_MAP = {
    0: lambda s: s,
    1: lambda s: 'AVG ( ' + s + ' ) ',
    2: lambda s: 'MAX ( ' + s + ' ) ',
    3: lambda s: 'MIN ( ' + s + ' ) ',
    4: lambda s: 'COUNT ( ' + s + ' ) ',
    5: lambda s: 'SUM ( ' + s + ' ) '
}

CMP_MAP = (' > ', ' < ', ' == ', ' != ')

COND = ('', ' and ', ' or ')


class UnparseSQLDict():

    def unparse(self, sql: dict, db: dict):
        column_names = db.get('column_names_original', 'column_names')
        select_cols = []
        for agg_id, col_id in zip(sql['agg'], sql['sel']):
            sel = AGG_MAP[agg_id](column_names[col_id][1])
            select_cols.append(sel)
        select_str = 'SELECT ' + ' , '.join(select_cols)
        conj = COND[sql['cond_conn_op']]
        where_conds = []
        for cond in sql['conds']:
            col_id, cmp_id, val = cond
            cond = CMP_MAP[cmp_id].join([column_names[col_id][1], '"' + val + '"'])
            where_conds.append(cond)
        where_str = ' WHERE ' + conj.join(where_conds)
        return select_str + where_str

def reconstruct_dataset(unparser, dataset_path, output_path, tables):
    dataset = json.load(open(dataset_path, 'r'))
    output_sqls = []
    for ex in dataset:
        db_id = ex['db_id'] if 'db_id' in ex else ex['table_id']
        sql = ex['sql'] if 'sql' in ex else ex
        s = unparser.unparse(sql, tables[db_id])
        output_sqls.append((s, db_id))

    with open(output_path, 'w') as of:
        for idx, (l, d) in enumerate(output_sqls):
            of.write('qid%s' % (idx + 1) + '\t' + l + '\t' + d + '\n')

def load_tables(table_path):
    table_list = json.load(open(table_path, 'r'))
    tables = {}
    for table in table_list:
        tables[table['db_id']] = table
    return tables

if __name__ == '__main__':

    tables = load_tables('data/nl2sql/tables.json')
    unparser = UnparseSQLDict()
    reconstruct_dataset(unparser, 'data/nl2sql/train.json', 'train_recovered.sql', tables)
    reconstruct_dataset(unparser, 'data/nl2sql/dev.json', 'dev_recovered.sql', tables)
    # reconstruct_dataset(unparser, 'data/nl2sql/test_labels.json', 'test_gold.sql2', tables)