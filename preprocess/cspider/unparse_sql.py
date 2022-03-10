#coding=utf8
import sys, os, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from preprocess.process_utils import is_number

AGG_MAP = {
    0: lambda s: s,
    1: lambda s: 'MAX(' + s + ')',
    2: lambda s: 'MIN(' + s + ')',
    3: lambda s: 'COUNT(' + s + ')',
    4: lambda s: 'SUM(' + s + ')',
    5: lambda s: 'AVG(' + s + ')'
}
class UnparseSQLDict():

    def unparse(self, sql: dict, db: dict):
        if sql['intersect']:
            return self.unparse_sql(sql, db) + ' INTERSECT ' + self.unparse(sql['intersect'], db)
        if sql['union']:
            return self.unparse_sql(sql, db) + ' UNION ' + self.unparse(sql['union'], db)
        if sql['except']:
            return self.unparse_sql(sql, db) + ' EXCEPT ' + self.unparse(sql['except'], db)
        return self.unparse_sql(sql, db)

    def unparse_sql(self, sql: dict, db: dict):
        select_str = 'SELECT ' + self.unparse_select(sql['select'], db)
        from_str = 'FROM ' + self.unparse_from(sql['from'], db)
        where_str = 'WHERE ' + self.unparse_conds(sql['where'], db) if sql['where'] else ''
        groupby_str = 'GROUP BY ' + self.unparse_groupby(sql['groupBy'], db) if sql['groupBy'] else ''
        having_str = 'HAVING ' + self.unparse_conds(sql['having'], db) if sql['having'] else ''
        orderby_str = 'ORDER BY ' + self.unparse_orderby(sql['orderBy'], db) if sql['orderBy'] else ''
        limit_str = 'LIMIT ' + str(sql['limit']) if sql['limit'] else ''
        return ' '.join([select_str, from_str, where_str, groupby_str, having_str, orderby_str, limit_str])

    def unparse_select(self, select, db):
        distinct_flag, select_str = 'DISTINCT ' if select[0] else '', []
        for agg_id, val_unit in select[1]:
            select_str.append(AGG_MAP[agg_id](self.unparse_val_unit(val_unit, db)))
        return distinct_flag + ' , '.join(select_str)

    def unparse_from(self, from_clause, db):
        table_units = from_clause['table_units']
        if table_units[0][0] == 'sql':
            return '( ' + self.unparse(table_units[0][1], db) + ' )'
        else:
            table_names = db['table_names_original']
            table_list = [table_names[int(table_unit[1])] for table_unit in table_units]
            if len(table_list) > 1:
                return ' JOIN '.join(table_list) + ' ON ' + self.unparse_conds(from_clause['conds'], db)
            else:
                return table_list[0]

    def unparse_groupby(self, groupby, db):
        return ' , '.join([self.unparse_col_unit(col_unit, db) for col_unit in groupby])

    def unparse_orderby(self, orderby, db):
        return ' , '.join([self.unparse_val_unit(val_unit, db) for val_unit in orderby[1]]) + ' ' + orderby[0].upper()

    def unparse_conds(self, conds: list, db: dict):
        if not conds: return ''
        cond_str = [self.unparse_cond(cond, db) if cond not in ['and', 'or'] else cond.upper() for cond in conds]
        return ' '.join(cond_str)

    def unparse_cond(self, cond: list, db: dict):
        not_op, cmp_op, val_unit, val1, val2 = cond
        val_str = self.unparse_val_unit(val_unit, db)
        val1_str = self.unparse_val(val1, db)
        if not_op:
            assert cmp_op in [8, 9]
            not_str = 'NOT '
        else: not_str = ''
        if cmp_op == 1:
            val2_str = self.unparse_val(val2, db)
            return val_str + ' BETWEEN ' + val1_str + ' AND ' + val2_str
        cmp_map = ('NOT', 'BETWEEN', '=', '>', '<', '>=', '<=', '!=', 'IN', 'LIKE', 'IS', 'EXISTS')
        cmp_str = cmp_map[cmp_op]
        return val_str + ' ' + not_str + cmp_str + ' ' + val1_str

    def unparse_val(self, val, db):
        if type(val) in [str, int, float, bool]:
            if is_number(val):
                val_str = str(int(val)) if float(val) % 1 == 0 else str(float(val))
            else:
                val_str = str(val)
        elif type(val) in [list, tuple]:
            val_str = self.unparse_col_unit(val, db)
        else:
            assert type(val) == dict
            val_str = '( ' + self.unparse(val, db) + ' )'
        return val_str

    def unparse_val_unit(self, val_unit: list, db: dict):
        unit_op, col_unit1, col_unit2 = val_unit
        if unit_op == 0:
            return self.unparse_col_unit(col_unit1, db)
        else:
            unit_map = ('none', ' - ', ' + ', ' * ', ' / ')
            return unit_map[unit_op].join([self.unparse_col_unit(col_unit1, db), self.unparse_col_unit(col_unit2, db)])

    def unparse_col_unit(self, col_unit: list, db: dict):
        agg_id, col_id, dis = col_unit
        tab_name = '' if col_id == 0 else db['table_names_original'][db['column_names_original'][col_id][0]] + '.'
        col_name = tab_name + db['column_names_original'][col_id][1]
        col_name = 'DISTINCT ' + col_name if dis else col_name
        return AGG_MAP[agg_id](col_name)

def reconstruct_dataset(unparser, dataset_path, output_path, tables):
    dataset = json.load(open(dataset_path, 'r'))
    output_sqls = []
    for ex in dataset:
        db_id = ex['db_id']
        s = unparser.unparse(ex['sql'], tables[db_id])
        output_sqls.append((ex['question_id'], s))

    with open(output_path, 'w') as of:
        for qid, l in output_sqls:
            of.write(str(qid) + '\t' + l + '\n')

def load_tables(table_path):
    table_list = json.load(open(table_path, 'r'))
    tables = {}
    for table in table_list:
        tables[table['db_id']] = table
    return tables

if __name__ == '__main__':

    tables = load_tables('data/cspider/tables.json')
    unparser = UnparseSQLDict()
    reconstruct_dataset(unparser, 'data/cspider/train.json', 'train_recovered.sql', tables)
    reconstruct_dataset(unparser, 'data/cspider/dev.json', 'dev_recovered.sql', tables)
