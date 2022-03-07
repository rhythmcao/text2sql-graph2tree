#coding=utf8
import re
from asdl.asdl import ASDLGrammar
from asdl.asdl_ast import AbstractSyntaxTree
from functools import wraps
from utils.constants import DEBUG
from preprocess.process_utils import UNIT_OP, UNIT_OP_NAME

def ignore_error(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if DEBUG: # allow error to be raised
            return func(*args, **kwargs)
        else: # prevent runtime error
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print('Something Error happened while unparsing:', e)
                return 'SELECT * FROM *', False
    return wrapper

class UnParser():

    def __init__(self, grammar: ASDLGrammar, table_path: str, db_dir: str):
        """ ASDLGrammar """
        super(UnParser, self).__init__()
        self.grammar = grammar

    @ignore_error
    def unparse(self, sql_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        if sql_ast is None:
            raise ValueError('root node is None')
        self.sketch = kargs.pop('sketch', False)
        sql = self.unparse_sql(sql_ast, db, *args, **kargs)
        sql = re.sub(r'\s+', ' ', sql)
        return sql, True

    def unparse_sql(self, sql_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        prod_name = sql_ast.production.constructor.name
        if prod_name != 'Single':
            left_sql = self.unparse_sql_unit(sql_ast[self.grammar.get_field_by_text('sql_unit left_sql_unit')][0].value, db, *args, **kargs)
            right_sql = self.unparse_sql_unit(sql_ast[self.grammar.get_field_by_text('sql_unit right_sql_unit')][0].value, db, *args, **kargs)
            return '%s %s %s' % (left_sql, prod_name.upper(), right_sql)
        else:
            return self.unparse_sql_unit(sql_ast[self.grammar.get_field_by_text('sql_unit sql_unit')][0].value, db, *args, **kargs)

    def unparse_sql_unit(self, sql_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        select_field = sql_ast[self.grammar.get_field_by_text('select select')][0]
        select_str = 'SELECT ' + self.unparse_select(select_field.value, db, *args, **kargs)
        from_field = sql_ast[self.grammar.get_field_by_text('from from')][0]
        from_str = 'FROM ' + self.unparse_from(from_field.value, db, *args, **kargs)
        where_str, groupby_str, orderby_str = '', '', ''
        ctr_name = sql_ast.production.constructor.name
        if 'Where' in ctr_name:
            where_field = sql_ast[self.grammar.get_field_by_text('condition where')][0]
            where_str = 'WHERE ' + self.unparse_condition(where_field.value, db, 'where', *args, **kargs)
        if 'GroupBy' in ctr_name:
            groupby_field = sql_ast[self.grammar.get_field_by_text('groupby groupby')][0]
            groupby_str = 'GROUP BY ' + self.unparse_groupby(groupby_field.value, db, *args, **kargs)
        if 'OrderBy' in ctr_name:
            orderby_field = sql_ast[self.grammar.get_field_by_text('orderby orderby')][0]
            orderby_str = 'ORDER BY ' + self.unparse_orderby(orderby_field.value, db, *args, **kargs)
        return ' '.join([select_str, from_str, where_str, groupby_str, orderby_str])

    def retrieve_column_name(self, col_id, db):
        if self.sketch:
            return db['table_names_original'][0] + '.' + db['column_names_original'][1][1]
        if col_id == 0:
            return '*'
        tab_id, col_name = db['column_names_original'][col_id]
        col_name = db['table_names_original'][tab_id] + '.' + col_name
        return col_name

    def unparse_col_unit(self, col_unit_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        agg_op = col_unit_ast[self.grammar.get_field_by_text('agg_op agg_op')][0].value.production.constructor.name.lower()
        ctr_name = col_unit_ast.production.constructor.name
        if 'Binary' in ctr_name:
            col_id1 = col_unit_ast[self.grammar.get_field_by_text('col_id left_col_id')][0].value
            col_name1 = self.retrieve_column_name(col_id1, db)
            col_id2 = col_unit_ast[self.grammar.get_field_by_text('col_id right_col_id')][0].value
            col_name2 = self.retrieve_column_name(col_id2, db)
            unit_op = col_unit_ast[self.grammar.get_field_by_text('unit_op unit_op')][0].value.production.constructor.name
            unit_op = UNIT_OP[UNIT_OP_NAME.index(unit_op)]
            col_name = col_name1 + ' ' + unit_op + ' ' + col_name2
        else:
            col_id1 = col_unit_ast[self.grammar.get_field_by_text('col_id col_id')][0].value
            col_name = self.retrieve_column_name(col_id1, db)
            unit_op = 'none'
        if agg_op == 'none':
            return col_name, (agg_op, unit_op, col_id1)
        else:
            return agg_op.upper() + '(' + col_name + ')', (agg_op, unit_op, col_id1)

    def unparse_select(self, select_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        select_fields_list = select_ast[self.grammar.get_field_by_text('col_unit col_unit')]
        select_items = []
        for col_unit_field in select_fields_list:
            col_unit_str = self.unparse_col_unit(col_unit_field.value, db, *args, **kargs)[0]
            select_items.append(col_unit_str)
        return ' , '.join(select_items)

    def unparse_from(self, from_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        ctr_name = from_ast.production.constructor.name
        if 'Table' in ctr_name:
            table_names, from_cond_str = [], ''
            table_fields = from_ast[self.grammar.get_field_by_text('tab_id tab_id')]
            if self.sketch:
                table_names = [db['table_names_original'][0]] * len(table_fields)
            else:
                for tab_field in table_fields:
                    table_name = db['table_names_original'][int(tab_field.value)]
                    table_names.append(table_name)
            if len(table_names) > 1:
                cond_fields = from_ast[self.grammar.get_field_by_text('join join')]
                cond_str = []
                for join_cond in cond_fields:
                    cond = join_cond.value
                    col_ids = cond[self.grammar.get_field_by_text('col_id col_id')]
                    col_id1, col_id2 = col_ids[0].value, col_ids[1].value
                    col_name1 = self.retrieve_column_name(col_id1, db)
                    col_name2 = self.retrieve_column_name(col_id2, db)
                    cond_str.append(col_name1 + ' = ' + col_name2)
                from_cond_str = ' ON ' + ' AND '.join(cond_str)
            return ' JOIN '.join(table_names) + from_cond_str
        else:
            return '( ' + self.unparse_sql(from_ast[self.grammar.get_field_by_text('sql from_sql')][0].value, db, *args, **kargs) + ' )'

    def unparse_groupby(self, groupby_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        groupby_str, having_str = [], ''
        groupby_fields = groupby_ast[self.grammar.get_field_by_text('col_id col_id')]
        for col_field in groupby_fields:
            col_id = col_field.value
            col_name = self.retrieve_column_name(col_id, db)
            groupby_str.append(col_name)
        groupby_str = ' , '.join(groupby_str)
        ctr_name = groupby_ast.production.constructor.name
        if 'Having' in ctr_name:
            having_field = groupby_ast[self.grammar.get_field_by_text('condition having')][0]
            having_str = ' HAVING ' + self.unparse_condition(having_field.value, db, 'having', *args, **kargs)
        return groupby_str + having_str

    def unparse_orderby(self, orderby_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        col_names = []
        col_fields = orderby_ast[self.grammar.get_field_by_text('col_unit col_unit')]
        for col_field in col_fields:
            col_name = self.unparse_col_unit(col_field.value, db, *args, **kargs)[0]
            col_names.append(col_name)
        orderby_str = ' , '.join(col_names)
        order = orderby_ast[self.grammar.get_field_by_text('order order')][0].value.production.constructor.name.upper()
        ctr_name = orderby_ast.production.constructor.name
        limit_str = ' LIMIT 1' if 'Limit' in ctr_name else ''
        return orderby_str + ' ' + order + limit_str

    def unparse_condition(self, conds_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        ctr_name = conds_ast.production.constructor.name
        if 'AndCondition' in ctr_name or 'OrCondition' in ctr_name:
            cond_fields = conds_ast[self.grammar.get_field_by_text('condition condition')]
            conds = []
            for cond_field in cond_fields:
                cond = self.unparse_condition(cond_field.value, db, *args, **kargs)
                conds.append(cond)
            conj = ' AND ' if 'AndCondition' in ctr_name else ' OR '
            return conj.join(conds)
        else:
            return self.unparse_cond(conds_ast, db, *args, **kargs)

    def unparse_cond(self, cond_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        CMP_OP_MAPPING = {
            'Between': 'BETWEEN', 'Equal': '=', 'NotEqual': '!=', 'GreaterThan': '>', 'GreaterEqual': '>=',
            'LessThan': '<', 'LessEqual': '<=', 'Like': 'LIKE', 'NotLike': 'NOT LIKE', 'In': 'IN', 'NotIn': 'NOT IN'
        }
        col_field = cond_ast[self.grammar.get_field_by_text('col_unit col_unit')][0]
        col_name, (agg_op, unit_op, col_id) = self.unparse_col_unit(col_field.value, db, *args, **kargs)
        ctr_name = cond_ast.production.constructor.name
        value_str = '"value"'
        if 'SQL' in ctr_name:
            value_str = '( ' + self.unparse_sql(cond_ast[self.grammar.get_field_by_text('sql cond_sql')][0].value, db, *args, **kargs) + ' )'
        cmp_op_str = cond_ast[self.grammar.get_field_by_text('cmp_op cmp_op')][0].value.production.constructor.name
        cmp_op = CMP_OP_MAPPING[cmp_op_str]
        if cmp_op == 'BETWEEN':
            value_str += ' AND "value"'
        return ' '.join([col_name, cmp_op, value_str])