#coding=utf8
import re
from asdl.asdl import ASDLGrammar
from asdl.asdl_ast import AbstractSyntaxTree
from preprocess.process_utils import UNIT_OP, UNIT_OP_NAME
from preprocess.dusql.postprocess import ValueProcessor
from preprocess.process_utils import State
from functools import wraps
from utils.constants import DEBUG

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
        self.value_processor = ValueProcessor(table_path, db_dir)

    @ignore_error
    def unparse(self, sql_ast: AbstractSyntaxTree, db: dict, value_candidates: list, entry: dict, *args, **kargs):
        self.sketch = kargs.pop('sketch', False)
        sql = self.unparse_sql(sql_ast, db, value_candidates, entry, *args, **kargs)
        sql = re.sub(r'\s+', ' ', sql)
        return sql, True

    def unparse_sql(self, sql_ast: AbstractSyntaxTree, db: dict, value_candidates: list, entry: dict, *args, **kargs):
        prod_name = sql_ast.production.constructor.name
        if prod_name != 'Single':
            left_sql = self.unparse_sql_unit(sql_ast[self.grammar.get_field_by_text('sql_unit left_sql_unit')][0].value, db, value_candidates, entry, *args, **kargs)
            right_sql = self.unparse_sql_unit(sql_ast[self.grammar.get_field_by_text('sql_unit right_sql_unit')][0].value, db, value_candidates, entry, *args, **kargs)
            return '%s %s %s' % (left_sql, prod_name.upper(), right_sql)
        else:
            return self.unparse_sql_unit(sql_ast[self.grammar.get_field_by_text('sql_unit sql_unit')][0].value, db, value_candidates, entry, *args, **kargs)

    def unparse_sql_unit(self, sql_ast: AbstractSyntaxTree, db: dict, value_candidates: list, entry: dict, *args, **kargs):
        from_field = sql_ast[self.grammar.get_field_by_text('from from')][0]
        from_str, is_nested = self.unparse_from(from_field.value, db, value_candidates, entry, *args, **kargs)
        from_str = 'FROM ' + from_str
        select_field = sql_ast[self.grammar.get_field_by_text('select select')][0]
        select_str = 'SELECT ' + self.unparse_select(select_field.value, db, is_nested, *args, **kargs)
        where_str, groupby_str, orderby_str = '', '', ''
        ctr_name = sql_ast.production.constructor.name
        if 'Where' in ctr_name:
            where_field = sql_ast[self.grammar.get_field_by_text('condition where')][0]
            where_str = 'WHERE ' + self.unparse_condition(where_field.value, db, value_candidates, entry, 'where', *args, **kargs)
        if 'GroupBy' in ctr_name:
            groupby_field = sql_ast[self.grammar.get_field_by_text('groupby groupby')][0]
            groupby_str = 'GROUP BY ' + self.unparse_groupby(groupby_field.value, db, value_candidates, entry, *args, **kargs)
        if 'OrderBy' in ctr_name:
            orderby_field = sql_ast[self.grammar.get_field_by_text('orderby orderby')][0]
            orderby_str = 'ORDER BY ' + self.unparse_orderby(orderby_field.value, db, value_candidates, entry, *args, **kargs)
        return ' '.join([select_str, from_str, where_str, groupby_str, orderby_str])

    def retrieve_column_name(self, col_id, db):
        if self.sketch:
            return db['table_names_original'][0] + '.' + db['column_names_original'][1][1]
        if col_id == 0:
            return 'TIME_NOW'
        elif col_id == 1:
            return '*'
        tab_id, col_name = db['column_names_original'][col_id - 1]
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
        col_id1 = 'TIME_NOW' if col_id1 == 0 else col_id1 - 1
        if agg_op == 'none':
            return col_name, (agg_op, unit_op, col_id1)
        else:
            return agg_op.upper() + ' ( ' + col_name + ' ) ', (agg_op, unit_op, col_id1)

    def unparse_col_unit_in_nested_from_sql(self, col_unit_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        col_id1 = col_unit_ast[self.grammar.get_field_by_text('col_id left_col_id')][0].value
        col_id2 = col_unit_ast[self.grammar.get_field_by_text('col_id right_col_id')][0].value
        if col_id1 < 2 or col_id2 < 2:
            return None
        if self.sketch:
            col_name1, col_name2 = db['column_names_original'][1][1], db['column_names_original'][1][1]
        else:
            col_name1, col_name2 = db['column_names_original'][col_id1 - 1][1], db['column_names_original'][col_id2 - 1][1]
        unit_op = col_unit_ast[self.grammar.get_field_by_text('unit_op unit_op')][0].value.production.constructor.name
        unit_op = UNIT_OP[UNIT_OP_NAME.index(unit_op)]
        col_name = 'a.' + col_name1 + ' ' + unit_op + ' b.' + col_name2
        return col_name

    def unparse_select(self, select_ast: AbstractSyntaxTree, db: dict, is_nested: bool, *args, **kargs):
        select_fields_list = select_ast[self.grammar.get_field_by_text('col_unit col_unit')]
        if is_nested:
            for col_field in select_fields_list:
                ctr_name = col_field.value.production.constructor.name
                if 'BinaryColumnUnit' in ctr_name:
                    col_name = self.unparse_col_unit_in_nested_from_sql(col_field.value, db, *args, **kargs)
                    if col_name is None: continue
                    return col_name
        select_items = []
        for col_unit_field in select_fields_list:
            col_unit_str = self.unparse_col_unit(col_unit_field.value, db, *args, **kargs)[0]
            select_items.append(col_unit_str)
        return ' , '.join(select_items)

    def unparse_from(self, from_ast: AbstractSyntaxTree, db: dict, value_candidates: list, entry: dict, *args, **kargs):
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
                from_cond_str = []
                for cond_field in cond_fields:
                    from_cond_str.append(self.unparse_table_join(cond_field.value, db))
                from_cond_str = ' ON ' + ' AND '.join(from_cond_str)
            return ' JOIN '.join(table_names) + from_cond_str, False
        else:
            left_ast = from_ast[self.grammar.get_field_by_text('sql left_from_sql')][0].value
            right_ast = from_ast[self.grammar.get_field_by_text('sql right_from_sql')][0].value
            return '( ' + self.unparse_sql(left_ast, db, value_candidates, entry, *args, **kargs) + ' ) a , ( ' + \
                self.unparse_sql(right_ast, db, value_candidates, entry, *args, **kargs) + ' ) b', True

    def unparse_table_join(self, join_ast: AbstractSyntaxTree, db: dict):
        fields = join_ast[self.grammar.get_field_by_text('col_id col_id')]
        col_id1, col_id2 = int(fields[0].value), int(fields[1].value)
        col_name1 = self.retrieve_column_name(col_id1, db)
        col_name2 = self.retrieve_column_name(col_id2, db)
        return ' = '.join([col_name1, col_name2])

    def unparse_groupby(self, groupby_ast: AbstractSyntaxTree, db: dict, value_candidates: list, entry: dict, *args, **kargs):
        having_str = ''
        groupby_field = groupby_ast[self.grammar.get_field_by_text('col_id col_id')][0]
        groupby_str = self.retrieve_column_name(groupby_field.value, db)
        ctr_name = groupby_ast.production.constructor.name
        if 'Having' in ctr_name:
            having_field = groupby_ast[self.grammar.get_field_by_text('condition having')][0]
            having_str = ' HAVING ' + self.unparse_condition(having_field.value, db, value_candidates, entry, 'having', *args, **kargs)
        return groupby_str + having_str

    def unparse_orderby(self, orderby_ast: AbstractSyntaxTree, db: dict, value_candidates: list, entry: dict, *args, **kargs):
        col_field = orderby_ast[self.grammar.get_field_by_text('col_unit col_unit')][0]
        orderby_str = self.unparse_col_unit(col_field.value, db, *args, **kargs)[0]
        order = orderby_ast[self.grammar.get_field_by_text('order order')][0].value.production.constructor.name.upper()
        limit_str = ''
        ctr_name = orderby_ast.production.constructor.name
        if 'Limit' in ctr_name:
            if self.sketch:
                limit_str = ' LIMIT 1'
            else:
                state = State('limit', 'none', '==', 'none', 0)
                value_id = orderby_ast[self.grammar.get_field_by_text('val_id limit')][0].value
                limit_str = ' LIMIT ' + self.value_processor.postprocess_value(value_id, value_candidates, db, state, entry)
        return orderby_str + ' ' + order + limit_str

    def unparse_condition(self, conds_ast: AbstractSyntaxTree, db: dict, value_candidates: list, entry: dict, clause: str, *args, **kargs):
        ctr_name = conds_ast.production.constructor.name
        if 'AndCondition' in ctr_name or 'OrCondition' in ctr_name:
            cond_fields = conds_ast[self.grammar.get_field_by_text('condition condition')]
            conds = []
            for cond_field in cond_fields:
                cond = self.unparse_condition(cond_field.value, db, value_candidates, entry, clause, *args, **kargs)
                conds.append(cond)
            conj = ' AND ' if 'AndCondition' in ctr_name else ' OR '
            return conj.join(conds)
        else:
            return self.unparse_cond(conds_ast, db, value_candidates, entry, clause, *args, **kargs)

    def unparse_cond(self, cond_ast: AbstractSyntaxTree, db: dict, value_candidates: list, entry: dict, clause: str, *args, **kargs):
        CMP_OP_MAPPING = {
            'Equal': '=', 'NotEqual': '!=', 'GreaterThan': '>', 'GreaterEqual': '>=',
            'LessThan': '<', 'LessEqual': '<=', 'Like': 'LIKE', 'In': 'IN', 'NotIn': 'NOT IN'
        }
        col_field = cond_ast[self.grammar.get_field_by_text('col_unit col_unit')][0]
        col_name, (agg_op, unit_op, col_id) = self.unparse_col_unit(col_field.value, db, *args, **kargs)
        cmp_op_str = cond_ast[self.grammar.get_field_by_text('cmp_op cmp_op')][0].value.production.constructor.name
        cmp_op = CMP_OP_MAPPING[cmp_op_str]
        state = State(clause, agg_op, cmp_op.lower(), unit_op, col_id)
        val_field = cond_ast[self.grammar.get_field_by_text('value value')][0]
        value_str = self.unparse_value(val_field.value, db, value_candidates, state, entry, *args, **kargs)
        return ' '.join([col_name, cmp_op, value_str])

    def unparse_value(self, val_ast: AbstractSyntaxTree, db: dict, value_candidates: list, state: State, entry: dict, *args, **kargs):
        ctr_name = val_ast.production.constructor.name
        if ctr_name == 'LiteralValue':
            if self.sketch:
                val_str = "'value'"
            else:
                val_id = int(val_ast[self.grammar.get_field_by_text('val_id val_id')][0].value)
                val_str = self.value_processor.postprocess_value(val_id, value_candidates, db, state, entry)
        else:
            val_field = val_ast[self.grammar.get_field_by_text('sql value_sql')][0]
            val_str = '( ' + self.unparse_sql(val_field.value, db, value_candidates, entry, *args, **kargs) + ' )'
        return val_str
