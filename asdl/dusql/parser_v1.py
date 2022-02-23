#coding=utf8
from asdl.asdl import ASDLConstructor, ASDLGrammar
from asdl.asdl_ast import AbstractSyntaxTree
from functools import wraps
from utils.constants import DEBUG
from preprocess.process_utils import SQLValue, State
from preprocess.dusql.value_utils import AGG_OP, CMP_OP, UNIT_OP

UNIT_OP_NAME = ('', 'Minus', 'Plus', 'Times', 'Divide')
CMP_OP_NAME = {
    '==': 'Equal', '>': 'GreaterThan', '<': 'LessThan', '>=': 'GreaterEqual', '<=': 'LessEqual',
    '!=': 'NotEqual', 'in': 'In', 'not_in': 'NotIn', 'like': 'Like'
}

def ignore_error(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if DEBUG: # allow error to be raised
            return func(self, *args, **kwargs)
        else: # prevent runtime error
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                print('Something Error happened while parsing:', e)
                # if fail to parse, just return SELECT * FROM table(id=0)
                error_sql = {
                    "select": [(0, [0, [0, 0, False], None])],
                    "from": {'table_units': [('table_unit', 0)], 'conds': []},
                    "where": [], "groupBy": [], "orderBy": [], "having": [], "limit": None,
                    "intersect": [], "union": [], "except": []
                }
                ast_node = self.parse_sql(error_sql, set(), track='')
                return ast_node
    return wrapper

class Parser():
    """ Parse a sql dict into AbstractSyntaxTree object according to predefined grammar rules.
    """
    def __init__(self, grammar: ASDLGrammar):
        super(Parser, self).__init__()
        self.grammar = grammar

    @ignore_error
    def parse(self, sql_json: dict, sql_values: set):
        """
        @params:
            sql_json: the 'sql' field of each data sample
            sql_values: set of SQLValue, pre-retrieved values that will be used
            track: used to differentiate values by recording the value path
        @return:
            ast_node: AbstractSyntaxTree of sql
        """
        ast_node = self.parse_sql(sql_json, sql_values, track='')
        return ast_node

    def parse_sql(self, sql: dict, sql_values: set, track: str):
        """ Determine whether sql has intersect/union/except,
        at most one in the current dict
        """
        for choice in ['intersect', 'union', 'except']:
            if sql[choice]:
                ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(choice.title()))
                nested_sql = sql[choice]
                sql_field1 = ast_node[self.grammar.get_field_by_text('sql_unit left_sql_unit')][0]
                sql_field1.add_value(self.parse_sql_unit(sql, sql_values, track))
                sql_field2 = ast_node[self.grammar.get_field_by_text('sql_unit right_sql_unit')][0]
                sql_field2.add_value(self.parse_sql_unit(nested_sql, sql_values, track + '->' + choice))
                return ast_node
        ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('Single'))
        sql_field = ast_node[self.grammar.get_field_by_text('sql_unit sql_unit')][0]
        sql_field.add_value(self.parse_sql_unit(sql, sql_values, track))
        return ast_node

    def parse_sql_unit(self, sql: dict, sql_values: set, track: str):
        """ Parse a single sql unit, determine the existence of different clauses
        """
        ctr_name = 'FromSelect'
        if sql['where']: ctr_name += 'Where'
        if sql['groupBy']: ctr_name += 'GroupBy'
        if sql['orderBy']: ctr_name += 'OrderBy'
        ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(ctr_name))
        select_field = ast_node[self.grammar.get_field_by_text('select select')][0]
        select_field.add_value(self.parse_select(sql['select']))
        from_field = ast_node[self.grammar.get_field_by_text('from from')][0]
        from_field.add_value(self.parse_from(sql['from'], sql_values, track))
        if sql['where']:
            where_field = ast_node[self.grammar.get_field_by_text('condition where')][0]
            where_field.add_value(self.parse_condition(sql['where'], sql_values, track=track + '->where', int_as_column=False))
        if sql['groupBy']:
            groupby_field = ast_node[self.grammar.get_field_by_text('groupby groupby')][0]
            groupby_field.add_value(self.parse_groupby(sql['groupBy'], sql['having'], sql_values, track))
        if sql['orderBy']:
            orderby_field = ast_node[self.grammar.get_field_by_text('orderby orderby')][0]
            orderby_field.add_value(self.parse_orderby(sql['orderBy'], sql['limit'], sql_values, track))
        return ast_node

    def convert_val_unit(self, val_unit):
        if val_unit[0] == 0:
            return [val_unit[1][0], val_unit[0], val_unit[1][1], None]
        return [val_unit[1][0], val_unit[0], val_unit[1][1], val_unit[2][1]]

    def convert_agg_val_unit(self, agg, val_unit):
        if val_unit[0] != 0:
            return [agg, val_unit[0], val_unit[1][1], val_unit[2][1]]
        else:
            return [agg, val_unit[0], val_unit[1][1], None]

    def parse_col_unit(self, col_unit):
        # remember to assign 0 for TIME_NOW, and add 1 for all other column ids
        if col_unit[1] != 0: # unit_op is not none
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('BinaryColumnUnit'))
            agg_op_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(AGG_OP[col_unit[0]].title()))
            ast_node[self.grammar.get_field_by_text('agg_op agg_op')][0].add_value(agg_op_node)
            unit_op_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(UNIT_OP_NAME[col_unit[1]]))
            ast_node[self.grammar.get_field_by_text('unit_op unit_op')][0].add_value(unit_op_node)
            col_id = 0 if type(col_unit[2]) == str else 1 + int(col_unit[2])
            ast_node[self.grammar.get_field_by_text('col_id left_col_id')][0].add_value(col_id)
            col_id = 0 if type(col_unit[3]) == str else 1 + int(col_unit[3])
            ast_node[self.grammar.get_field_by_text('col_id right_col_id')][0].add_value(col_id)
        else:
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('UnaryColumnUnit'))
            agg_op_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(AGG_OP[col_unit[0]].title()))
            ast_node[self.grammar.get_field_by_text('agg_op agg_op')][0].add_value(agg_op_node)
            col_id = 0 if type(col_unit[2]) == str else 1 + int(col_unit[2])
            ast_node[self.grammar.get_field_by_text('col_id col_id')][0].add_value(col_id)
        return ast_node

    def parse_select(self, agg_val_list: list):
        ctr_name = 'SelectColumn' + ASDLConstructor.number2word[len(agg_val_list)]
        ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(ctr_name))
        ast_fields = ast_node[self.grammar.get_field_by_text('col_unit col_unit')]
        for idx, (agg, val_unit) in enumerate(agg_val_list):
            col_unit = self.convert_agg_val_unit(agg, val_unit)
            col_node = self.parse_col_unit(col_unit)
            ast_fields[idx].add_value(col_node)
        return ast_node

    def parse_from(self, from_clause: dict, sql_values: set, track: str):
        """ Ignore from conditions, since it is not evaluated in evaluation script
        """
        table_units, from_conds = from_clause['table_units'], from_clause['conds']
        t = table_units[0][0]
        if t == 'table_unit':
            ctr_name = 'FromTable' + ASDLConstructor.number2word[len(table_units)]
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(ctr_name))
            table_fields = ast_node[self.grammar.get_field_by_text('tab_id tab_id')]
            for idx, (_, tab_id) in enumerate(table_units):
                table_fields[idx].add_value(int(tab_id))
            if len(from_conds) > 0:
                cond_field = ast_node[self.grammar.get_field_by_text('condition from')][0]
                cond_field.add_value(self.parse_condition(from_conds, sql_values, track=track + '->from', int_as_column=True))
        else:
            assert t == 'sql'
            left_from_sql, right_from_sql = table_units[0][1], table_units[1][1]
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('FromSQL'))
            ast_node[self.grammar.get_field_by_text('sql left_from_sql')][0].add_value(self.parse_sql(left_from_sql, sql_values, track + '->from'))
            ast_node[self.grammar.get_field_by_text('sql right_from_sql')][0].add_value(self.parse_sql(right_from_sql, sql_values, track + '->from'))
        return ast_node

    def parse_groupby(self, groupby_clause: list, having_clause: list, sql_values: set, track: str):
        ctr_name = 'GroupByColumn' if not having_clause else 'GroupByHavingColumn'
        ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(ctr_name))
        groupby_field = ast_node[self.grammar.get_field_by_text('col_id col_id')][0]
        groupby_field.add_value(int(groupby_clause[0][1]) + 1)
        if having_clause:
            having_field = ast_node[self.grammar.get_field_by_text('condition having')][0]
            having_field.add_value(self.parse_condition(having_clause, sql_values, track=track + '->having', int_as_column=False))
        return ast_node

    def parse_orderby(self, orderby_clause: list, limit: int, sql_values: set, track: str):
        ctr_name = 'OrderByLimitColumn' if limit else 'OrderByColumn'
        ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(ctr_name))
        col_field = ast_node[self.grammar.get_field_by_text('col_unit col_unit')][0]
        agg, val_unit = orderby_clause[1][0]
        col_field.add_value(self.parse_col_unit(self.convert_agg_val_unit(agg, val_unit)))
        order_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(orderby_clause[0].title()))
        ast_node[self.grammar.get_field_by_text('order order')][0].add_value(order_node)
        if limit:
            sqlvalue = SQLValue(str(limit), State(track + '->limit', 'none', '==', 'none', 0))
            for v in sql_values:
                if v == sqlvalue:
                    ast_node[self.grammar.get_field_by_text('val_id limit')][0].add_value(int(v.value_id))
                    break
            else:
                raise ValueError('Unable to find LIMIT %s in extracted values' % (limit))
        return ast_node

    def parse_condition(self, condition: list, sql_values: set, track: str, int_as_column: bool = False):
        if len(condition) > 1:
            ctr_name = 'OrConditionTwo' if condition[1] == 'or' else 'AndConditionTwo'
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(ctr_name))
            condition_fields = ast_node[self.grammar.get_field_by_text('condition condition')]
            idx = 0
            for cond in condition:
                if cond in ['and', 'or']: continue
                condition_fields[idx].add_value(self.parse_cond(cond, sql_values, track, int_as_column))
                idx += 1
        else:
            ast_node = self.parse_cond(condition[0], sql_values, track, int_as_column)
        return ast_node

    def parse_cond(self, cond: list, sql_values: set, track: str, int_as_column: bool = False):
        agg_op, cmp_op, val_unit, val1, _ = cond
        unit_op, col_id = val_unit[0], val_unit[1][1]
        ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('CmpCondition'))
        ctr_name = CMP_OP_NAME[CMP_OP[cmp_op]]
        cmp_op_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(ctr_name))
        ast_node[self.grammar.get_field_by_text('cmp_op cmp_op')][0].add_value(cmp_op_node)
        col_node = self.parse_col_unit(self.convert_agg_val_unit(agg_op, val_unit))
        ast_node[self.grammar.get_field_by_text('col_unit col_unit')][0].add_value(col_node)
        state = State(track, AGG_OP[agg_op], CMP_OP[cmp_op], UNIT_OP[unit_op], col_id)
        value_node = self.parse_value(val1, sql_values, track, state, int_as_column)
        ast_node[self.grammar.get_field_by_text('value value')][0].add_value(value_node)
        return ast_node

    def parse_value(self, val, sql_values, track, state, int_as_column):
        if type(val) == dict: # nested sql
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('SQLValue'))
            ast_node[self.grammar.get_field_by_text('sql value_sql')][0].add_value(self.parse_sql(val, sql_values, track))
        elif type(val) == int and int_as_column: # column
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('ColumnValue'))
            col_id = 1 + int(val)
            ast_node[self.grammar.get_field_by_text('col_id col_id')][0].add_value(col_id)
        else: # literal value
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('LiteralValue'))
            sql_value = SQLValue(str(val), state)
            for v in sql_values:
                if v == sql_value:
                    ast_node[self.grammar.get_field_by_text('val_id val_id')][0].add_value(int(v.value_id))
                    break
            else:
                raise ValueError('Unable to find %s value %s in extracted values' % (track.split('->')[-1], val))
        return ast_node
