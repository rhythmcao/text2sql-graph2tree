#coding=utf8
from asdl.asdl import ASDLConstructor, ASDLGrammar
from asdl.asdl_ast import AbstractSyntaxTree
from functools import wraps
from utils.constants import DEBUG
from preprocess.process_utils import AGG_OP, UNIT_OP_NAME
from itertools import chain, repeat

CMP_OP = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
CMP_OP_NAME = {
    '=': 'Equal', '>': 'GreaterThan', '<': 'LessThan', '>=': 'GreaterEqual', '<=': 'LessEqual',
    '!=': 'NotEqual', 'in': 'In', 'not in': 'NotIn', 'like': 'Like', 'not like': 'NotLike',
    'between': 'Between'
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
                    "select": [False, [(0, [0, [0, 0, False], None])]],
                    "from": {'table_units': [('table_unit', 0)], 'conds': []},
                    "where": [], "groupBy": [], "orderBy": [], "having": [], "limit": None,
                    "intersect": [], "union": [], "except": []
                }
                ast_node = self.parse_sql(error_sql)
                return ast_node
    return wrapper

class Parser():
    """ Parse a sql dict into AbstractSyntaxTree object according to predefined grammar rules.
    """
    def __init__(self, grammar: ASDLGrammar):
        super(Parser, self).__init__()
        self.grammar = grammar

    @ignore_error
    def parse(self, sql_json: dict, sql_values: set = None):
        """
        @params:
            sql_json: the 'sql' field of each data sample
            sql_values: set of SQLValue, pre-retrieved values that will be used
        @return:
            ast_node: AbstractSyntaxTree of sql
        """
        ast_node = self.parse_sql(sql_json)
        return ast_node

    def parse_sql(self, sql: dict):
        """ Determine whether sql has intersect/union/except,
        at most one in the current dict
        """
        for choice in ['intersect', 'union', 'except']:
            if sql[choice]:
                ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(choice.title()))
                nested_sql = sql[choice]
                sql_field1 = ast_node[self.grammar.get_field_by_text('sql_unit left_sql_unit')][0]
                sql_field1.add_value(self.parse_sql_unit(sql))
                sql_field2 = ast_node[self.grammar.get_field_by_text('sql_unit right_sql_unit')][0]
                sql_field2.add_value(self.parse_sql_unit(nested_sql))
                return ast_node
        ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('Single'))
        sql_field = ast_node[self.grammar.get_field_by_text('sql_unit sql_unit')][0]
        sql_field.add_value(self.parse_sql_unit(sql))
        return ast_node

    def parse_sql_unit(self, sql: dict):
        """ Parse a single sql unit, determine the existence of different clauses
        """
        ctr_name = 'FromSelect'
        if sql['where']: ctr_name += 'Where'
        if sql['groupBy']: ctr_name += 'GroupBy'
        if sql['orderBy']: ctr_name += 'OrderBy'
        ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(ctr_name))
        select_field = ast_node[self.grammar.get_field_by_text('select select')][0]
        select_field.add_value(self.parse_select(sql['select'][1]))
        from_field = ast_node[self.grammar.get_field_by_text('from from')][0]
        from_field.add_value(self.parse_from(sql['from']))
        if sql['where']:
            where_field = ast_node[self.grammar.get_field_by_text('condition where')][0]
            where_field.add_value(self.parse_condition(sql['where']))
        if sql['groupBy']:
            groupby_field = ast_node[self.grammar.get_field_by_text('groupby groupby')][0]
            groupby_field.add_value(self.parse_groupby(sql['groupBy'], sql['having']))
        if sql['orderBy']:
            orderby_field = ast_node[self.grammar.get_field_by_text('orderby orderby')][0]
            orderby_field.add_value(self.parse_orderby(sql['orderBy'], sql['limit']))
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
        if col_unit[1] != 0: # unit_op is not none
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('BinaryColumnUnit'))
            agg_op_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(AGG_OP[col_unit[0]].title()))
            ast_node[self.grammar.get_field_by_text('agg_op agg_op')][0].add_value(agg_op_node)
            unit_op_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(UNIT_OP_NAME[col_unit[1]]))
            ast_node[self.grammar.get_field_by_text('unit_op unit_op')][0].add_value(unit_op_node)
            ast_node[self.grammar.get_field_by_text('col_id left_col_id')][0].add_value(int(col_unit[2]))
            ast_node[self.grammar.get_field_by_text('col_id right_col_id')][0].add_value(int(col_unit[3]))
        else:
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('UnaryColumnUnit'))
            agg_op_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(AGG_OP[col_unit[0]].title()))
            ast_node[self.grammar.get_field_by_text('agg_op agg_op')][0].add_value(agg_op_node)
            ast_node[self.grammar.get_field_by_text('col_id col_id')][0].add_value(int(col_unit[2]))
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

    def parse_from(self, from_clause: dict):
        """ Ignore from conditions, since it is not evaluated in evaluation script
        """
        table_units, from_conds = from_clause['table_units'], from_clause['conds']
        t = table_units[0][0]
        if t == 'table_unit':
            ctr_name = 'FromTable' + ASDLConstructor.number2word[len(table_units)] + ASDLConstructor.number2word[(len(from_conds) + 1) // 2]
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(ctr_name))
            table_fields = ast_node[self.grammar.get_field_by_text('tab_id tab_id')]
            for idx, (_, tab_id) in enumerate(table_units):
                table_fields[idx].add_value(int(tab_id))
            if len(from_conds) > 0:
                idx, cond_fields = 0, ast_node[self.grammar.get_field_by_text('join join')]
                for cond in from_conds:
                    if cond in ['and', 'or']: continue
                    _, _, val_unit, val1, _ = cond
                    col_id1, col_id2 = val_unit[1][1], val1[1]
                    cur_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('ColumnJoin'))
                    col_field1 = cur_node[self.grammar.get_field_by_text('col_id col_id')][0]
                    col_field1.add_value(int(col_id1))
                    col_field2 = cur_node[self.grammar.get_field_by_text('col_id col_id')][1]
                    col_field2.add_value(int(col_id2))
                    cond_fields[idx].add_value(cur_node)
                    idx += 1
        else:
            assert t == 'sql'
            from_sql = table_units[0][1]
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('FromSQL'))
            ast_node[self.grammar.get_field_by_text('sql from_sql')][0].add_value(self.parse_sql(from_sql))
        return ast_node

    def parse_groupby(self, groupby_clause: list, having_clause: list):
        ctr_name = 'GroupByColumn' + ASDLConstructor.number2word[len(groupby_clause)] if not having_clause else \
            'GroupByHavingColumn' + ASDLConstructor.number2word[len(groupby_clause)]
        ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(ctr_name))
        groupby_fields = ast_node[self.grammar.get_field_by_text('col_id col_id')]
        for idx, col_unit in enumerate(groupby_clause):
            groupby_fields[idx].add_value(int(col_unit[1]))
        if having_clause:
            having_field = ast_node[self.grammar.get_field_by_text('condition having')][0]
            having_field.add_value(self.parse_condition(having_clause))
        return ast_node

    def parse_orderby(self, orderby_clause: list, limit: int):
        ctr_name = 'OrderByLimitColumn' + ASDLConstructor.number2word[len(orderby_clause[1])] if limit else \
            'OrderByColumn' + ASDLConstructor.number2word[len(orderby_clause[1])]
        ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(ctr_name))
        col_fields = ast_node[self.grammar.get_field_by_text('col_unit col_unit')]
        for idx, val_unit in enumerate(orderby_clause[1]):
            col_unit = self.convert_val_unit(val_unit)
            col_node = self.parse_col_unit(col_unit)
            col_fields[idx].add_value(col_node)
        order_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(orderby_clause[0].title()))
        ast_node[self.grammar.get_field_by_text('order order')][0].add_value(order_node)
        return ast_node

    def parse_condition(self, condition: list):
        and_conds, or_conds = [], []
        prev_conj, prev_cond = 'and', condition[0]
        if len(condition) > 1:
            for cond in condition[1:]:
                if cond in ['and', 'or']:
                    prev_conj = cond
                    if prev_conj == 'and':
                        and_conds.append(prev_cond)
                    else:
                        or_conds.append(prev_cond)
                else: prev_cond = cond
        if prev_conj == 'and': and_conds.append(prev_cond)
        else: or_conds.append(prev_cond)
        if len(and_conds) > 0 and len(or_conds) > 0:
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('AndConditionTwo'))
            condition_fields = ast_node[self.grammar.get_field_by_text('condition condition')]
            new_and_conds = list(chain.from_iterable(zip(and_conds, repeat("and"))))[:-1]
            condition_fields[0].add_value(self.parse_condition(new_and_conds))
            new_or_conds = list(chain.from_iterable(zip(or_conds, repeat("or"))))[:-1]
            condition_fields[1].add_value(self.parse_condition(new_or_conds))
        elif len(and_conds) > 0:
            if len(and_conds) == 1: ast_node = self.parse_cond(and_conds[0])
            else:
                ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('AndCondition' + ASDLConstructor.number2word[len(and_conds)]))
                condition_fields = ast_node[self.grammar.get_field_by_text('condition condition')]
                for idx, cond in enumerate(and_conds):
                    condition_fields[idx].add_value(self.parse_cond(cond))
        else:
            assert len(or_conds) > 1, 'Number of OR conditions should be larger than one'
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('OrCondition' + ASDLConstructor.number2word[len(or_conds)]))
            condition_fields = ast_node[self.grammar.get_field_by_text('condition condition')]
            for idx, cond in enumerate(or_conds):
                condition_fields[idx].add_value(self.parse_cond(cond))
        return ast_node

    def parse_cond(self, cond: list):
        not_op, cmp_op, val_unit, val1, val2 = cond
        ctr_name = 'CmpSQLCondition' if type(val1) == dict else 'CmpCondition'
        ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(ctr_name))
        col_node = self.parse_col_unit(self.convert_val_unit(val_unit))
        ast_node[self.grammar.get_field_by_text('col_unit col_unit')][0].add_value(col_node)
        if type(val1) == dict:
            sql_node = self.parse_sql(val1)
            ast_node[self.grammar.get_field_by_text('sql cond_sql')][0].add_value(sql_node)
        not_op = 'not ' if not_op else ''
        ctr_name = CMP_OP_NAME[not_op + CMP_OP[cmp_op]]
        cmp_op_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(ctr_name))
        ast_node[self.grammar.get_field_by_text('cmp_op cmp_op')][0].add_value(cmp_op_node)
        return ast_node