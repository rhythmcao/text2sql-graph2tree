#coding=utf8
from asdl.asdl import ASDLConstructor, ASDLGrammar
from asdl.asdl_ast import AbstractSyntaxTree
from functools import wraps
from utils.constants import DEBUG
from preprocess.process_utils import SQLValue, State
from preprocess.nl2sql.value_utils import AGG_OP, CMP_OP

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
                    "sel": [0], "agg": [0],
                    "cond_conn_op": 0, 'conds': [[1, 2, "1"]]
                }
                ast_node = self.parse_sql(error_sql, set())
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
        @return:
            ast_node: AbstractSyntaxTree of sql
        """
        ast_node = self.parse_sql(sql_json, sql_values)
        return ast_node
    
    def parse_sql(self, sql: dict, values: set):
        ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('SelectWhere'))
        select_field = ast_node[self.grammar.get_field_by_text('select select')][0]
        select_field.add_value(self.parse_select(sql['sel'], sql['agg']))
        where_field = ast_node[self.grammar.get_field_by_text('condition condition')][0]
        where_field.add_value(self.parse_where(sql['conds'], values, sql['cond_conn_op']))
        return ast_node
    
    def parse_select(self, sel: list, agg: list):
        ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('SelectColumn' + ASDLConstructor.number2word[len(sel)]))
        ast_fields = ast_node[self.grammar.get_field_by_text('col_unit col_unit')]
        for idx, (agg_id, col_id) in enumerate(zip(agg, sel)):
            col_unit_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('UnaryColumnUnit'))
            col_field = col_unit_node[self.grammar.get_field_by_text('col_id col_id')][0]
            col_field.add_value(int(col_id))
            agg_field = col_unit_node[self.grammar.get_field_by_text('agg_op agg_op')][0]
            agg_field.add_value(AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(AGG_OP[agg_id].title())))
            ast_fields[idx].add_value(col_unit_node)
        return ast_node
    
    def parse_where(self, conds: list, values: set, cond_conn: int):
        if len(conds) == 1:
            ast_node = self.parse_cond_unit(conds[0], values, 0)
        else:
            conj = 'Or' if cond_conn == 2 else 'And'
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(conj + 'Condition' + ASDLConstructor.number2word[len(conds)]))
            ast_fields = ast_node[self.grammar.get_field_by_text('condition condition')]
            for idx, cond_unit in enumerate(conds):
                cond_ast_node = self.parse_cond_unit(cond_unit, values, idx)
                ast_fields[idx].add_value(cond_ast_node)
        return ast_node
    
    def parse_cond_unit(self, cond: list, values: set, idx: int):
        ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name('CmpCondition'))
        ast_node[self.grammar.get_field_by_text('col_id col_id')][0].add_value(cond[0])
        CMP_OP_NAME = ['GreaterThan', 'LessThan', 'Equal', 'NotEqual']
        cmp_node = AbstractSyntaxTree(self.grammar.get_prod_by_ctr_name(CMP_OP_NAME[cond[1]]))
        ast_node[self.grammar.get_field_by_text('cmp_op cmp_op')][0].add_value(cmp_node)
        state = State(str(idx), 'none', CMP_OP[cond[1]], 'none', cond[0]) # namedtuple of (track, agg_op, cmp_op, unit_op, col_id)
        sql_value = SQLValue(str(cond[2]), state)
        try:
            for val in values:
                if val == sql_value:
                    ast_node[self.grammar.get_field_by_text('val_id val_id')][0].add_value(int(val.value_id))
                    break
        except:
            raise ValueError('Unable to find value %s in extracted values' % (cond[2]))
        return ast_node
