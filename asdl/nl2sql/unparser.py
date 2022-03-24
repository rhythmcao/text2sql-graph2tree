#coding=utf8
import re
from asdl.asdl import ASDLGrammar
from asdl.asdl_ast import AbstractSyntaxTree
from preprocess.nl2sql.postprocess import ValueProcessor
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
                return 'SELECT *', False
    return wrapper

class UnParser():

    def __init__(self, grammar: ASDLGrammar, table_path: str, db_dir: str):
        """ ASDLGrammar """
        super(UnParser, self).__init__()
        self.grammar = grammar
        self.value_processor = ValueProcessor(table_path, db_dir)
    
    @ignore_error
    def unparse(self, sql_ast: AbstractSyntaxTree, *args, **kargs):
        self.sketch = kargs.pop('sketch', False)
        sql = self.unparse_sql(sql_ast, *args, **kargs)
        # sql = re.sub(r'\s+', ' ', sql)
        return sql, True
    
    def unparse_sql(self, sql_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        select_field = sql_ast[self.grammar.get_field_by_text('select select')][0]
        select_str = 'SELECT ' + self.unparse_select(select_field.value, db)
        where_field = sql_ast[self.grammar.get_field_by_text('condition condition')][0]
        where_str = ' WHERE' + self.unparse_where(where_field.value, db, *args, **kargs)
        return select_str + where_str
    
    def retrieve_col_name(self, col_id, db):
        if col_id == 0:
            return '*'
        return db['column_names_original'][col_id][1]

    def unparse_select(self, select_ast: AbstractSyntaxTree, db: dict):
        select_list = []
        select_fields = select_ast[self.grammar.get_field_by_text('col_unit col_unit')]
        for rf in select_fields:
            col_ast = rf.value
            col_id = int(col_ast[self.grammar.get_field_by_text('col_id col_id')][0].value)
            col_name = self.retrieve_col_name(col_id, db)
            agg_ast = col_ast[self.grammar.get_field_by_text('agg_op agg_op')][0].value
            agg_op = agg_ast.production.constructor.name.lower()
            if agg_op == 'none':
                select_list.append(col_name)
            else: select_list.append(agg_op.upper() + ' ( ' + col_name + ' ) ')
        return ' , '.join(select_list)
    
    def unparse_where(self, where_ast: AbstractSyntaxTree, db: dict, *args, **kargs):
        ctr_name = where_ast.production.constructor.name.lower()
        if ctr_name.startswith('cmp'):
            return self.unparse_cond_unit(where_ast, db, *args, **kargs)
        else:
            cond_list = []
            cond_fields = where_ast[self.grammar.get_field_by_text('condition condition')]
            for rf in cond_fields:
                cond_list.append(self.unparse_cond_unit(rf.value, db, *args, **kargs))
            conj = ' AND ' if ctr_name.startswith('and') else ' OR '
            return conj.join(cond_list)
    
    def unparse_col_unit(self, cond_ast: AbstractSyntaxTree, db: dict, value_candidates: list, entry: dict, *args, **kargs):
        col_id = int(cond_ast[self.grammar.get_field_by_text('col_id col_id')][0].value)
        col_name = self.retrieve_col_name(col_id, db)
        CMP_OP_MAPPING = {'GreaterThan': ' > ', 'LessThan': ' < ', 'Equal': ' = ', 'NotEqual': ' != '}
        cmp_name = cond_ast[self.grammar.get_field_by_text('cmp_op cmp_op')][0].value.production.constructor.name
        cmp_op = CMP_OP_MAPPING[cmp_name]
        if self.sketch:
            value_str = "1"
        else:
            val_id = int(cond_ast[self.grammar.get_field_by_text('val_id val_id')][0].value)
            state = State('', 'none', cmp_name.strip(), 'none', col_id)
            value_str = self.value_processor.postprocess_value(val_id, value_candidates, db, state, entry)
        return col_name + cmp_op + value_str