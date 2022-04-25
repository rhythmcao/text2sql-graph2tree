#coding=utf8
import json
from asdl.asdl import ASDLGrammar
from asdl.asdl_ast import AbstractSyntaxTree
from asdl.transition_system import SelectValueAction
from preprocess.wikisql.postprocess import ValueProcessor
from preprocess.wikisql.value_utils import CMP_OP
from preprocess.process_utils import AGG_OP, SQLValue, State
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
                return json.dumps({'sel': 0, 'agg': 0, 'conds': []}, ensure_ascii=False), False
    return wrapper

class UnParser():

    def __init__(self, grammar: ASDLGrammar, table_path: str, db_dir: str):
        """ ASDLGrammar """
        super(UnParser, self).__init__()
        self.grammar = grammar
        self.value_processor = ValueProcessor(table_path, db_dir)
    
    @ignore_error
    def unparse(self, sql_ast: AbstractSyntaxTree, db: dict, value_candidates: list, entry: dict, *args, **kargs):
        """ Return json dict, not SQL query string """
        self.sketch = kargs.pop('sketch', False)
        sql = self.unparse_sql(sql_ast, db, value_candidates, entry, *args, **kargs)
        return json.dumps(sql), True
    
    def unparse_sql(self, sql_ast: AbstractSyntaxTree, db: dict, value_candidates: list, entry: dict, *args, **kargs):
        sql = {}
        col_id = int(sql_ast[self.grammar.get_field_by_text('col_id col_id')][0].value)
        agg_ast = sql_ast[self.grammar.get_field_by_text('agg_op agg_op')][0].value
        agg_op = agg_ast.production.constructor.name.lower()
        agg_id = AGG_OP.index(agg_op)
        sql['agg'], sql['sel'] = agg_id, col_id
        where_field = sql_ast[self.grammar.get_field_by_text('condition condition')][0]
        sql['conds'] = self.unparse_where(where_field.value, db, value_candidates, entry, *args, **kargs)
        return sql
    
    # def retrieve_col_name(self, col_id, db):
    #     if self.sketch:
    #         return db['column_names_original'][0][1]
    #     return db['column_names_original'][col_id][1]
    
    def unparse_where(self, where_ast: AbstractSyntaxTree, db: dict, value_candidates: list, entry: dict, *args, **kargs):
        if len(where_ast.fields) == 0: return [] # no where condition
        cond_list = []
        cond_fields = where_ast[self.grammar.get_field_by_text('cond cond')]
        for rf in cond_fields:
            cond_list.append(self.unparse_cond_unit(rf.value, db, value_candidates, entry, *args, **kargs))
        return cond_list
    
    def unparse_cond_unit(self, cond_ast: AbstractSyntaxTree, db: dict, value_candidates: list, entry: dict, *args, **kargs):
        col_id = int(cond_ast[self.grammar.get_field_by_text('col_id col_id')][0].value)
        CMP_OP_MAPPING = {'Equal': 0, 'GreaterThan': 1, 'LessThan': 2}
        cmp_name = cond_ast[self.grammar.get_field_by_text('cmp_op cmp_op')][0].value.production.constructor.name
        cmp_id = CMP_OP_MAPPING[cmp_name]
        cmp_op = CMP_OP[cmp_id]
        if self.sketch:
            col_id, value = 0, "\"1\""
        else:
            val_id = int(cond_ast[self.grammar.get_field_by_text('val_id val_id')][0].value)
            candidate = val_id if val_id < SelectValueAction.size('wikisql') else value_candidates[val_id - SelectValueAction.size('wikisql')]
            state = State('', 'none', cmp_op, 'none', col_id)
            sqlvalue = SQLValue('', state)
            sqlvalue.add_candidate(candidate)
            value = self.value_processor.postprocess_value(sqlvalue, db, entry)
        return [col_id, cmp_id, value]