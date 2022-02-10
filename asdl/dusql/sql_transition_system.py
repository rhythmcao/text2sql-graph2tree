# coding=utf-8
import os, sys, json, pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from asdl.transition_system import TransitionSystem
from utils.constants import DATASETS


class SQLTransitionSystem(TransitionSystem):
    def __init__(self, grammar, table_path=None, db_dir=None):
        super(SQLTransitionSystem, self).__init__(grammar)
        if grammar.grammar_version == '1':
            from asdl.dusql.parser_v1 import Parser
            from asdl.dusql.unparser_v1 import UnParser
        else:
            from asdl.dusql.parser_v2 import Parser
            from asdl.dusql.unparser_v2 import UnParser
        if table_path is None:
            table_path = os.path.join(DATASETS['dusql']['data'], 'tables.bin')
        if db_dir is None:
            db_dir = DATASETS['dusql']['database']
        self.parser = Parser(self.grammar)
        self.unparser = UnParser(self.grammar, table_path, db_dir)


if __name__ == '__main__':

    from eval.evaluation import evaluate
    from asdl.asdl import ASDLGrammar
    grammar = ASDLGrammar.from_filepath(DATASETS['dusql']['grammar'])
    print('Total number of productions:', len(grammar))
    # for each in grammar.productions: print(each)
    print('Total number of types:', len(grammar.types))
    # for each in grammar.types: print(each)
    print('Total number of fields:', len(grammar.fields))
    # for each in grammar.fields: print(each)
    trans = SQLTransitionSystem(grammar)

    data_dir = DATASETS['dusql']['data']
    table_path = os.path.join(data_dir, 'tables.json')
    tables = json.load(open(table_path, 'r'))
    train = pickle.load(open(os.path.join(data_dir, 'train.bin'), 'rb'))
    dev = pickle.load(open(os.path.join(data_dir, 'dev.bin'), 'rb'))


    def sql_to_ast_to_sql(dataset):
        recovered_sqls = []
        for ex in dataset:
            sql_ast = trans.surface_code_to_ast(ex['sql'], ex['values'])
            sql_ast.sanity_check()
            recovered_sql, flag = trans.ast_to_surface_code(sql_ast, tables[ex['db_id']], ex['candidates'], ex)
            recovered_sqls.append(recovered_sql)
        return recovered_sqls
    

    def evaluate_sqls(recovered_sqls, choice='train'):
        pred_path = os.path.join(data_dir, choice + '_pred.sql')
        with open(pred_path, 'w') as of:
            dataset = train if choice == 'train' else dev
            for ex, each in zip(dataset, recovered_sqls):
                of.write(ex['question_id'] + '\t' + each + '\n')
        gold_path = os.path.join(data_dir, choice + '_gold.sql')
        output_path = os.path.join(data_dir, choice + '_eval.log')
        with open(output_path, 'w') as of:
            sys.stdout, old_print = of, sys.stdout
            evaluate(table_path, gold_path, pred_path, dataset='DuSQL')
            sys.stdout = old_print

    train_sqls = sql_to_ast_to_sql(train)
    evaluate_sqls(train_sqls, 'train')

    dev_sqls = sql_to_ast_to_sql(dev)
    evaluate_sqls(dev_sqls, 'dev')
