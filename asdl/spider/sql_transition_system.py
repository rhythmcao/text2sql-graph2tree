# coding=utf-8
import os, sys, pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from asdl.transition_system import TransitionSystem
from utils.constants import DATASETS


class SQLTransitionSystem(TransitionSystem):
    def __init__(self, grammar, table_path=None, db_dir=None):
        super(SQLTransitionSystem, self).__init__(grammar)
        from asdl.spider.parser import Parser
        from asdl.spider.unparser import UnParser
        self.parser = Parser(self.grammar)
        if table_path is None:
            table_path = os.path.join(DATASETS['spider']['data'], 'tables.bin')
        if db_dir is None:
            db_dir = DATASETS['spider']['database']
        self.unparser = UnParser(self.grammar, table_path, db_dir)


if __name__ == '__main__':

    from asdl.asdl import ASDLGrammar
    from eval.spider.evaluation import evaluate, build_foreign_key_map_from_json
    grammar = ASDLGrammar.from_filepath(DATASETS['spider']['grammar'])
    print('Total number of productions:', len(grammar))
    # for each in grammar.productions: print(each)
    print('Total number of types:', len(grammar.types))
    # for each in grammar.types: print(each)
    print('Total number of fields:', len(grammar.fields))
    # for each in grammar.fields: print(each)
    trans = SQLTransitionSystem(grammar)

    data_dir = DATASETS['spider']['data']
    kmaps = build_foreign_key_map_from_json(os.path.join(data_dir, 'tables.json'))
    tables = pickle.load(open(os.path.join(data_dir, 'tables.bin'), 'rb'))
    train = pickle.load(open(os.path.join(data_dir, 'train.lgesql.bin'), 'rb'))
    dev = pickle.load(open(os.path.join(data_dir, 'dev.lgesql.bin'), 'rb'))

    def create_gold_sql(choice):
        gold_path = os.path.join(data_dir, choice + '_gold.sql')
        dataset = train if choice == 'train' else dev
        with open(gold_path, 'w') as of:
            for ex in dataset:
                of.write(' '.join(ex['query'].split('\t')) + '\t' + ex['db_id'] + '\n')
        return


    def sql_to_ast_to_sql(dataset):
        recovered_sqls = []
        for ex in dataset:
            sql_ast = trans.surface_code_to_ast(ex['sql'], ex['values'])
            sql_ast.sanity_check()
            recovered_sql, flag = trans.ast_to_surface_code(sql_ast, tables[ex['db_id']], ex['candidates'], ex)
            recovered_sqls.append(recovered_sql)
        return recovered_sqls


    def evaluate_sqls(recovered_sqls, choice='train', etype='all'):
        pred_path = os.path.join(data_dir, choice + '_pred.sql')
        with open(pred_path, 'w') as of:
            for each in recovered_sqls:
                of.write(each + '\n')
        gold_path = os.path.join(data_dir, choice + '_gold.sql')
        output_path = os.path.join(data_dir, choice + '_eval.log')
        with open(output_path, 'w') as of:
            sys.stdout, old_print = of, sys.stdout
            evaluate(gold_path, pred_path, DATASETS['spider']['database_testsuite'], etype, kmaps, False, False, False)
            sys.stdout = old_print

    create_gold_sql('train')
    train_sqls = sql_to_ast_to_sql(train)
    evaluate_sqls(train_sqls, 'train', 'all')

    create_gold_sql('dev')
    dev_sqls = sql_to_ast_to_sql(dev)
    evaluate_sqls(dev_sqls, 'dev', 'all')
