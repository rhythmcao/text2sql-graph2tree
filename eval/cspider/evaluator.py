#coding=utf8
import sys, tempfile, os, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.evaluation import evaluate, get_sql, Schema, build_valid_col_units, rebuild_sql_col
from eval.evaluation import Evaluator as Engine
from eval.evaluator import Evaluator
from eval.spider.evaluator import SurfaceChecker
from eval.dusql.evaluator import ExecutionChecker


class CSpiderEvaluator(Evaluator):

    def __init__(self, *args, **kargs):
        super(CSpiderEvaluator, self).__init__(*args, **kargs)
        self.engine = Engine()
        self.surface_checker = SurfaceChecker()
        self.exec_checker = ExecutionChecker()


    def _load_schemas(self, table_path):
        schemas = {}
        tables = json.load(open(table_path, 'r'))
        for db in tables:
            schemas[db['db_id']] = Schema(db)
        return schemas


    def evaluate_with_adaptive_interface(self, pred_sql, gold_sql, db_id, etype):
        """ @return: score(float): 0 or 1, etype score
        """
        schema, kmap = self.schema[db_id], self.kmaps[db_id]
        try:
            pred_sql = pred_sql.replace('==', '=')
            pred_sql = get_sql(schema, pred_sql, single_equal=True)
            p_valid_col_units = build_valid_col_units(pred_sql['from']['table_units'], schema)
            pred_sql = rebuild_sql_col(p_valid_col_units, pred_sql, kmap)
        except:
            # If p_sql is not valid, then we will use an empty sql to evaluate with the correct sql
            return 0.
        gold_sql = get_sql(schema, gold_sql, single_equal=True)
        g_valid_col_units = build_valid_col_units(gold_sql['from']['table_units'], schema)
        gold_sql = rebuild_sql_col(g_valid_col_units, gold_sql, kmap)
        score = self.engine.eval_exact_match(pred_sql, gold_sql, value_match=(etype == 'match'))
        return score


    def evaluate_with_official_interface(self, pred_sqls, ref_sqls, dbs, dataset, output_path, etype, checker=False):
        with tempfile.NamedTemporaryFile('w+t', encoding='utf8', suffix='.sql') as tmp_pred, \
            tempfile.NamedTemporaryFile('w+t', encoding='utf8', suffix='.sql') as tmp_ref:
            of = open(output_path, 'w', encoding='utf8') if output_path is not None \
                else tempfile.TemporaryFile('w+t')
            # write pred and ref sqls
            for idx, s in enumerate(pred_sqls):
                qid = str(dataset[idx].ex['question_id'])
                tmp_pred.write(qid + '\t' + s + '\n')
            tmp_pred.flush()
            for idx, (s, db) in enumerate(zip(ref_sqls, dbs)):
                qid = str(dataset[idx].ex['question_id'])
                tmp_ref.write(qid + '\t' + s + '\t' + db['db_id'] + '\n')
            tmp_ref.flush()
            # calculate sql accuracy
            old_print = sys.stdout
            sys.stdout = of
            acc_with_value, acc_without_value = evaluate(self.table_path, tmp_ref.name, tmp_pred.name, dataset='CSpider', verbose=True)
            sys.stdout = old_print
            of.close()
        # to be compatible, exec means acc_with_value
        return { "exact": acc_without_value, "exec": acc_with_value}


if __name__ == '__main__':

    from utils.constants import DATASETS
    data_dir = DATASETS['cspider']['data']
    tables = {db['db_id']: db for db in json.load(open(os.path.join(data_dir, 'tables.json'), 'r'))}
    checker = SurfaceChecker()

    count = 0
    train = json.load(open(os.path.join(data_dir, 'train.json'), 'r'))
    dev = json.load(open(os.path.join(data_dir, 'dev.json'), 'r'))
    for ex in train + dev:
        sql, db = ex['query'].strip(), ex['db_id']
        flag = checker.validity_check(sql, tables[db])
        if not flag:
            count += 1
            print(ex['question_id'], ': ' + sql + '\t' + db)
    print('Total invalid is %d' % (count))