#coding=utf8
import sys, tempfile, os, json
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.cspider_raw.evaluation import evaluate, build_valid_col_units, rebuild_sql_val, rebuild_sql_col
from eval.cspider_raw.evaluation import Evaluator as Engine
from preprocess.cspider_raw.process_sql import get_schema, Schema, get_sql
from eval.spider.evaluator import SurfaceChecker
from eval.dusql.evaluator import ExecutionChecker
from eval.evaluator import Evaluator


class CSpiderRawEvaluator(Evaluator):

    def __init__(self, *args, **kargs):
        super(CSpiderRawEvaluator, self).__init__(*args, **kargs)
        self.engine = Engine()
        self.surface_checker = SurfaceChecker()
        self.exec_checker = ExecutionChecker()


    def _load_schemas(self, table_path):
        schemas = {}
        tables = json.load(open(table_path, 'r'))
        for db in tables:
            db = db['db_id']
            db_path = os.path.join(self.db_dir, db, db + ".sqlite")
            schemas[db] = Schema(get_schema(db_path))
        return schemas


    def evaluate_with_adaptive_interface(self, pred_sql, gold_sql, db_id, etype):
        """ @return: score(float): 0 or 1, only exact set match w/o values available for cspider
        """
        schema, kmap = self.schemas[db_id], self.kmaps[db_id]
        try:
            p_sql = get_sql(schema, pred_sql)
        except:
            return 0.
        g_sql = get_sql(schema, gold_sql)
        g_valid_col_units = build_valid_col_units(g_sql['from']['table_units'], schema)
        g_sql = rebuild_sql_val(g_sql)
        g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap) # kmap: map __tab.col__ to pivot __tab.col__
        p_valid_col_units = build_valid_col_units(p_sql['from']['table_units'], schema)
        p_sql = rebuild_sql_val(p_sql)
        p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)
        score = float(self.engine.eval_exact_match(p_sql, g_sql))
        return score


    def evaluate_with_official_interface(self, pred_sqls, ref_sqls, dbs, dataset, output_path, etype):
        with tempfile.NamedTemporaryFile('w+t', encoding='utf8', suffix='.sql') as tmp_pred, \
            tempfile.NamedTemporaryFile('w+t', encoding='utf8', suffix='.sql') as tmp_ref:
            of = open(output_path, 'w', encoding='utf8') if output_path is not None \
                else tempfile.TemporaryFile('w+t')
            # write pred and ref sqls
            for s in pred_sqls:
                tmp_pred.write(s + '\n')
            tmp_pred.flush()
            for s, db in zip(ref_sqls, dbs):
                tmp_ref.write(s + '\t' + db['db_id'] + '\n')
            tmp_ref.flush()
            # calculate sql accuracy
            old_print = sys.stdout
            sys.stdout = of
            results = evaluate(tmp_ref.name, tmp_pred.name, self.db_dir, 'match', self.kmaps)['all']
            sys.stdout = old_print
            of.close()
        results['exec'] = results['exact'] # can not execute
        return results


if __name__ == '__main__':

    from utils.constants import DATASETS
    data_dir = DATASETS['cspider_raw']['data']
    tables = {}
    tables_list = json.load(open(os.path.join(data_dir, 'tables.json'), 'r'))
    for db in tables_list:
        tables[db['db_id']] = db

    count = 0
    checker = SurfaceChecker()
    train = json.load(open(os.path.join(data_dir, 'train.json'), 'r'))
    dev = json.load(open(os.path.join(data_dir, 'dev.json'), 'r'))
    for idx, ex in enumerate(train + dev):
        sql, db = ex['query'].strip(), ex['db_id']
        flag = checker.validity_check(sql, tables[db])
        if not flag:
            print(idx, ': ' + sql + '\t' + db)
            count += 1
    print('Total invalid is %d' % (count))