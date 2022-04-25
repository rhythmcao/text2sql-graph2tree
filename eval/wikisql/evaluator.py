#coding=utf8
import sys, tempfile, os, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.wikisql.evaluate import evaluate
from eval.lib.dbengine import DBEngine
from eval.lib.query import Query
from eval.evaluator import Evaluator


class Engine():

    def eval_hardness(self, sql: dict):
        # 'easy', 'medium', 'hard', 'extra'
        hard = len(sql['conds'])
        if hard == 0: return 'easy'
        elif hard == 1: return 'medium'
        elif hard == 2: return 'hard'
        else: return 'extra'

class WikiSQLEvaluator(Evaluator):

    def __init__(self, transition_system, table_path, db_dir):
        super(WikiSQLEvaluator, self).__init__(transition_system, table_path, db_dir)
        self.dataset = 'wikisql'
        self.engine = Engine()
        self.dbengines = { choice: DBEngine(os.path.join(db_dir, choice + '.db')) for choice in ['train', 'dev', 'test'] }
        self.surface_checker = SurfaceChecker(self.schemas)
        self.exec_checker = ExecutionChecker(self.dbengines)


    def _load_schemas(self, table_path):
        tables = {db['db_id']: db for db in json.load(open(table_path, 'r'))}
        return tables


    def evaluate_with_adaptive_interface(self, pred_sql, gold_sql, db, etype):
        """ @return: score(float): 0 or 1, etype score
        """
        try:
            pred_sql, gold_sql = json.loads(pred_sql), json.loads(gold_sql)
            if 'sql' in gold_sql: gold_sql = gold_sql['sql']
            pred, gold = Query.from_dict(pred_sql, False), Query.from_dict(gold_sql, False)
            if etype == 'match': return pred == gold
            dbengine = self.dbengines[db['data_split']]
            pred_ans = dbengine.execute_query(db['db_id'], pred, lower=True)
            gold_ans = dbengine.execute_query(db['db_id'], gold, lower=True)
            return pred_ans == gold_ans
        except: pass
        return 0.0


    def evaluate_with_official_interface(self, pred_sqls, ref_sqls, dbs, dataset, output_path, etype):
        with tempfile.NamedTemporaryFile('w+t', encoding='utf8', suffix='.sql') as tmp_pred, \
            tempfile.NamedTemporaryFile('w+t', encoding='utf8', suffix='.sql') as tmp_ref:
            of = open(output_path, 'w', encoding='utf8') if output_path is not None \
                else tempfile.TemporaryFile('w+t')
            # write pred and ref sqls
            for pred in pred_sqls:
                tmp_pred.write(pred + '\n')
            tmp_pred.flush()
            for s, db in zip(ref_sqls, dbs):
                ref = json.loads(s)
                # if gold sql does not contain db_id, reconstruct the dict
                if 'sql' not in ref: s = json.dumps({'sql': ref, 'table_id': db['db_id']}, ensure_ascii=False)
                tmp_ref.write(s + '\n')
            tmp_ref.flush()
            # calculate sql accuracy
            old_print = sys.stdout
            sys.stdout = of
            data_split = 'train' if dataset[0].id.startswith('train') else 'dev' if dataset[0].id.startswith('dev') else 'test'
            db_file = os.path.join(self.db_dir, data_split + '.db')
            result = evaluate(tmp_ref.name, db_file, tmp_pred.name, ordered=False)
            sys.stdout = old_print
            of.close()
        return result


class SurfaceChecker():

    def __init__(self, schemas) -> None:
        self.schemas = schemas

    def validity_check(self, sql: str, db: dict):
        """ Check whether the given sql query is valid, including:
        1. comparison operator or MAX/MIN/SUM/AVG only applied to columns of type real
        @params:
            sql(str): SQL query
            db(dict): database dict
        @return:
            flag(boolean)
        """
        try:
            sql, col_types = json.loads(sql), db["column_types"]
            if 'sql' in sql: sql = sql['sql']
            col_id, agg_id = sql['sel'], sql['agg']
            # MAX/MIN/SUM/AVG
            if agg_id in [1, 2, 4, 5] and col_types[col_id] != 'real': return False
            for col_id, cmp_id, _ in sql['conds']: # >/<
                if cmp_id in [1, 2] and col_types[col_id] != 'real': return False
            return True
        except Exception as e:
            print('Runtime error occurs in SurfaceChecker:', e)
            return False

class ExecutionChecker():

    def __init__(self, engines: dict) -> None:
        self.engines = engines

    def validity_check(self, sql: str, db: dict):
        """ Can not execute, directly return True """
        engine = self.engines[db['data_split']]
        try:
            sql = json.loads(sql)
            if 'sql' in sql: sql = sql['sql']
            pred = Query.from_dict(sql, ordered=False)
            pred = engine.execute_query(db['db_id'], pred, lower=True)
            return True
        except Exception as e: pass
        return False


if __name__ == '__main__':

    from tqdm import tqdm
    from utils.constants import DATASETS
    data_dir, db_dir = DATASETS['wikisql']['data'], DATASETS['wikisql']['database']
    tables = {db['db_id']: db for db in json.load(open(os.path.join(data_dir, 'tables.json'), 'r'))}
    surface_checker = SurfaceChecker(tables)
    dbengines = { choice: DBEngine(os.path.join(db_dir, choice + '.db')) for choice in ['train', 'dev', 'test'] }
    exec_checker = ExecutionChecker(dbengines)

    train = json.load(open(os.path.join(data_dir, 'train.json'), 'r'))
    dev = json.load(open(os.path.join(data_dir, 'dev.json'), 'r'))
    test = json.load(open(os.path.join(data_dir, 'test.json'), 'r'))
    count, total = 0, len(train + dev + test)
    for ex in tqdm(train + dev + test, total=total):
        sql, db_id = ex['query'].strip(), ex['db_id']
        flag1 = surface_checker.validity_check(sql, tables[db_id])
        flag2 = exec_checker.validity_check(sql, tables[db_id])
        if not (flag1 and flag2):
            count += 1
            print(sql)
    print('Total invalid is %d' % (count))