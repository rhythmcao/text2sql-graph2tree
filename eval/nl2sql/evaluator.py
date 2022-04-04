#coding=utf8
import sys, tempfile, os, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.evaluation import evaluate
from eval.eval_utils import query2sql, compare_set, get_scores, Engine
from eval.evaluator import Evaluator


class NL2SQLEvaluator(Evaluator):

    def __init__(self, *args, **kargs):
        super(NL2SQLEvaluator, self).__init__(*args, **kargs)
        self.engine = Engine()
        self.surface_checker = SurfaceChecker(self.schemas)
        self.exec_checker = ExecutionChecker()


    def _load_schemas(self, table_path):
        tables = {db['db_id']: db for db in json.load(open(table_path, 'r'))}
        return tables


    def evaluate_with_adaptive_interface(self, pred_sql, gold_sql, db_id, etype):
        """ @return: score(float): 0 or 1, etype score
        """
        cols = [i[1] for i in self.schemas[db_id]["column_names"]]
        try:
            gold_sql = gold_sql.replace('==', '=')
            pred_sql = pred_sql.replace('==', '=')
            components_gold, sels_gold = query2sql(gold_sql, cols, single_equal=True, with_value=(etype != 'match'))
            components_pred, sels_pred = query2sql(pred_sql, cols, single_equal=True, with_value=(etype != 'match'))
            cnt, pred_total, gold_total = compare_set(sels_gold, sels_pred)
            score_sels, _, _ = get_scores(cnt, pred_total, gold_total)
            cnt, pred_total, gold_total = compare_set(components_gold["conds"], components_pred["conds"])
            score_conds, _, _ = get_scores(cnt, pred_total, gold_total)
            score_conn = components_gold["cond_conn_op"] == components_pred["cond_conn_op"]
            if score_sels and score_conds and score_conn: return 1.0
        except: pass
        return 0.0


    def evaluate_with_official_interface(self, pred_sqls, ref_sqls, dbs, dataset, output_path, etype):
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
            acc_with_value, acc_without_value = evaluate(self.table_path, tmp_ref.name, tmp_pred.name, dataset='NL2SQL', verbose=True)
            sys.stdout = old_print
            of.close()
        # to be compatible, exec means acc_with_value
        return { "exact": acc_without_value, "exec": acc_with_value}


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
            cols = [i[1] for i in db["column_names"]]
            col_types = db['column_types']
            components, sels = query2sql(sql, cols, single_equal=True, with_value=False)
            for agg_id, col_id in sels:
                col_type = col_types[col_id]
                if agg_id in [1, 2, 3, 5] and col_type != 'real': return False
            conds = [(cond[1], cond[0]) for cond in components['conds']]
            for cmp_id, col_id in conds:
                col_type = col_types[col_id]
                if cmp_id in [0, 1] and col_type != 'real': return False
            return True
        except Exception as e:
            # print('Runtime error occurs in SurfaceChecker:', e)
            return False

class ExecutionChecker():
    def validity_check(self, sql: str, db: dict):
        """ Can not execute, directly return True """
        return True


if __name__ == '__main__':

    from utils.constants import DATASETS
    data_dir = DATASETS['nl2sql']['data']
    tables = {db['db_id']: db for db in json.load(open(os.path.join(data_dir, 'tables.json'), 'r'))}
    checker = SurfaceChecker(tables)

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