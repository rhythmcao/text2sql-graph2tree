#coding=utf8
import sys, tempfile, os, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.evaluation import evaluate, get_sql, Schema, build_valid_col_units, rebuild_sql_col
from eval.evaluation import Evaluator as Engine
from eval.evaluator import Evaluator


class DuSQLEvaluator(Evaluator):

    def __init__(self, *args, **kargs):
        super(DuSQLEvaluator, self).__init__(*args, **kargs)
        self.engine = Engine()
        self.surface_checker = SurfaceChecker(self.schemas)
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
        schema, kmap = self.schemas[db_id], self.kmaps[db_id]
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
        score = self.engine.eval_exact_match(pred_sql, gold_sql, value_match=(etype != 'match'))
        return score


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
            acc_with_value, acc_without_value = evaluate(self.table_path, tmp_ref.name, tmp_pred.name, dataset='DuSQL', verbose=True)
            sys.stdout = old_print
            of.close()
        # to be compatible, exec means acc_with_value
        return { "exact": acc_without_value, "exec": acc_with_value}


class SurfaceChecker():

    def __init__(self, schemas) -> None:
        self.schemas = schemas

    def validity_check(self, sql: str, db: dict):
        """ Check whether the given sql query is valid, including:
        1. only use columns in tables mentioned in FROM clause
        2. comparison operator or MAX/MIN/SUM/AVG only applied to columns of type number/time
        @params:
            sql(str): SQL query
            db(dict): database dict
        @return:
            flag(boolean)
        """
        try:
            sql = get_sql(self.schemas[db['db_id']], sql, single_equal=True)
            return self.sql_check(sql, db)
        except Exception as e:
            # print('Runtime error occurs in SurfaceChecker:', e)
            return False

    def sql_check(self, sql: dict, db: dict):
        if sql['intersect']:
            return self.sqlunit_check(sql, db) & self.sqlunit_check(sql['intersect'], db)
        if sql['union']:
            return self.sqlunit_check(sql, db) & self.sqlunit_check(sql['union'], db)
        if sql['except']:
            return self.sqlunit_check(sql, db) & self.sqlunit_check(sql['except'], db)
        return self.sqlunit_check(sql, db)

    def sqlunit_check(self, sql: dict, db: dict):
        """ table_names restrict that columns must be */TIME_NOW or belong to those in table_names
        """
        table_names = []
        if sql['from']['table_units'][0][0] == 'sql':
            for _, nested_sql in sql['from']['table_units']:
                if not self.sql_check(nested_sql, db): return False
                table_names += list(map(lambda table_unit: table_unit[1].strip('_'), nested_sql['from']['table_units']))
        else:
            table_names = list(map(lambda table_unit: table_unit[1].strip('_'), sql['from']['table_units']))
        return self.select_check(sql['select'], table_names, db) & \
            self.cond_check(sql['from']['conds'], table_names, db, False) & \
            self.cond_check(sql['where'], table_names, db, False) & \
            self.groupby_check(sql['groupBy'], table_names, db) & \
            self.cond_check(sql['having'], table_names, db, True) & \
            self.orderby_check(sql['orderBy'], table_names, db)

    def select_check(self, select: list, table_names: list, db: dict):
        if not select: return False
        for agg, val_unit in select:
            if not self.valunit_check(val_unit, table_names, db):
                return False
            # MAX/MIN/SUM/AVG
            if agg in [1, 2, 4, 5] and (self.valunit_type(val_unit, db) not in ['number', 'time']):
                return False
        return True

    def cond_check(self, cond, table_names: list, db: dict, must_agg: bool = False):
        if len(cond) == 0:
            return True
        for idx in range(0, len(cond), 2):
            cond_unit = cond[idx]
            agg, cmp_op, val_unit, val1, val2 = cond_unit
            if (must_agg and agg == 0) or (not must_agg and agg != 0): return False
            flag = self.valunit_check(val_unit, table_names, db)
            if agg in [1, 2, 4, 5] and (self.valunit_type(val_unit, db) not in ['number', 'time']):
                return False
            if cmp_op in [3, 4, 5, 6]: # >, <, >=, <=
                flag &= (agg > 2) or (self.valunit_type(val_unit, db) in ['number', 'time'])
            if type(val1) == dict:
                flag &= self.sql_check(val1, db)
            if type(val2) == dict:
                flag &= self.sql_check(val2, db)
            if not flag: return False
        return True

    def groupby_check(self, groupby, table_names: list, db: dict):
        if not groupby: return True
        for col_unit in groupby:
            if not self.colunit_check(col_unit, table_names, db): return False
        return True

    def orderby_check(self, orderby, table_names: list, db: dict):
        if not orderby: return True
        orderby = orderby[1]
        for agg, val_unit in orderby:
            if not self.valunit_check(val_unit, table_names, db): return False
            # MAX/MIN/SUM/AVG
            if agg in [1, 2, 4, 5] and (self.valunit_type(val_unit, db) not in ['number', 'time']):
                return False
        return True

    def extract_column_type(self, col_name, db):
        if col_name.lower() == '__all__': return 'text'
        elif col_name.lower() == 'time_now': return 'time'
        tab_name, col_name = col_name.strip('_').split('.')
        ref_names = [t.lower() for t in db.get('table_names_original', db['table_names'])]
        assert tab_name in ref_names
        tab_id = ref_names.index(tab_name)
        column_names = [c.lower() for t_id, c in db.get('column_names_original', db['column_names']) if t_id == tab_id]
        assert col_name in column_names
        col_id = column_names.index(col_name)
        offset = sum([1 for t_id, _ in db['column_names'] if t_id < tab_id])
        return db['column_types'][offset + col_id]

    def colunit_check(self, col_unit: list, table_names: list, db: dict):
        """ Check from the following aspects:
        1. column belongs to the tables in FROM clause, or in the column_names if not empty
        2. column type is valid for AGG_OP
        """
        agg_id, col_name = col_unit
        if col_name == '__all__': return True
        if col_name.lower() == 'time_now': return True
        tab_name = col_name.strip('_').split('.')[0]
        if tab_name not in table_names: return False
        return True

    def valunit_check(self, val_unit: list, table_names: list, db: dict):
        unit_op, col_unit1, col_unit2 = val_unit
        if unit_op == 0:
            return self.colunit_check(col_unit1, table_names, db)
        if not (self.colunit_check(col_unit1, table_names, db) and self.colunit_check(col_unit2, table_names, db)):
            return False
        # COUNT/SUM/AVG -> number
        agg_id1, col_name1 = col_unit1
        agg_id2, col_name2 = col_unit2
        t1 = 'number' if agg_id1 > 2 else self.extract_column_type(col_name1, db)
        t2 = 'number' if agg_id2 > 2 else self.extract_column_type(col_name2, db)
        if t1 in ['number', 'time'] and t2 in ['number', 'time']: return True
        if t1 == t2: return True
        return False

    def valunit_type(self, val_unit: list, db: dict):
        unit_op, col_unit1, _ = val_unit
        if unit_op == 0:
            agg_id, col_name = col_unit1
            if agg_id > 2: return 'number'
            else: return self.extract_column_type(col_name, db)
        else:
            return 'number'


class ExecutionChecker():
    def validity_check(self, sql: str, db: dict):
        """ Can not execute, directly return True """
        return True


if __name__ == '__main__':

    from utils.constants import DATASETS
    data_dir = DATASETS['dusql']['data']

    tables, schemas = {}, {}
    tables_list = json.load(open(os.path.join(data_dir, 'tables.json'), 'r'))
    for db in tables_list:
        tables[db['db_id']] = db
        schemas[db['db_id']] = Schema(db)
    checker = SurfaceChecker(schemas)

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