#coding=utf8
import sys, tempfile, os, json
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.cspider_raw.evaluation import evaluate, build_valid_col_units, rebuild_sql_val, rebuild_sql_col
from eval.cspider_raw.evaluation import Evaluator as Engine
from preprocess.cspider_raw.process_sql import get_schema, Schema, get_sql
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
            results = evaluate(tmp_ref.name, tmp_pred.name, self.db_dir, etype, self.kmaps)['all']
            sys.stdout = old_print
            of.close()
        return results


class ExecutionChecker():

    def __init__(self, db_dir='data/cspider_raw/database') -> None:
        super(ExecutionChecker, self).__init__()
        self.db_dir = db_dir

    def validity_check(self, sql: str, db: dict) -> bool:
        return True


class SurfaceChecker():

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
            sql = get_sql(SchemaID(db), sql)
            return self.sql_check(sql, db)
        except Exception as e:
            print('Runtime error occurs in SurfaceChecker:', e)
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
        if sql['from']['table_units'][0][0] == 'sql':
            if not self.sql_check(sql['from']['table_units'][0][1], db): return False
            table_ids = []
        else:
            table_ids = list(map(lambda table_unit: table_unit[1], sql['from']['table_units']))
        return self.select_check(sql['select'], table_ids, db) & \
            self.cond_check(sql['where'], table_ids, db) & \
            self.groupby_check(sql['groupBy'], table_ids, db) & \
            self.cond_check(sql['having'], table_ids, db) & \
            self.orderby_check(sql['orderBy'], table_ids, db)

    def select_check(self, select, table_ids: list, db: dict):
        select = select[1]
        for agg_id, val_unit in select:
            if not self.valunit_check(val_unit, table_ids, db): return False
            # MAX/MIN/SUM/AVG
            # if agg_id in [1, 2, 4, 5] and (self.valunit_type(val_unit, db) not in ['number', 'time']):
                # return False
        return True

    def cond_check(self, cond, table_ids: list, db: dict):
        if len(cond) == 0:
            return True
        for idx in range(0, len(cond), 2):
            cond_unit = cond[idx]
            _, cmp_op, val_unit, val1, val2 = cond_unit
            flag = self.valunit_check(val_unit, table_ids, db)
            # if cmp_op in [3, 4, 5, 6]: # >, <, >=, <=
                # flag &= (self.valunit_type(val_unit, db) in ['number', 'time'])
            if type(val1) == dict:
                flag &= self.sql_check(val1, db)
            if type(val2) == dict:
                flag &= self.sql_check(val2, db)
            if not flag: return False
        return True

    def groupby_check(self, groupby, table_ids: list, db: dict):
        if not groupby: return True
        for col_unit in groupby:
            if not self.colunit_check(col_unit, table_ids, db): return False
        return True

    def orderby_check(self, orderby, table_ids: list, db: dict):
        if not orderby: return True
        orderby = orderby[1]
        for val_unit in orderby:
            if not self.valunit_check(val_unit, table_ids, db): return False
        return True

    def colunit_check(self, col_unit: list, table_ids: list, db: dict):
        """ Check from the following aspects:
        1. column belongs to the tables in FROM clause
        2. column type is valid for AGG_OP
        """
        agg_id, col_id, _ = col_unit
        if col_id == 0: return True
        tab_id = db['column_names'][col_id][0]
        if tab_id not in table_ids: return False
        col_type = db['column_types'][col_id]
        if agg_id in [1, 2, 4, 5]: # MAX, MIN, SUM, AVG
            return (col_type in ['time', 'number'])
        return True

    def valunit_check(self, val_unit: list, table_ids: list, db: dict):
        unit_op, col_unit1, col_unit2 = val_unit
        if unit_op == 0:
            return self.colunit_check(col_unit1, table_ids, db)
        if not (self.colunit_check(col_unit1, table_ids, db) and self.colunit_check(col_unit2, table_ids, db)):
            return False
        # COUNT/SUM/AVG -> number
        agg_id1, col_id1, _ = col_unit1
        agg_id2, col_id2, _ = col_unit2
        t1 = 'number' if agg_id1 > 2 else db['column_types'][col_id1]
        t2 = 'number' if agg_id2 > 2 else db['column_types'][col_id2]
        if (t1 not in ['number', 'time']) or (t2 not in ['number', 'time']) or t1 != t2:
            return False
        return True

    def valunit_type(self, val_unit: list, db: dict):
        unit_op, col_unit1, col_unit2 = val_unit
        if unit_op == 0:
            agg_id, col_id, _ = col_unit1
            if agg_id > 2: return 'number'
            else: return ('number' if col_id == 0 else db['column_types'][col_id])
        else:
            return 'number'


class SchemaID():
    """
    Simple schema which maps table&column to a unique identifier
    """
    def __init__(self, db):
        self._schema = self._build_schema(db)
        self._idMap = self._map(db)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _build_schema(self, db):
        """build <table, list of columns> schema by input db
        Args:
            db (dict): NULL
        Returns: TODO
        Raises: NULL
        """
        tables = [x.lower() for x in db.get('table_names_original', db['table_names'])]
        dct_table2cols = defaultdict(list)
        for table_id, column in db.get('column_names_original', db['column_names']):
            if table_id < 0:
                continue
            dct_table2cols[tables[table_id]].append(column.lower())
        return dct_table2cols

    def _map(self, table):
        idMap = { '*': 0 }
        table_names = table['table_names_original']
        column_names = table['column_names_original']
        for idx, (tab_id, col) in enumerate(column_names):
            if idx == 0: continue
            key = table_names[tab_id].lower()
            val = col.lower()
            idMap[key + "." + val] = idx

        for idx, tab in enumerate(table_names):
            key = tab.lower()
            idMap[key] = idx
        return idMap


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
    for idx, ex in enumerate(train):
        sql, db = ex['query'].strip(), ex['db_id']
        flag = checker.validity_check(sql, tables[db])
        if not flag:
            print(idx, ': ' + sql + '\t' + db)
            count += 1
    print('Total invalid is %d' % (count))