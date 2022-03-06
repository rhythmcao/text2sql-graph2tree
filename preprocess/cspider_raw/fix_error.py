#coding=utf8
import os, json, re, sys, sqlite3
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import shutil, collections, itertools
from utils.constants import DATASETS
from preprocess.cspider_raw.process_sql import get_sql
from preprocess.cspider_raw.parse_raw_json import get_schemas_from_json, Schema


def amend_primary_keys(tables: list, verbose: bool = True):
    # if the following situation occurs, we add primary key c2:
    # 1. (c1, c2) is foreign key
    # 2. c2 is not primary key
    # 3. table~(which has c2) does not have primary key
    count = 0
    for table in tables:
        if len(table['column_names']) > 100:
            continue
        pks = table['primary_keys']
        pk_tables = set([table['column_names'][c][0] for c in pks])
        fks = table['foreign_keys']
        candidates = set([pair[1] for pair in fks if pair[1] not in pks and table['column_names'][pair[1]][0] not in pk_tables])
        for c in candidates:
            pks.append(c)
            if verbose:
                print('DB[{}]: add primary key: {}'.format(
                    table['db_id'], table['column_names_original'][c]))
        count += len(candidates)
        table['primary_keys'] = sorted(list(pks))
    if verbose:
        print('{} primary keys added'.format(count))
    return tables

def amend_foreign_keys(tables: list, verbose: bool = True):
    # if the following situation occurs, we add foreign key (c1, c2)
    # 1. c1 and c2 have the same name but belong to different tables
    # 2. c1 is not primary key and c2 is primary key
    # 3. c1 belongs to table t1, no column in t1 is/has foreign key of/with c2
    num_foreign_keys_reversed, num_foreign_keys_added = 0, 0
    for table in tables:
        if len(table['column_names']) > 100:
            continue
        c_dict = collections.defaultdict(list)
        for c_id, c in enumerate(table['column_names_original']):
            t_id, c_name = c[0], c[1].lower()
            c_dict[c_name].append((t_id, c_id))
        primary_keys = table['primary_keys']
        foreign_keys = set([tuple(x) for x in table['foreign_keys']])
        for c_name in c_dict:
            if c_name in ['*', 'name', 'id', 'code']:
                continue
            if len(c_dict[c_name]) > 1:
                for (p_t, p), (q_t, q) in itertools.combinations(c_dict[c_name], 2):
                    if p_t == q_t: continue
                    if p in primary_keys and q not in primary_keys:
                        if ((p, q) in foreign_keys) and ((q, p) not in foreign_keys):
                            if verbose:
                                print('DB[{}]: reversed foreign key: {}->{}'.format(
                                    table['db_id'],
                                    table['column_names_original'][q],
                                    table['column_names_original'][p]))
                            foreign_keys.remove((p, q))
                            foreign_keys.add((q, p))
                            num_foreign_keys_reversed += 1
                        elif (q, p) not in foreign_keys:
                            connect_tables = set(map(lambda k: table['column_names'][k[1]][0] if k[0] == p else table['column_names'][k[0]][0], filter(lambda k: k[0] == p or k[1] == p, foreign_keys)))
                            if connect_tables and q_t not in connect_tables:
                                if verbose:
                                    print('DB[{}]: add foreign key: {}->{}'.format(table['db_id'],
                                        table['column_names_original'][q],
                                        table['column_names_original'][p]))
                                foreign_keys.add((q, p))
                                num_foreign_keys_added += 1
                    elif q in primary_keys and p not in primary_keys:
                        if (q, p) in foreign_keys and (p, q) not in foreign_keys:
                            if verbose:
                                print('DB[{}]: reversed foreign key: {}->{}'.format(
                                    table['column_names_original'][p],
                                    table['column_names_original'][q]))
                            foreign_keys.remove((q, p))
                            foreign_keys.add((p, q))
                            num_foreign_keys_reversed += 1
                        elif (p, q) not in foreign_keys:
                            connect_tables = set(map(lambda k: table['column_names'][k[1]][0] if k[0] == q else table['column_names'][k[0]][0], filter(lambda k: k[0] == q or k[1] == q, foreign_keys)))
                            if connect_tables and p_t not in connect_tables:
                                if verbose:
                                    print('DB[{}]: add foreign key: {}->{}'.format(table['db_id'],
                                        table['column_names_original'][p],
                                        table['column_names_original'][q]))
                                foreign_keys.add((p, q))
                                num_foreign_keys_added += 1
        foreign_keys = sorted(list(foreign_keys), key=lambda x: x[0])
        table['foreign_keys'] = foreign_keys
    if verbose:
        print('{} foreign key pairs reversed'.format(num_foreign_keys_reversed))
        print('{} foreign key pairs added'.format(num_foreign_keys_added))
    return tables

def obtain_column_type_and_values(db_dir, db_id, tab_name, col_name):
    db_file = os.path.join(db_dir, db_id, db_id + '.sqlite')
    conn = sqlite3.connect(db_file)
    conn.text_factory = lambda b: b.decode(errors='ignore')
    cur = conn.execute("PRAGMA table_info('{}') ".format(tab_name))
    columns = list(cur.fetchall())
    cur = conn.execute('SELECT DISTINCT \"%s\" FROM \"%s\" ;' % (col_name, tab_name))
    vlist = list(cur.fetchall())
    vset = set([str(v[0]).lower() for v in vlist if str(v[0]).strip() != ''])
    conn.close()
    c_type = 'text'
    for col in columns:
        if col[1].lower() == col_name.lower():
            c_type = col[2].lower()
            break
    return c_type, vset

def is_new_bool_candidate(db_dir, db_id, t_name, c_name, old_type):
    """ Determine whether a given column should be changed into type boolean
    1. not type number/time, nor boolean already
    2. column name starts with 'if' or 'is', or ends with 'yn' or 'tf'
    3. column cells/values is contained in the following set: {y, n}, {yes, no}, {t, f}, {true, false}
    4. column cells/values belongs to set {0, 1}, but column type should be in varchar(1), char(1), bit
    """
    if old_type in ['boolean', 'number', 'time'] or 'gender' in c_name.lower() or 'sex' in c_name.lower(): return False
    if re.search(r'^([Ii][Ff])[_ A-Z]', c_name) is not None or re.search(r'^([Ii][Ss])[_ A-Z]', c_name) is not None \
        or re.search(r'[_ ]([yY][nN])$', c_name) is not None or re.search(r'[_ ]([tT][fF])$', c_name) is not None:
        return True
    new_type, vset = obtain_column_type_and_values(db_dir, db_id, t_name, c_name)
    evidence_set = [{'y', 'n'}, {'yes', 'no'}, {'t', 'f'}, {'true', 'false'}]
    if new_type == 'bool': return True
    elif len(vset) > 0 and any(len(vset - s) == 0 for s in evidence_set): return True
    elif len(vset) > 0 and len(vset - {'0', '1'}) == 0 and all(v not in c_name.lower() for v in ['value', 'code', 'id']): return True
    return False

def amend_boolean_types(tables: list, db_dir: str = 'data/cspider_raw/database', verbose: bool = True):
    count = 0
    for db in tables:
        db_id, table_list = db['db_id'], db['table_names_original']
        column_list, column_types = db['column_names_original'], db['column_types']
        for j, (t_id, c_name) in enumerate(column_list):
            if c_name == '*': continue
            t_name, old_type = table_list[t_id], column_types[j]
            if is_new_bool_candidate(db_dir, db_id, t_name, c_name, old_type):
                if verbose:
                    print('DB[{}]: revise column[{}.{}] type: {}->boolean'.format(db_id, t_name, c_name, old_type))
                db['column_types'][j] = 'boolean'
                count += 1
    if verbose:
        print('{} column types are changed to boolean'.format(count))
    return tables

question_query_mappings = {
    "最近签订合同的公司的类型描述是什么？": "SELECT T1.company_type FROM Third_Party_Companies AS T1 JOIN Maintenance_Contracts AS T2 ON T1.company_id  =  T2.maintenance_contract_company_id ORDER BY T2.contract_end_date DESC LIMIT 1",
    "哪位工程师去过的次数最多？显示工程师id、名字和姓氏。": "SELECT T1.engineer_id ,  T1.first_name ,  T1.last_name FROM Maintenance_Engineers AS T1 JOIN Engineer_Visits AS T2 ON T1.engineer_id = T2.engineer_id GROUP BY T1.engineer_id ORDER BY count(*) DESC LIMIT 1",
    "哪些产品被最少的客户投诉？": "SELECT DISTINCT t1.product_name FROM products AS t1 JOIN complaints AS t2 ON t1.product_id  =  t2.product_id JOIN customers AS t3 ON t2.customer_id = t3.customer_id GROUP BY t3.customer_id ORDER BY count(*) LIMIT 1",
    "返回提交最少客户投诉的产品名称。": "SELECT DISTINCT t1.product_name FROM products AS t1 JOIN complaints AS t2 ON t1.product_id  =  t2.product_id JOIN customers AS t3 ON t2.customer_id = t3.customer_id GROUP BY t3.customer_id ORDER BY count(*) LIMIT 1",
    "治疗费用低于平均的专家的名字和姓氏是什么？": "SELECT DISTINCT T1.first_name ,  T1.last_name FROM Professionals AS T1 JOIN Treatments AS T2 ON T1.professional_id = T1.professional_id WHERE cost_of_treatment  <  ( SELECT avg(cost_of_treatment) FROM Treatments )",
    "哪些专家的治疗费用低于平均水平？给出名字和姓氏。": "SELECT DISTINCT T1.first_name ,  T1.last_name FROM Professionals AS T1 JOIN Treatments AS T2 ON T1.professional_id = T1.professional_id WHERE cost_of_treatment  <  ( SELECT avg(cost_of_treatment) FROM Treatments )",
    "参与任何一个课程次数最多的学生的姓名、中间名、姓氏、id和参与次数是多少？": "SELECT T1.first_name ,  T1.middle_name ,  T1.last_name ,  T1.student_id ,  count(*) FROM Students AS T1 JOIN Student_Enrolment AS T2 ON T1.student_id  =  T2.student_id GROUP BY T1.student_id ORDER BY count(*) DESC LIMIT 1",
    "找出在1960年和1961年都获奖的球员的名字和姓氏。": "SELECT T1.name_first , T1.name_last FROM player AS T1 JOIN player_award AS T2 ON T1.player_id = T2.player_id WHERE T2.year  =  1960 INTERSECT SELECT T1.name_first , T1.name_last FROM player AS T1 JOIN player_award AS T2 ON T1.player_id = T2.player_id WHERE T2.year  =  1961",
    "哪位选手在1960和1961都获奖？返回他们的名字和姓氏。": "SELECT T1.name_first , T1.name_last FROM player AS T1 JOIN player_award AS T2 ON T1.player_id = T2.player_id WHERE T2.year  =  1960 INTERSECT SELECT T1.name_first , T1.name_last FROM player AS T1 JOIN player_award AS T2 ON T1.player_id = T2.player_id WHERE T2.year  =  1961"
}

query_mappings = {
    "SELECT T1.fname FROM student AS T1 JOIN lives_in AS T2 ON T1.stuid  =  T2.stuid WHERE T2.dormid IN (SELECT T2.dormid FROM dorm AS T3 JOIN has_amenity AS T4 ON T3.dormid  =  T4.dormid JOIN dorm_amenity AS T5 ON T4.amenid  =  T5.amenid GROUP BY T3.dormid ORDER BY count(*) DESC LIMIT 1)":
    "SELECT T1.fname FROM student AS T1 JOIN lives_in AS T2 ON T1.stuid  =  T2.stuid WHERE T2.dormid IN (SELECT T3.dormid FROM dorm AS T3 JOIN has_amenity AS T4 ON T3.dormid  =  T4.dormid JOIN dorm_amenity AS T5 ON T4.amenid  =  T5.amenid GROUP BY T3.dormid ORDER BY count(*) DESC LIMIT 1)",
}

question_query_replacement = {
    "哪些学生报名参加任何项目的次数最多？列出id、名字、中间名、姓氏、参加次数和学生id。":
    (
        "哪些学生报名参加任何项目的次数最多？列出学生id、名字、中间名、姓氏和参加次数。",
        "SELECT T1.student_id ,  T1.first_name ,  T1.middle_name ,  T1.last_name ,  count(*) FROM Students AS T1 JOIN Student_Enrolment AS T2 ON T1.student_id  =  T2.student_id GROUP BY T1.student_id ORDER BY count(*) DESC LIMIT 1"
    )
}

def amend_examples_in_dataset(dataset, schemas, tables, verbose=True):
    count = 0
    for ex in dataset:
        if ex['question'] in question_query_mappings:
            if verbose:
                print('DB:', ex['db_id'])
                print('Question:', ex['question'])
                print('SQL:', ex['query'])
                print('SQL revised:', question_query_mappings[ex['question']])
            ex['query'] = question_query_mappings[ex['question']]
            count += 1
        elif ex['question'] in question_query_replacement:
            if verbose:
                print('DB:', ex['db_id'])
                print('Question:', ex['question'])
                print('Question revised:', question_query_replacement[ex['question']][0])
                print('SQL:', ex['query'])
                print('SQL revised:', question_query_replacement[ex['question']][1])
            ex['query'] = question_query_replacement[ex['question']][1]
            ex['question'] = question_query_replacement[ex['question']][0]
            count += 1
        elif ex['query'] in query_mappings:
            if verbose:
                print('DB:', ex['db_id'])
                print('Question:', ex['question'])
                print('SQL:', ex['query'])
                print('SQL revised:', query_mappings[ex['query']])
            ex['query'] = query_mappings[ex['query']]
        db_id = ex['db_id']
        ex['sql'] = get_sql(Schema(schemas[db_id], tables[db_id]), ex['query'])
    print('Fix %d examples in the dataset' % (count))
    return dataset


def fix_two_special_tables(tables):
    for db in tables:
        if db['db_id'] == 'scholar':
            db['table_names'] = ['venue', 'author', 'dataset', 'journal', 'key phrase', 'paper', 'cite', 'paper dataset', 'paper key phrase', 'writes']
            db['column_names'] = [[-1, '*'], [0, 'venue id'], [0, 'venue name'], [1, 'author id'], [1, 'author name'], [2, 'dataset id'], [2, 'dataset name'], [3, 'journal id'], [3, 'journal name'], [4, 'key phrase id'], [4, 'key phrase name'], [5, 'paper id'], [5, 'title'], [5, 'venue id'], [5, 'year'], [5, 'number citing'], [5, 'number cited by'], [5, 'journal id'], [6, 'citing paper id'], [6, 'cited paper id'], [7, 'paper id'], [7, 'dataset id'], [8, 'paper id'], [8, 'key phrase id'], [9, 'paper id'], [9, 'author id']]
        elif db['db_id'] == 'formula_1':
            db['table_names'] = ['circuits', 'races', 'drivers', 'status', 'seasons', 'constructors', 'constructor standings', 'results', 'driver standings', 'constructor results', 'qualifying', 'pit stops', 'lap times']
            db['column_names'] = [[-1, '*'], [0, 'circuit id'], [0, 'circuit reference'], [0, 'name'], [0, 'location'], [0, 'country'], [0, 'latitude'], [0, 'longitude'], [0, 'altitude'], [0, 'url'], [1, 'race id'], [1, 'year'], [1, 'round'], [1, 'circuit id'], [1, 'name'], [1, 'date'], [1, 'time'], [1, 'url'], [2, 'driver id'], [2, 'driver reference'], [2, 'number'], [2, 'code'], [2, 'forename'], [2, 'surname'], [2, 'dob'], [2, 'nationality'], [2, 'url'], [3, 'status id'], [3, 'status'], [4, 'year'], [4, 'url'], [5, 'constructor id'], [5, 'constructor reference'], [5, 'name'], [5, 'nationality'], [5, 'url'], [6, 'constructor standings id'], [6, 'race id'], [6, 'constructor id'], [6, 'points'], [6, 'position'], [6, 'position text'], [6, 'wins'], [7, 'result id'], [7, 'race id'], [7, 'driver id'], [7, 'constructor id'], [7, 'number'], [7, 'grid'], [7, 'position'], [7, 'position text'], [7, 'position order'], [7, 'points'], [7, 'laps'], [7, 'time'], [7, 'milliseconds'], [7, 'fastest lap'], [7, 'rank'], [7, 'fastest lap time'], [7, 'fastest lap speed'], [7, 'status id'], [8, 'driver standings id'], [8, 'race id'], [8, 'driver id'], [8, 'points'], [8, 'position'], [8, 'position text'], [8, 'wins'], [9, 'constructor results id'], [9, 'race id'], [9, 'constructor id'], [9, 'points'], [9, 'status'], [10, 'qualify id'], [10, 'race id'], [10, 'driver id'], [10, 'constructor id'], [10, 'number'], [10, 'position'], [10, 'q1'], [10, 'q2'], [10, 'q3'], [11, 'race id'], [11, 'driver id'], [11, 'stop'], [11, 'lap'], [11, 'time'], [11, 'duration'], [11, 'milliseconds'], [12, 'race id'], [12, 'driver id'], [12, 'lap'], [12, 'position'], [12, 'time'], [12, 'milliseconds']]
    return tables


if __name__ == '__main__':

    data_dir, db_dir = DATASETS['cspider_raw']['data'], DATASETS['cspider_raw']['database']
    table_path = os.path.join(data_dir, 'tables.json')
    origin_table_path = os.path.join(data_dir, 'tables.original.json')
    update_table_path = origin_table_path if os.path.exists(origin_table_path) else table_path
    tables = fix_two_special_tables(json.load(open(update_table_path, 'r')))
    tables = amend_primary_keys(tables, verbose=True)
    tables = amend_foreign_keys(tables, verbose=True)
    tables = amend_boolean_types(tables, db_dir, verbose=True)
    if not os.path.exists(origin_table_path):
        shutil.copyfile(table_path, origin_table_path)
    json.dump(tables, open(table_path, 'w'), indent=4)

    schemas, _, tables = get_schemas_from_json(table_path)
    for data_split in ['train', 'dev']:
        dataset_path = os.path.join(data_dir, data_split + '.json')
        origin_dataset_path = os.path.join(data_dir, data_split + '.original.json')
        if os.path.exists(origin_dataset_path):
            dataset = json.load(open(origin_dataset_path, 'r'))
        else:
            dataset = json.load(open(dataset_path, 'r'))
            shutil.copyfile(dataset_path, origin_dataset_path)
        dataset = amend_examples_in_dataset(dataset, schemas, tables, verbose=True)
        json.dump(dataset, open(dataset_path, 'w'), indent=4, ensure_ascii=False)