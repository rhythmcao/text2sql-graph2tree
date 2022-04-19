#coding=utf8
import os, json, re, sys, sqlite3
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import shutil, collections, itertools
from utils.constants import DATASETS
from preprocess.spider.process_sql import get_sql
from preprocess.spider.parse_sql_one import get_schemas_from_json, Schema

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
                print('DB[{}]: add primary key: {}'.format(table['db_id'], table['column_names_original'][c]))
        count += len(candidates)
        table['primary_keys'] = sorted(list(pks))
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

def amend_boolean_types(tables: list, db_dir: str = 'data/spider/database', verbose: bool = True):
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
    print('{} column types are changed to boolean'.format(count))
    return tables

# incorrect SQL:
# 1. missing JOIN ON conditions for multi-tables (4)
# 2. replace HAVING clause with WHERE clause (1)
# 3. incorrect SQL due to table alias, nested SQL uses outer table alias (1)
query_mappings = {
    "SELECT T1.engineer_id ,  T1.first_name ,  T1.last_name FROM Maintenance_Engineers AS T1 JOIN Engineer_Visits AS T2 GROUP BY T1.engineer_id ORDER BY count(*) DESC LIMIT 1":
    "SELECT T1.engineer_id ,  T1.first_name ,  T1.last_name FROM Maintenance_Engineers AS T1 JOIN Engineer_Visits AS T2 ON T1.engineer_id = T2.engineer_id GROUP BY T1.engineer_id ORDER BY count(*) DESC LIMIT 1",
    "SELECT T1.name_first , T1.name_last FROM player AS T1 JOIN player_award AS T2 WHERE T2.year  =  1960 INTERSECT SELECT T1.name_first , T1.name_last FROM player AS T1 JOIN player_award AS T2 WHERE T2.year  =  1961":
    "SELECT T1.name_first , T1.name_last FROM player AS T1 JOIN player_award AS T2 ON T1.player_id = T2.player_id WHERE T2.year  =  1960 INTERSECT SELECT T1.name_first , T1.name_last FROM player AS T1 JOIN player_award AS T2 ON T1.player_id = T2.player_id WHERE T2.year  =  1961",
    "SELECT DISTINCT t1.product_name FROM products AS t1 JOIN complaints AS t2 ON t1.product_id  =  t2.product_id JOIN customers AS t3 GROUP BY t3.customer_id ORDER BY count(*) LIMIT 1":
    "SELECT DISTINCT t1.product_name FROM products AS t1 JOIN complaints AS t2 ON t1.product_id  =  t2.product_id JOIN customers AS t3 ON t2.customer_id = t3.customer_id GROUP BY t3.customer_id ORDER BY count(*) LIMIT 1",
    "SELECT DISTINCT T1.first_name ,  T1.last_name FROM Professionals AS T1 JOIN Treatments AS T2 WHERE cost_of_treatment  <  ( SELECT avg(cost_of_treatment) FROM Treatments )":
    "SELECT DISTINCT T1.first_name ,  T1.last_name FROM Professionals AS T1 JOIN Treatments AS T2 ON T1.professional_id = T2.professional_id WHERE cost_of_treatment  <  ( SELECT avg(cost_of_treatment) FROM Treatments )",
    "SELECT count(*) FROM Restaurant JOIN Type_Of_Restaurant ON Restaurant.ResID =  Type_Of_Restaurant.ResID JOIN Restaurant_Type ON Type_Of_Restaurant.ResTypeID = Restaurant_Type.ResTypeID GROUP BY Type_Of_Restaurant.ResTypeID HAVING Restaurant_Type.ResTypeName = 'Sandwich'":
    "SELECT count(*) FROM Restaurant JOIN Type_Of_Restaurant ON Restaurant.ResID =  Type_Of_Restaurant.ResID JOIN Restaurant_Type ON Type_Of_Restaurant.ResTypeID = Restaurant_Type.ResTypeID WHERE Restaurant_Type.ResTypeName = 'Sandwich'",
    "SELECT T1.fname FROM student AS T1 JOIN lives_in AS T2 ON T1.stuid  =  T2.stuid WHERE T2.dormid IN (SELECT T2.dormid FROM dorm AS T3 JOIN has_amenity AS T4 ON T3.dormid  =  T4.dormid JOIN dorm_amenity AS T5 ON T4.amenid  =  T5.amenid GROUP BY T3.dormid ORDER BY count(*) DESC LIMIT 1)":
    "SELECT T1.fname FROM student AS T1 JOIN lives_in AS T2 ON T1.stuid  =  T2.stuid WHERE T2.dormid IN (SELECT T3.dormid FROM dorm AS T3 JOIN has_amenity AS T4 ON T3.dormid  =  T4.dormid JOIN dorm_amenity AS T5 ON T4.amenid  =  T5.amenid GROUP BY T3.dormid ORDER BY count(*) DESC LIMIT 1)",
    "SELECT count(*) FROM Reservations AS T1 JOIN Rooms AS T2 ON T1.Room  =  T2.RoomId WHERE T2.maxOccupancy  =  T1.Adults + T1.Kids;":
    "SELECT count(*) FROM Reservations AS T1 JOIN Rooms AS T2 ON T1.Room  =  T2.RoomId WHERE T1.Adults + T1.Kids = T2.maxOccupancy;",
    "select sum(population) ,  avg(surfacearea) from country where continent  =  \"north america\" and surfacearea  >  3000":
    "select sum(population) ,  avg(surfacearea) from country where continent  =  \"North America\" and surfacearea  >  3000",
    "select distinct t3.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode join city as t3 on t1.code  =  t3.countrycode where t2.isofficial  =  't' and t2.language  =  'chinese' and t1.continent  =  \"asia\"":
    "select distinct t3.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode join city as t3 on t1.code  =  t3.countrycode where t2.isofficial  =  'T' and t2.language  =  'Chinese' and t1.continent  =  \"Asia\"",
    "select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  \"english\" and isofficial  =  \"t\" union select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  \"dutch\" and isofficial  =  \"t\"":
    "select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  \"English\" and isofficial  =  \"T\" union select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  \"Dutch\" and isofficial  =  \"T\"",
    "select t1.first_name from students as t1 join addresses as t2 on t1.permanent_address_id  =  t2.address_id where t2.country  =  'haiti' or t1.cell_mobile_number  =  '09700166582'":
    "select t1.first_name from students as t1 join addresses as t2 on t1.permanent_address_id  =  t2.address_id where t2.country  =  'Haiti' or t1.cell_mobile_number  =  '09700166582'",
    "select cell_mobile_number from students where first_name  =  'timmothy' and last_name  =  'ward'":
    "select cell_mobile_number from students where first_name  =  'Timmothy' and last_name  =  'Ward'",
    "select name from teacher where hometown != \"little lever urban district\"":
    "select name from teacher where hometown != \"Little Lever Urban District\"",
    "select other_details from paragraphs where paragraph_text like 'korea'":
    "select other_details from paragraphs where paragraph_text like '%Korea%'",
    "SELECT max(LEVEL) FROM manager WHERE Country != \"Australia\t\"":
    "SELECT max(LEVEL) FROM manager WHERE Country != \"Australia\"",
    "SELECT T2.name FROM Certificate AS T1 JOIN Aircraft AS T2 ON T2.aid  =  T1.aid WHERE T2.distance  >  5000 GROUP BY T1.aid ORDER BY count(*)  >=  5":
    "SELECT T2.name , T2.distance FROM Certificate AS T1 JOIN Aircraft AS T2 ON T2.aid  =  T1.aid WHERE T2.distance  >  5000 GROUP BY T1.aid HAVING count(*)  >=  5",
    "SELECT T1.Area FROM APPELLATIONS AS T1 JOIN WINE AS T2 ON T1.Appelation  =  T2.Appelation GROUP BY T2.Appelation HAVING T2.year  <  2010 ORDER BY count(*) DESC LIMIT 1":
    "SELECT T1.Area FROM APPELLATIONS AS T1 JOIN WINE AS T2 ON T1.Appelation  =  T2.Appelation WHERE T2.year  <  2010 GROUP BY T2.Appelation ORDER BY count(*) DESC LIMIT 1",
    "SELECT DISTINCT * FROM employees AS T1 JOIN departments AS T2 ON T1.department_id  =  T2.department_id WHERE T1.employee_id  =  T2.manager_id":
    "SELECT DISTINCT * FROM employees AS T1 JOIN departments AS T2 ON T1.department_id  =  T2.department_id AND T1.employee_id  =  T2.manager_id",
    "SELECT DISTINCT T3.name ,  T2.title ,  T1.stars FROM Rating AS T1 JOIN Movie AS T2 ON T1.mID  =  T2.mID JOIN Reviewer AS T3 ON T1.rID  =  T3.rID WHERE T2.director  =  T3.name":
    "SELECT DISTINCT T3.name ,  T2.title ,  T1.stars FROM Rating AS T1 JOIN Movie AS T2 ON T1.mID  =  T2.mID JOIN Reviewer AS T3 ON T1.rID  =  T3.rID AND T2.director  =  T3.name",
    "SELECT DISTINCT T1.lname FROM Faculty AS T1 JOIN Faculty_participates_in AS T2 ON T1.facID  =  T2.facID JOIN activity AS T3 ON T2.actid  =  T2.actid WHERE T3.activity_name  =  'Canoeing' OR T3.activity_name  =  'Kayaking'":
    "SELECT DISTINCT T1.lname FROM Faculty AS T1 JOIN Faculty_participates_in AS T2 ON T1.facID  =  T2.facID JOIN activity AS T3 ON T2.actid  =  T3.actid WHERE T3.activity_name  =  'Canoeing' OR T3.activity_name  =  'Kayaking'",
    "SELECT lname FROM faculty WHERE rank  =  'Professor' EXCEPT SELECT DISTINCT T1.lname FROM Faculty AS T1 JOIN Faculty_participates_in AS T2 ON T1.facID  =  T2.facID JOIN activity AS T3 ON T2.actid  =  T2.actid WHERE T3.activity_name  =  'Canoeing' OR T3.activity_name  =  'Kayaking'":
    "SELECT lname FROM faculty WHERE rank  =  'Professor' EXCEPT SELECT DISTINCT T1.lname FROM Faculty AS T1 JOIN Faculty_participates_in AS T2 ON T1.facID  =  T2.facID JOIN activity AS T3 ON T2.actid  =  T3.actid WHERE T3.activity_name  =  'Canoeing' OR T3.activity_name  =  'Kayaking'",
    "SELECT T1.lname FROM Faculty AS T1 JOIN Faculty_participates_in AS T2 ON T1.facID  =  T2.facID JOIN activity AS T3 ON T2.actid  =  T2.actid WHERE T3.activity_name  =  'Canoeing' INTERSECT SELECT T1.lname FROM Faculty AS T1 JOIN Faculty_participates_in AS T2 ON T1.facID  =  T2.facID JOIN activity AS T3 ON T2.actid  =  T2.actid WHERE T3.activity_name  =  'Kayaking'":
    "SELECT T1.lname FROM Faculty AS T1 JOIN Faculty_participates_in AS T2 ON T1.facID  =  T2.facID JOIN activity AS T3 ON T2.actid  =  T3.actid WHERE T3.activity_name  =  'Canoeing' INTERSECT SELECT T1.lname FROM Faculty AS T1 JOIN Faculty_participates_in AS T2 ON T1.facID  =  T2.facID JOIN activity AS T3 ON T2.actid  =  T3.actid WHERE T3.activity_name  =  'Kayaking'",
    "SELECT T1.stuid FROM participates_in AS T1 JOIN activity AS T2 ON T2.actid  =  T2.actid WHERE T2.activity_name  =  'Canoeing' INTERSECT SELECT T1.stuid FROM participates_in AS T1 JOIN activity AS T2 ON T2.actid  =  T2.actid WHERE T2.activity_name  =  'Kayaking'":
    "SELECT T1.stuid FROM participates_in AS T1 JOIN activity AS T2 ON T1.actid  =  T2.actid WHERE T2.activity_name  =  'Canoeing' INTERSECT SELECT T1.stuid FROM participates_in AS T1 JOIN activity AS T2 ON T1.actid  =  T2.actid WHERE T2.activity_name  =  'Kayaking'"
}

# incorrect questions: mainly missing/mis-matched values in the question 
question_mappings = {
    "What are the names of the stations that are located in Palo Alto but have never been the ending point of the trips":
    "What are the names of the stations that are located in Palo Alto but have never been the ending point of the trips more than 100 times ?",
    "Find the organisation ids and details of the organisations which are involved in":
    "Find the organisation ids and details of the organisations which have grants more than 6000 dollars ?",
    "What are the ids of the trips that lasted the longest and how long did they last?":
    "What are the ids of the top 3 trips that lasted the longest and how long did they last ?",
    "What are the numbers of the shortest flights?":
    "What are the numbers of the shortest three flights ?",
    "What are the names of the countries and average invoice size of the top countries by size?":
    "What are the names of the countries and average invoice size of the top ten countries by size ?",
    "Return the poll source corresponding to the candidate who has the oppose rate.":
    "Return the poll source corresponding to the candidate who has the highest oppose rate .",
    "List the project details of the projects launched by the organisation":
    "List the project details of the projects launched by the organisation with most projects .",
    "What are the first and last names of all the employees and how many people report to them?":
    "What are the first and last names of all the employees who manage most number of people and how many people report to them ?",
    "What are the advisors":
    "What are the advisors who have at least 2 students ?",
    "What is the name and country of origin for each artist who has released a song with a resolution higher than 900?":
    "What is the name and country of origin for each artist who has released at least one song with a resolution higher than 900 ?",
    "How many undergraduates are there at San Jose State":
    "How many undergraduates are there at San Jose State University in year 2004 ?",
    "What are the id and name of the photos for mountains?":
    "What are the id and name of the photos for mountains higher than 4000 ?",
    "For each dorm, how many amenities does it have?":
    "For each dorm which can accommodate more than 100 students , how many amenities does it have ?",
    "Please give me a list of cities whose regional population is over 8000000 or under 5000000.":
    "Please give me a list of cities whose regional population is over 10000000 or under 5000000 .",
    "Which cities have regional population above 8000000 or below 5000000?":
    "Which cities have regional population above 10000000 or below 5000000 ?",
    "What are the maximum fastest lap speed in races held after 2004 grouped by race name and ordered by year?":
    "What are the maximum fastest lap speed in races held after 2014 grouped by race name and ordered by year ?",
    "For each race name, What is the maximum fastest lap speed for races after 2004 ordered by year?":
    "For each race name , What is the maximum fastest lap speed for races after 2014 ordered by year ?",
    "What are the average fastest lap speed in races held after 2004 grouped by race name and ordered by year?":
    "What are the average fastest lap speed in races held after 2014 grouped by race name and ordered by year ?",
    "What is the average fastest lap speed for races held after 2004, for each race, ordered by year?":
    "What is the average fastest lap speed for races held after 2014 , for each race , ordered by year ?",
    "Show the musical nominee with award \"Bob Fosse\" or \"Cleavant Derricks\".":
    "Show the musical nominee with award \" Tony Award \" or \" Cleavant Derricks \" .",
    "Who are the nominees who were nominated for either of the Bob Fosse or Cleavant Derricks awards?":
    "Who are the nominees who were nominated for either of the Tony Award or Cleavant Derricks awards ?",
    "How many invoices were billed from each state?":
    "How many invoices were billed from each state in the USA ?",
    "What are the states with the most invoices?":
    "What are the states in the USA with the most invoices ?",
    "What are the names of all tracks that belong to the Rock genre and whose media type is MPEG?":
    "What are the names of all tracks that belong to the Rock genre and whose media type is MPEG audio file ?",
    "Which customers have used both the service named \"Close a policy\" and the service named \"Upgrade a policy\"? Give me the customer names.":
    "Which customers have used both the service named \" Close a policy \" and the service named \" New policy application \" ? Give me the customer names .",
    "Show the medicine names and trade names that cannot interact with the enzyme with product 'Heme'.":
    "Show the medicine names and trade names that cannot interact with the enzyme with product ' Protoporphyrinogen IX ' .",
    "What are the names of parties that have both delegates on \"Appropriations\" committee and":
    "What are the names of parties that have both delegates on \" Appropriations \" committee and \" Economic Matters \" committee ?",
    "How many courses does the department of Computer Information Systmes offer?":
    "How many courses does the department of Computer Information Systems offer ?",
    "Find the first name and last name of the instructor of course that has course name":
    "Find the first name and last name of the instructor of course that has course name \" COMPUTER LITERACY \"",
    "What are the names of the districts that have both mall and village store style shops?":
    "What are the names of the districts that have both city mall and village store style shops ?",
    "How many distinct characteristic names does the product \"cumin\" have?":
    "How many distinct characteristic names does the product \" sesame \" have ?",
    "Count the number of different characteristic names the product 'cumin' has.":
    "Count the number of different characteristic names the product ' sesame ' has .",
    "Find out the send dates of the documents with the grant amount of more than 5000 were granted by organisation type described":
    "Find out the send dates of the documents with the grant amount of more than 5000 were granted by organisation type described as ' Research '",
    "I want the papers on keyphrase0 by brian curless":
    "I want the papers on convolution by brian curless",
    "what is the highest point in each state whose lowest point is sea level":
    "what is the highest point in each state whose lowest point is 0"
}

# revise SQL to match values in the question: 
# 1. normalize operators, more than: > , at least: >=
# 2. remove unnecessary values not mentioned in the question
# 3. replace some placeholders with real values in the question
question_query_mappings = {
    "Return the country name and the numbers of languages spoken for each country that speaks at least 3 languages.":
    "SELECT COUNT(T2.Language) ,  T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode GROUP BY T1.Name HAVING COUNT(*)  >=  3",
    "What are the students ids of students who have more than one allergy?":
    "SELECT StuID FROM Has_allergy GROUP BY StuID HAVING count(*)  >  1",
    "For every medicine id, what are the names of the medicines that can interact with more than one enzyme?":
    "SELECT T1.id ,  T1.Name FROM medicine AS T1 JOIN medicine_enzyme_interaction AS T2 ON T2.medicine_id  =  T1.id GROUP BY T1.id HAVING count(*)  >  1",
    "For each movie that received more than 3 reviews, what is the average rating?":
    "SELECT mID ,  avg(stars) FROM Rating GROUP BY mID HAVING count(*)  >  3",
    "What are department ids for departments with managers managing more than 3 employees?":
    "SELECT DISTINCT department_id FROM employees GROUP BY department_id ,  manager_id HAVING COUNT(employee_id)  > 3",
    "What are the job ids for jobs done more than once for a period of more than 300 days?":
    "SELECT job_id FROM job_history WHERE end_date - start_date  > 300 GROUP BY job_id HAVING COUNT(*) > 1",
    "What are the task details, task ids, and project ids for the progrects that are detailed as 'omnis' or have at least 3 outcomes?":
    "SELECT T1.task_details ,  T1.task_id ,  T2.project_id FROM Tasks AS T1 JOIN Projects AS T2 ON T1.project_id  =  T2.project_id WHERE T2.project_details  =  'omnis' UNION SELECT T1.task_details ,  T1.task_id ,  T2.project_id FROM Tasks AS T1 JOIN Projects AS T2 ON T1.project_id  =  T2.project_id JOIN Project_outcomes AS T3 ON T2.project_id  =  T3.project_id GROUP BY T2.project_id HAVING count(*)  >=  3",
    "How many papers were written on Multiuser Receiver in the Decision Feedback this year ?":
    "SELECT COUNT(DISTINCT t3.paperid) FROM paperkeyphrase AS t2 JOIN keyphrase AS t1 ON t2.keyphraseid  =  t1.keyphraseid JOIN paper AS t3 ON t3.paperid  =  t2.paperid WHERE t1.keyphrasename  =  \"Multiuser Receiver in the Decision Feedback\" AND t3.year  =  2016 ;",
    "where is a arabic restaurant on buchanan in san francisco ?":
    "SELECT t2.house_number  ,  t1.name FROM restaurant AS t1 JOIN LOCATION AS t2 ON t1.id  =  t2.restaurant_id WHERE t2.city_name  =  \"san francisco\" AND t2.street_name  =  \"buchanan\" AND t1.food_type  =  \"arabic\" ;",
    "where can i eat arabic food on buchanan in san francisco ?":
    "SELECT t2.house_number  ,  t1.name FROM restaurant AS t1 JOIN LOCATION AS t2 ON t1.id  =  t2.restaurant_id WHERE t2.city_name  =  \"san francisco\" AND t2.street_name  =  \"buchanan\" AND t1.food_type  =  \"arabic\" ;",
    "Count the number of storms in which at least 1 person died.":
    "SELECT count(*) FROM storm WHERE Number_Deaths  >=  1",
    "What are the nicknames of schools whose division is not 1?":
    "SELECT Nickname FROM school_details WHERE Division != \"1\"",
    "journal articles by mohammad rastegari":
    "SELECT DISTINCT t3.paperid FROM writes AS t2 JOIN author AS t1 ON t2.authorid  =  t1.authorid JOIN paper AS t3 ON t2.paperid  =  t3.paperid WHERE t1.authorname  =  \"mohammad rastegari\" ;",
    "Journal Papers by mohammad rastegari":
    "SELECT DISTINCT t3.paperid FROM writes AS t2 JOIN author AS t1 ON t2.authorid  =  t1.authorid JOIN paper AS t3 ON t2.paperid  =  t3.paperid WHERE t1.authorname  =  \"mohammad rastegari\" ;",
    "What are the medicine and trade names that cannot interact with the enzyme with the product 'Heme'?":
    "SELECT name ,  trade_name FROM medicine EXCEPT SELECT T1.name ,  T1.trade_name FROM medicine AS T1 JOIN medicine_enzyme_interaction AS T2 ON T2.medicine_id  =  T1.id JOIN enzyme AS T3 ON T3.id  =  T2.enzyme_id WHERE T3.product  =  'Heme' ",
    "Who are the friends of Alice that are doctors?":
    "SELECT T2.friend FROM Person AS T1 JOIN PersonFriend AS T2 ON T1.name  =  T2.friend WHERE T2.name  =  'Alice' AND T1.job  =  'doctor' ",
    "List from which date and to which date these staff work: project staff of the project which hires the most staffs":
    "SELECT date_from ,  date_to FROM Project_Staff WHERE project_id IN( SELECT project_id FROM Project_Staff GROUP BY project_id ORDER BY count(*) DESC LIMIT 1 )",
    "Find all cities in which there is a restaurant called \" MGM Grand Buffet \"":
    "SELECT t1.city FROM category AS t2 JOIN business AS t1 ON t2.business_id  =  t1.business_id WHERE t1.name  =  \"MGM Grand Buffet\" AND t2.category_name  =  \"restaurant\"; ",
    "List all the reviews by Michelle for Italian restaurant":
    "SELECT t4.text FROM category AS t2 JOIN business AS t1 ON t2.business_id  =  t1.business_id JOIN category AS t3 ON t3.business_id  =  t1.business_id JOIN review AS t4 ON t4.business_id  =  t1.business_id JOIN USER AS t5 ON t5.user_id  =  t4.user_id WHERE t2.category_name  =  \"Italian\" AND t3.category_name  =  \"restaurant\" AND t5.name  =  \"Michelle\"; ",
    "What neighbourhood is restaurant \" Flat Top Grill \" in ?":
    "SELECT t1.neighbourhood_name FROM category AS t3 JOIN business AS t2 ON t3.business_id  =  t2.business_id JOIN neighbourhood AS t1 ON t1.business_id  =  t2.business_id WHERE t2.name  =  \"Flat Top Grill\" AND t3.category_name  =  \"restaurant\"; ",
    "How many people reviewed restaurant \" Vintner Grill \" in 2010 ?":
    "SELECT COUNT ( DISTINCT t4.name ) FROM category AS t2 JOIN business AS t1 ON t2.business_id  =  t1.business_id JOIN review AS t3 ON t3.business_id  =  t1.business_id JOIN USER AS t4 ON t4.user_id  =  t3.user_id WHERE t1.name  =  \"Vintner Grill\" AND t2.category_name  =  \"restaurant\" AND t3.year  =  2010; ",
    "Give me the name and year of opening of the manufacturers that have either less than 10 factories or more than 10 shops.":
    "SELECT name ,  open_year FROM manufacturer WHERE Num_of_Factories  <  10 OR num_of_shops  >  10 ",
    "What is the first, middle, and last name, along with the id and number of enrollments, for the student who enrolled the most in any program?":
    "SELECT T1.student_id ,  T1.first_name ,  T1.middle_name ,  T1.last_name ,  count(*) FROM Students AS T1 JOIN Student_Enrolment AS T2 ON T1.student_id  =  T2.student_id GROUP BY T1.student_id ORDER BY count(*) DESC LIMIT 1"
}

# both wrong in the question and query
question_query_replacement = {
    "What are the country names, area and population which has both roller coasters with speed higher":
    ("What are the country names , area and population which has both roller coasters with speed higher than 60 and slower than 70 ?",
    "SELECT T1.name ,  T1.area ,  T1.population FROM country AS T1 JOIN roller_coaster AS T2 ON T1.Country_ID  =  T2.Country_ID WHERE T2.speed  >  60 AND T2.speed < 70"),
    "what are the major cities in states through which the mississippi runs":
    ("what are the cities with population larger than 150000 in states through which the mississippi runs",
    "SELECT city_name FROM city WHERE population  >  150000 AND state_name IN ( SELECT traverse FROM river WHERE river_name  =  \"mississippi\" );"),
    "What is the description of the type of the company who concluded its contracts most recently?":
    ("What is the type of the company who concluded its contracts most recently ?",
    "SELECT T1.company_type FROM Third_Party_Companies AS T1 JOIN Maintenance_Contracts AS T2 ON T1.company_id  =  T2.maintenance_contract_company_id ORDER BY T2.contract_end_date DESC LIMIT 1"),
    "What are the codes of types of documents of which there are for or more?":
    ("What are the codes of types of documents of which there are four or more?",
    "SELECT document_type_code FROM documents GROUP BY document_type_code HAVING count(*)  >=  4"),
    "What are the staff roles of the staff who":
    ("What are the staff roles of the staff who work from 2003-04-19 to 2016-03-15",
    "SELECT role_code FROM Project_Staff WHERE date_from  >  '2003-04-19' AND date_to  <  '2016-03-15' "),
    "When was \" Kevin Spacey \" born ?":
    ("When was actor \" Kevin Spacey \" born ?",
    "SELECT birth_year FROM actor WHERE name  =  \"Kevin Spacey\"; "),
    "In what year was \" Kevin Spacey \" born ?":
    ("In what year was actor \" Kevin Spacey \" born ?",
    "SELECT birth_year FROM actor WHERE name  =  \"Kevin Spacey\"; "),
    "Where is the birth place of \" Kevin Spacey \"":
    ("Where is the birth place of director \" Kevin Spacey \"",
    "SELECT birth_city FROM director WHERE name  =  \"Kevin Spacey\"; "),
    "In what city was \" Kevin Spacey \" born ?":
    ("In what city was director \" Kevin Spacey \" born ?",
    "SELECT birth_city FROM director WHERE name  =  \"Kevin Spacey\"; "),
    "What is the nationality of \" Kevin Spacey \" ?":
    ("What is the nationality of director \" Kevin Spacey \" ?",
    "SELECT nationality FROM director WHERE name  =  \"Kevin Spacey\"; "),
    "Find all the forenames of distinct drivers who was in position 1 as standing and won?":
    ("Find all the forenames of distinct drivers who was in position 1 as standing ?",
    "SELECT DISTINCT T1.forename FROM drivers AS T1 JOIN driverstandings AS T2 ON T1.driverid = T2.driverid WHERE T2.position = 1"),
    "What are all the different first names of the drivers who are in position as standing and won?":
    ("What are all the different first names of the drivers who are in position 1 as standing ?",
    "SELECT DISTINCT T1.forename FROM drivers AS T1 JOIN driverstandings AS T2 ON T1.driverid = T2.driverid WHERE T2.position = 1"),
    "Find all the forenames of distinct drivers who won in position 1 as driver standing and had more than 20 points?":
    ("Find all the forenames of distinct drivers who were in position 1 as driver standing and had more than 20 points ?",
    "SELECT DISTINCT T1.forename FROM drivers AS T1 JOIN driverstandings AS T2 ON T1.driverid = T2.driverid WHERE T2.position = 1 AND T2.points > 20"),
    "What are the first names of the different drivers who won in position 1 as driver standing and had more than 20 points?":
    ("What are the first names of the different drivers who were in position 1 as driver standing and had more than 20 points ?",
    "SELECT DISTINCT T1.forename FROM drivers AS T1 JOIN driverstandings AS T2 ON T1.driverid = T2.driverid WHERE T2.position = 1 AND T2.points > 20"),
    "Which student has enrolled for the most times in any program? List the id, first name, middle name, last name, the number of enrollments and student id.":
    ("Which student has enrolled for the most times in any program ? List the student id , first name , middle name , last name and the number of enrollments .",
    "SELECT T1.student_id ,  T1.first_name ,  T1.middle_name ,  T1.last_name ,  count(*) FROM Students AS T1 JOIN Student_Enrolment AS T2 ON T1.student_id  =  T2.student_id GROUP BY T1.student_id ORDER BY count(*) DESC LIMIT 1")
}

# missing values in geo examples (38)
geo_major_semantics = {
    "major cities": "cities with population larger than 150000", # population > 150000
    "major city": "city with a population larger than 150000",
    "big cities": "cities with population larger than 150000",
    "major lakes": "lakes with area larger than 750", # area > 750
    "major rivers": "rivers which are longer than 750", # length > 750
    "major river": "river which is longer than 750"
}

def amend_examples_in_dataset(dataset: list, schemas: dict, tables: dict, verbose: bool = False):
    count = 0
    for data in dataset:
        flag = False
        # we only focus the fields question, question_toks, query, and sql
        # never use fields query_toks and query_toks_no_value, thus not modify
        for k in question_mappings:
            if data['question'] == k:
                if verbose:
                    print('DB:', data['db_id'])
                    print('Question:', data['question'])
                    print('Question revised:', question_mappings[k])
                    print('SQL:', data['query'])
                data['question'] = question_mappings[k]
                data['question_toks'] = data['question'].split()
                count += 1
                flag = True
                break
        if flag: continue
        for k in query_mappings:
            if data['query'] == k:
                if verbose:
                    print('DB:', data['db_id'])
                    print('Question:', data['question'])
                    print('SQL:', data['query'])
                    print('SQL revised:', query_mappings[k])
                data['query'] = query_mappings[k]
                db_id = data['db_id']
                data['sql'] = get_sql(Schema(schemas[db_id], tables[db_id]), data['query'])
                count += 1
                flag = True
                break
        if flag: continue
        for k in question_query_mappings:
            if data['question'] == k:
                if verbose:
                    print('DB:', data['db_id'])
                    print('Question:', data['question'])
                    print('SQL:', data['query'])
                    print('SQL revised:', question_query_mappings[k])
                data['query'] = question_query_mappings[k]
                db_id = data['db_id']
                data['sql'] = get_sql(Schema(schemas[db_id], tables[db_id]), data['query'])
                count += 1
                flag = True
                break
        if flag: continue
        for k in question_query_replacement:
            if data['question'] == k:
                if verbose:
                    print('DB:', data['db_id'])
                    print('Question:', data['question'])
                    print('Question revised:', question_query_replacement[k][0])
                    print('SQL:', data['query'])
                    print('SQL revised:', question_query_replacement[k][1])
                data['question'] = question_query_replacement[k][0]
                data['question_toks'] = data['question'].split()
                data['query'] = question_query_replacement[k][1]
                db_id = data['db_id']
                data['sql'] = get_sql(Schema(schemas[db_id], tables[db_id]), data['query'])
                count += 1
                flag = True
                break
        if flag: continue
        if data['db_id'] == 'restaurants' and 'good' in data['question']:
            assert '?' in data['question']
            if verbose:
                print('DB:', data['db_id'])
                print('Question:', data['question'])
                print('Question revised:', data['question'].replace('?', 'with ratings higher than 2.5 ?'))
                print('SQL:', data['query'])
            data['question'] = data['question'].replace('?', 'with ratings higher than 2.5 ?')
            data['question_toks'] = data['question'].split()
            if 'region0' in data['query']:
                if verbose:
                    print('SQL revised:', data['query'].replace('region0', 'bay area'))
                data['query'] = data['query'].replace('region0', 'bay area')
                db_id = data['db_id']
                data['sql'] = get_sql(Schema(schemas[db_id], tables[db_id]), data['query'])
            count += 1
            continue
        if data['db_id'] == 'geo':
            for k in geo_major_semantics.keys():
                if k in data['question']:
                    data['question'] = data['question'].replace(k, geo_major_semantics[k])
                    flag = True
            if flag:
                if verbose:
                    print('DB:', data['db_id'])
                    print('Question:', data['question'])
                    print('SQL:', data['query'])
                data['question_toks'] = data['question'].split()
                count += 1
                continue
        # fix some parsing errors: table alias contradiction when the same table alias points to different tables in nested SQLs
        data['sql'] = get_sql(Schema(schemas[data['db_id']], tables[data['db_id']]), data['query'])

    print('Fix %d examples in the dataset' % (count))
    return dataset

if __name__ == '__main__':

    verbose = False
    data_dir, db_dir = DATASETS['spider']['data'], DATASETS['spider']['database']
    table_path = os.path.join(data_dir, 'tables.json')
    origin_table_path = os.path.join(data_dir, 'tables.original.json')
    update_table_path = origin_table_path if os.path.exists(origin_table_path) else table_path
    tables = amend_primary_keys(json.load(open(update_table_path, 'r')), verbose=verbose)
    tables = amend_foreign_keys(tables, verbose=verbose)
    tables = amend_boolean_types(tables, db_dir, verbose=verbose)
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
        dataset = amend_examples_in_dataset(dataset, schemas, tables, verbose=verbose)
        json.dump(dataset, open(dataset_path, 'w'), indent=4)
