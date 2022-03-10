#coding=utf8
import os, json, re, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import shutil, collections, itertools
from utils.constants import DATASETS
from preprocess.process_utils import load_db_contents, extract_db_contents
from preprocess.cspider.process_sql import get_sql
from preprocess.cspider.parse_raw_json import get_schemas_from_json, Schema

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

def is_new_bool_candidate(cells, c_name, old_type):
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
    vset = set([str(v[0]).lower() for v in cells])
    evidence_set = [{'y', 'n'}, {'yes', 'no'}, {'t', 'f'}, {'true', 'false'}]
    if len(vset) > 0 and any(vset == s for s in evidence_set): return True
    return False

def amend_boolean_types(tables: list, db_dir: str = 'data/cspider/db_content.json', verbose: bool = True):
    count, contents = 0, load_db_contents(db_dir)
    for db in tables:
        db_id, table_list = db['db_id'], db['table_names_original']
        column_list, column_types = db['column_names_original'], db['column_types']
        cells = extract_db_contents(contents, db)
        for j, (t_id, c_name) in enumerate(column_list):
            if c_name == '*': continue
            t_name, old_type, column_cells = table_list[t_id], column_types[j], cells[j]
            if is_new_bool_candidate(column_cells, c_name, old_type):
                if verbose:
                    print('DB[{}]: revise column[{}.{}] type: {}->boolean'.format(db_id, t_name, c_name, old_type))
                db['column_types'][j] = 'boolean'
                count += 1
    print('{} column types are changed to boolean'.format(count))
    return tables

train_qids = ["qid190", "qid191", "qid237", "qid238", "qid336", "qid337", "qid503", "qid514", "qid517", "qid559", "qid768", "qid769", "qid778", "qid779", "qid988", "qid989", "qid1000", "qid1001", "qid1002", "qid1003", "qid1078", "qid1079", "qid1106", "qid1107", "qid1120", "qid1121", "qid1140", "qid1143", "qid1302", "qid1303", "qid1304", "qid1305", "qid1592", "qid1593", "qid1622", "qid1623", "qid1640", "qid1641", "qid1642", "qid1643", "qid1899", "qid1900", "qid1933", "qid1934", "qid1935", "qid1936", "qid1937", "qid1938", "qid2070", "qid2089", "qid2090", "qid2399", "qid2400", "qid2491", "qid2495", "qid2928", "qid3249", "qid3451", "qid3452", "qid3515", "qid3516", "qid3517", "qid3518", "qid3525", "qid3526", "qid3535", "qid3536", "qid3601", "qid3602", "qid3603", "qid3604", "qid3617", "qid3618", "qid3649", "qid3650", "qid3673", "qid3674", "qid3675", "qid3676", "qid3677", "qid3678", "qid3679", "qid3680", "qid3686", "qid3700", "qid3765", "qid3766", "qid3801", "qid3802", "qid3807", "qid3808", "qid3879", "qid3880", "qid3945", "qid3946", "qid3977", "qid3983", "qid4002", "qid4003", "qid4016", "qid4017", "qid4018", "qid4019", "qid4102", "qid4103", "qid4106", "qid4107", "qid4110", "qid4111", "qid4120", "qid4121", "qid4252", "qid4253", "qid4473", "qid4763", "qid4777", "qid4778", "qid4839", "qid4840", "qid4841", "qid4842", "qid4845", "qid4846", "qid4847", "qid4848", "qid4849", "qid4850", "qid5762", "qid5763", "qid5792", "qid5793", "qid5794", "qid5795", "qid5808", "qid5809", "qid5871", "qid5877", "qid5918", "qid5919", "qid5943", "qid5945", "qid5997", "qid5998", "qid6002", "qid6009", "qid6010", "qid6013", "qid6014", "qid6015", "qid6016", "qid6093", "qid6094", "qid6121", "qid6122", "qid6153", "qid6154", "qid6199", "qid6200", "qid6201", "qid6202", "qid6203", "qid6211", "qid6212", "qid6225", "qid6241", "qid6347", "qid6348", "qid6380", "qid6408", "qid6419", "qid6452", "qid6453", "qid6454", "qid6504", "qid6505", "qid6606", "qid6607"]

dev_qids = ["qid46", "qid47", "qid110", "qid111", "qid112", "qid113", "qid161", "qid162", "qid292", "qid293", "qid330", "qid331", "qid367", "qid590", "qid591", "qid681", "qid779", "qid801", "qid807", "qid809"]

question_query_mappings = {
    '自2009以来“诺亚·史密斯”共同参与创作了多少篇论文？': 'SELECT DISTINCT COUNT ( DISTINCT t2.paperid ) FROM writes AS t2 JOIN author AS t1 ON t2.authorid  =  t1.authorid JOIN paper AS t3 ON t2.paperid  =  t3.paperid WHERE t1.authorname != "诺亚·史密斯" AND t3.year  >  2009 AND t2.paperid IN ( SELECT t2.paperid FROM writes AS t2 JOIN author AS t1 ON t2.authorid  =  t1.authorid WHERE t1.authorname LIKE "诺亚·史密斯" );',
    '国家元首是“布什”的国家的官方语言是什么？': 'SELECT T2.Language FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T1.HeadOfState  =  "布什" AND T2.IsOfficial  =  "T"',
    '国家元首是“布什”的国家使用的官方语言是什么？': 'SELECT T2.Language FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T1.HeadOfState  =  "布什" AND T2.IsOfficial  =  "T"',
    '从“Initial”一词开始描述的文档的类型是什么？': "SELECT document_type_code FROM Document_Types WHERE document_description LIKE 'initial'",
    '找出犹他各州的银行的平均客户数。': "SELECT avg(no_of_customers) FROM bank WHERE state  =  '犹他'",
    '返回由名为“山东航空公司”运营的以“意大利”机场为目的地的航线数。': 'SELECT count(*) FROM routes AS T1 JOIN airports AS T2 ON T1.dst_apid  =  T2.apid JOIN airlines AS T3 ON T1.alid  =  T3.alid WHERE T2.country  =  "意大利" AND T3.name  =  "山东航空公司"',
    '“山东航空公司”运营的以“意大利”机场为目的地的航线数是多少？': 'SELECT count(*) FROM routes AS T1 JOIN airports AS T2 ON T1.dst_apid  =  T2.apid JOIN airlines AS T3 ON T1.alid  =  T3.alid WHERE T2.country  =  "意大利" AND T3.name  =  "山东航空公司"',
    '将标题以字母顺序排序，列出每个专辑。': "SELECT title FROM albums ORDER BY title;",
    '以字母顺序排序的所有专辑的标题是什么？': "SELECT title FROM albums ORDER BY title;",
    '找出使用电子邮件作为联系渠道的客户的姓名和活动日期。': 'SELECT t1.customer_name ,  t2.active_from_date FROM customers AS t1 JOIN customer_contact_channels AS t2 ON t1.customer_id  =  t2.customer_id WHERE t2.channel_code  =  "电子邮件"',
    '联系渠道代码是电子邮件的客户的姓名和活动日期是什么？': 'SELECT t1.customer_name ,  t2.active_from_date FROM customers AS t1 JOIN customer_contact_channels AS t2 ON t1.customer_id  =  t2.customer_id WHERE t2.channel_code  =  "电子邮件"',
    '找到高清晰度电视频道的套餐选择和系列节目名称。': 'SELECT package_option ,  series_name FROM TV_Channel WHERE hight_definition_TV  =  "T"',
    '支持高清电视的电视频道的套餐选项和系列节目名称是什么？': 'SELECT package_option ,  series_name FROM TV_Channel WHERE hight_definition_TV  =  "T"',
    '那些以停车或购物为特色的旅游景点叫什么名字？': 'SELECT T1.Name FROM Tourist_Attractions AS T1 JOIN Tourist_Attraction_Features AS T2 ON T1.tourist_attraction_id  =  T2.tourist_attraction_id JOIN Features AS T3 ON T2.Feature_ID  =  T3.Feature_ID WHERE T3.feature_Details  =  "停车" UNION SELECT T1.Name FROM Tourist_Attractions AS T1 JOIN Tourist_Attraction_Features AS T2 ON T1.tourist_attraction_id  =  T2.tourist_attraction_id JOIN Features AS T3 ON T2.Feature_ID  =  T3.Feature_ID WHERE T3.feature_Details  =  "购物"',
    '找出有停车或购物作为特色细节的旅游景点。景点名称是什么？': 'SELECT T1.Name FROM Tourist_Attractions AS T1 JOIN Tourist_Attraction_Features AS T2 ON T1.tourist_attraction_id  =  T2.tourist_attraction_id JOIN Features AS T3 ON T2.Feature_ID  =  T3.Feature_ID WHERE T3.feature_Details  =  "停车" UNION SELECT T1.Name FROM Tourist_Attractions AS T1 JOIN Tourist_Attraction_Features AS T2 ON T1.tourist_attraction_id  =  T2.tourist_attraction_id JOIN Features AS T3 ON T2.Feature_ID  =  T3.Feature_ID WHERE T3.feature_Details  =  "购物"',
    '列出不在东边的省的不同警察机关。': 'SELECT DISTINCT Police_force FROM county_public_safety WHERE LOCATION != "东"',
    '不位于东部的省有哪些不同的警察机关？': 'SELECT DISTINCT Police_force FROM county_public_safety WHERE LOCATION != "东"',
    '显示东部和西部共同管辖的警察机构。': 'SELECT Police_force FROM county_public_safety WHERE LOCATION  =  "东" INTERSECT SELECT Police_force FROM county_public_safety WHERE LOCATION  =  "西"',
    '在东部和西部的两个省都有哪些警察机构同时管辖？': 'SELECT Police_force FROM county_public_safety WHERE LOCATION  =  "东" INTERSECT SELECT Police_force FROM county_public_safety WHERE LOCATION  =  "西"',
    '平均有14辆以上可用自行车或在12月安装自行车的所有车站的名称和ID是什么？': 'SELECT T1.name ,  T1.id FROM station AS T1 JOIN status AS T2 ON T1.id  =  T2.station_id GROUP BY T2.station_id HAVING avg(T2.bikes_available)  >  14 UNION SELECT name ,  id FROM station WHERE installation_date LIKE "%12%"',
    "平均有14辆以上可用自行车或在12月份安装自行车的车站的名称和ID是什么？": 'SELECT T1.name ,  T1.id FROM station AS T1 JOIN status AS T2 ON T1.id  =  T2.station_id GROUP BY T2.station_id HAVING avg(T2.bikes_available)  >  14 UNION SELECT name ,  id FROM station WHERE installation_date LIKE "%12%"',
    "找到由“创新实验室”和“索尼”两家公司生产的产品的名称。": "SELECT T1.name FROM products AS T1 JOIN manufacturers AS T2 ON T1.Manufacturer  =  T2.code WHERE T2.name  =  '创新实验室' INTERSECT SELECT T1.name FROM products AS T1 JOIN manufacturers AS T2 ON T1.Manufacturer  =  T2.code WHERE T2.name  =  '索尼'",
    "“创新实验室”和“索尼”生产的产品名称是什么？": "SELECT T1.name FROM products AS T1 JOIN manufacturers AS T2 ON T1.Manufacturer  =  T2.code WHERE T2.name  =  '创新实验室' INTERSECT SELECT T1.name FROM products AS T1 JOIN manufacturers AS T2 ON T1.Manufacturer  =  T2.code WHERE T2.name  =  '索尼'",
    '找出那些排名积分最高的并且参加过“澳大利亚公开赛”的获胜者的名字。': "SELECT winner_name FROM matches WHERE tourney_name  =  '澳大利亚公开赛' ORDER BY winner_rank_points DESC LIMIT 1",
    "参加“澳大利亚公开赛”的排名积分最高的获胜者叫什么名字？": "SELECT winner_name FROM matches WHERE tourney_name  =  '澳大利亚公开赛' ORDER BY winner_rank_points DESC LIMIT 1",
    "按年份排序，在2004年以后按比赛名称分组的比赛最快圈速是多少？": "SELECT max(T2.fastestlapspeed) ,  T1.name ,  T1.year FROM races AS T1 JOIN results AS T2 ON T1.raceid = T2.raceid WHERE T1.year > 2004 GROUP BY T1.name ORDER BY T1.year",
    "在2004年后的每个比赛中比赛的最快圈速是多少？": "SELECT max(T2.fastestlapspeed) ,  T1.name ,  T1.year FROM races AS T1 JOIN results AS T2 ON T1.raceid = T2.raceid WHERE T1.year > 2004 GROUP BY T1.name ORDER BY T1.year",
    "按年份排序，在2004年按比赛名称分组举办的比赛平均最快圈速是多少？": "SELECT avg(T2.fastestlapspeed) ,  T1.name ,  T1.year FROM races AS T1 JOIN results AS T2 ON T1.raceid = T2.raceid WHERE T1.year > 2004 GROUP BY T1.name ORDER BY T1.year",
    "按年份排序，2004年后举行的每场比赛的平均最快圈速是多少？": "SELECT avg(T2.fastestlapspeed) ,  T1.name ,  T1.year FROM races AS T1 JOIN results AS T2 ON T1.raceid = T2.raceid WHERE T1.year > 2004 GROUP BY T1.name ORDER BY T1.year",
    "在1974年的8缸汽车的最小重量是多少？": "SELECT Weight FROM CARS_DATA WHERE Cylinders  =  8 AND YEAR  =  1974 ORDER BY Weight ASC LIMIT 1;",
    "1974年所生产的8缸汽车的最小重量是多少？": "SELECT Weight FROM CARS_DATA WHERE Cylinders  =  8 AND YEAR  =  1974 ORDER BY Weight ASC LIMIT 1;",
    "在不是最小马力的汽车中，那些少于4个汽缸的汽车制造商的ID和名称是什么？": "SELECT T2.MakeId ,  T2.Make FROM CARS_DATA AS T1 JOIN CAR_NAMES AS T2 ON T1.Id  =  T2.MakeId WHERE T1.Horsepower  >  (SELECT min(Horsepower) FROM CARS_DATA) AND T1.Cylinders  <  4;",
    "请出示与总旅客数超过10万人的机场有关的飞机的名称和说明。": "SELECT T1.Aircraft ,  T1.Description FROM aircraft AS T1 JOIN airport_aircraft AS T2 ON T1.Aircraft_ID  =  T2.Aircraft_ID JOIN airport AS T3 ON T2.Airport_ID  =  T3.Airport_ID WHERE T3.Total_Passengers  >  100000",
    "与总旅客数超过10万人的机场有关的飞机的名称和描述是什么？": "SELECT T1.Aircraft ,  T1.Description FROM aircraft AS T1 JOIN airport_aircraft AS T2 ON T1.Aircraft_ID  =  T2.Aircraft_ID JOIN airport AS T3 ON T2.Airport_ID  =  T3.Airport_ID WHERE T3.Total_Passengers  >  100000",
    "显示至少有5000和不超过40000座位数的赛道开放的年份。": "SELECT year_opened FROM track WHERE seating BETWEEN 5000 AND 40000",
    "返回人口在160000到900000之间的城市的名字。": "SELECT name FROM city WHERE Population BETWEEN 160000 AND 900000",
    "在2013年以后音乐会最多的体育场名称和容量是多少？": "SELECT T2.name ,  T2.capacity FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id WHERE T1.year  >  2013 GROUP BY T2.stadium_id ORDER BY count(*) DESC LIMIT 1",
    "请告诉我工厂少于10家或商店多于10家的制造商的名称和开业年份。": "SELECT name ,  open_year FROM manufacturer WHERE Num_of_Factories  <  10 OR num_of_shops  >  10",
    "有多个过敏反应的的学生的学生ID是什么？": "SELECT StuID FROM Has_allergy GROUP BY StuID HAVING count(*)  >  1",
    "持续时间最长的路线ID是什么？路线时间多长？": "SELECT id ,  duration FROM trip ORDER BY duration DESC LIMIT 1",
    "位于“西安”但从未成为路线终点的车站叫什么名字？": 'SELECT name FROM station WHERE city  =  "西安" EXCEPT SELECT end_station_name FROM trip GROUP BY end_station_name HAVING count(*)  >  0',
    "最短航班的号码是多少？": "SELECT flno FROM Flight ORDER BY distance ASC LIMIT 1",
    "列出曾经购买产品“食物”的顾客的名字。": 'SELECT T1.customer_name FROM customers AS T1 JOIN orders AS T2 JOIN order_items AS T3 JOIN products AS T4 ON T1.customer_id = T2.customer_id AND T2.order_id = T3.order_id AND T3.product_id = T4.product_id WHERE T4.product_name = "食物" GROUP BY T1.customer_id HAVING count(*)  >  0',
    "列出曾经取消购买产品“食品”的客户的姓名（项目状态为“取消”）。": 'SELECT T1.customer_name FROM customers AS T1 JOIN orders AS T2 JOIN order_items AS T3 JOIN products AS T4 ON T1.customer_id = T2.customer_id AND T2.order_id = T3.order_id AND T3.product_id = T4.product_id WHERE T3.order_item_status = "取消" AND T4.product_name = "食品" GROUP BY T1.customer_id HAVING count(*)  >  0',
    "哪些客户曾经取消购买产品“食品”（项目状态是“取消”）？": 'SELECT T1.customer_name FROM customers AS T1 JOIN orders AS T2 JOIN order_items AS T3 JOIN products AS T4 ON T1.customer_id = T2.customer_id AND T2.order_id = T3.order_id AND T3.product_id = T4.product_id WHERE T3.order_item_status = "取消" AND T4.product_name = "食品" GROUP BY T1.customer_id HAVING count(*)  >  0',
    "找到具有两个预备课程的课程的名称、学分和系名？": "SELECT T1.title ,  T1.credits , T1.dept_name FROM course AS T1 JOIN prereq AS T2 ON T1.course_id  =  T2.course_id GROUP BY T2.course_id HAVING count(*)  =  2",
    "管理超过3名员工的经理所在部门的部门id是什么？": "SELECT DISTINCT department_id FROM employees GROUP BY department_id ,  manager_id HAVING COUNT(employee_id)  > 3",
    "对于不止一次超过300天才完成的工作，这些工作的ID是什么？": "SELECT job_id FROM job_history WHERE end_date - start_date  > 300 GROUP BY job_id HAVING COUNT(*) > 1",
    "对于每个药物ID，可以与一种以上酶相互作用的药物的名称是什么？": "SELECT T1.id ,  T1.Name FROM medicine AS T1 JOIN medicine_enzyme_interaction AS T2 ON T2.medicine_id  =  T1.id GROUP BY T1.id HAVING count(*)  >  1",
    "对于每部超过3次评论的电影的平均评级是多少？": "SELECT mID ,  avg(stars) FROM Rating GROUP BY mID HAVING count(*)  >  3",
    "平均发票大小最大的国家的名称和平均发票大小是多少？": "SELECT billing_country ,  AVG(total) FROM invoices GROUP BY billing_country ORDER BY AVG(total) DESC LIMIT 1;",
    "对于被详细描述为“详细”或者至少有3个结果的项目的任务细节、任务id和项目id是什么？": "SELECT T1.task_details ,  T1.task_id ,  T2.project_id FROM Tasks AS T1 JOIN Projects AS T2 ON T1.project_id  =  T2.project_id WHERE T2.project_details  =  '详细' UNION SELECT T1.task_details ,  T1.task_id ,  T2.project_id FROM Tasks AS T1 JOIN Projects AS T2 ON T1.project_id  =  T2.project_id JOIN Project_outcomes AS T3 ON T2.project_id  =  T3.project_id GROUP BY T2.project_id HAVING count(*)  >=  3",
    "哪些产品被最少的客户投诉？": "SELECT DISTINCT t1.product_name FROM products AS t1 JOIN complaints AS t2 ON t1.product_id  =  t2.product_id JOIN customers AS t3 ON t2.customer_id = t3.customer_id GROUP BY t3.customer_id ORDER BY count(*) LIMIT 1",
    "返回至少使用3种语言的不同国家名称和语言数量。": "SELECT COUNT(T2.Language) ,  T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode GROUP BY T1.Name HAVING COUNT(*)  >=  3",
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
    "SELECT channel_code ,  contact_number FROM customer_contact_channels WHERE active_to_date - active_from_date  =  (SELECT active_to_date - active_from_date FROM customer_contact_channels ORDER BY (active_to_date - active_from_date) DESC LIMIT 1)":
    "SELECT channel_code ,  contact_number FROM customer_contact_channels WHERE active_to_date - active_from_date  =  (SELECT active_to_date - active_from_date FROM customer_contact_channels ORDER BY active_to_date - active_from_date DESC LIMIT 1)",
    "SELECT river_name FROM river GROUP BY ( river_name ) ORDER BY COUNT ( DISTINCT traverse ) DESC LIMIT 1;":
    "SELECT river_name FROM river GROUP BY river_name ORDER BY COUNT ( DISTINCT traverse ) DESC LIMIT 1;",
    "SELECT T1.name FROM accounts AS T1 JOIN checking AS T2 ON T1.custid  =  T2.custid WHERE T2.balance  >  (SELECT avg(balance) FROM checking) INTERSECT SELECT T1.name FROM accounts AS T1 JOIN savings AS T2 ON T1.custid  =  T2.custid WHERE T2.balance  <  (SELECT avg(balance) FROM savings)":
    "SELECT T1.name FROM accounts AS T1 JOIN checking AS T2 ON T1.custid  =  T2.custid WHERE T2.balance  >  (SELECT avg(balance) FROM checking) INTERSECT SELECT T3.name FROM accounts AS T3 JOIN savings AS T4 ON T3.custid  =  T4.custid WHERE T4.balance  <  (SELECT avg(balance) FROM savings)",
    "SELECT T2.balance FROM accounts AS T1 JOIN checking AS T2 ON T1.custid  =  T2.custid WHERE T1.name IN (SELECT T1.name FROM accounts AS T1 JOIN savings AS T2 ON T1.custid  =  T2.custid WHERE T2.balance  >  (SELECT avg(balance) FROM savings))":
    "SELECT T2.balance FROM accounts AS T1 JOIN checking AS T2 ON T1.custid  =  T2.custid WHERE T1.name IN (SELECT T3.name FROM accounts AS T3 JOIN savings AS T4 ON T3.custid  =  T4.custid WHERE T4.balance  >  (SELECT avg(balance) FROM savings))"
}

question_query_replacement = {
    "哪些学生报名参加任何项目的次数最多？列出id、名字、中间名、姓氏、参加次数和学生id。": (
        "哪些学生报名参加任何项目的次数最多？列出学生id、名字、中间名、姓氏和参加次数。",
        "SELECT T1.student_id ,  T1.first_name ,  T1.middle_name ,  T1.last_name ,  count(*) FROM Students AS T1 JOIN Student_Enrolment AS T2 ON T1.student_id  =  T2.student_id GROUP BY T1.student_id ORDER BY count(*) DESC LIMIT 1"
    ),
    "拥有更快速的过山车的国家名称，区域和人口是什么？": (
        "拥有过山车的国家名称，区域和人口是什么？",
        "SELECT T1.name ,  T1.area ,  T1.population FROM country AS T1 JOIN roller_coaster AS T2 ON T1.Country_ID  =  T2.Country_ID"
    ),
    "请问超过花费超过160且可以住两个人的所有房间的名称和ID是什么？": (
        "请问超过花费超过160且至少可以住两个人的所有房间的名称和ID是什么？",
        "SELECT roomName ,  RoomId FROM Rooms WHERE basePrice  >  160 AND maxOccupancy  >=  2;"
    ),
    "“长江”流经省内的主要城市有哪些？": (
        "“长江”流经省内的主要城市（人口大于150000）有哪些？",
        'SELECT city_name FROM city WHERE population  >  150000 AND state_name IN ( SELECT traverse FROM river WHERE river_name  =  "长江" );'
    ),
    '显示名字不包含字母“M”的雇员的全名（名字和姓氏）、雇用日期、工资和部门号码。': (
        '显示名字不包含字母“X”的雇员的全名（名字和姓氏）、雇用日期、工资和部门号码。',
        "SELECT first_name ,  last_name ,  hire_date ,  salary ,  department_id FROM employees WHERE first_name NOT LIKE 'X'"
    ),
    '名字中没有字母“M”的员工的全名、聘用日期、薪水和部门ID是什么？': (
        '名字中没有字母“X”的员工的全名、聘用日期、薪水和部门ID是什么？',
        "SELECT first_name ,  last_name ,  hire_date ,  salary ,  department_id FROM employees WHERE first_name NOT LIKE 'X'"
    ),
    '显示名字不包含字母“M”的雇员的全名（姓氏和名字）、雇用日期、工资和部门编号，并将结果按部门编号升序排列。': (
        '显示名字不包含字母“X”的雇员的全名（姓氏和名字）、雇用日期、工资和部门编号，并将结果按部门编号升序排列。',
        "SELECT first_name ,  last_name ,  hire_date ,  salary ,  department_id FROM employees WHERE first_name NOT LIKE 'X' ORDER BY department_id"
    ),
    '名字中没有字母“M”的员工的全名、雇佣数据、工资和部门ID是什么？': (
        '名字中没有字母“X”的员工的全名、雇佣数据、工资和部门ID是什么？',
        "SELECT first_name ,  last_name ,  hire_date ,  salary ,  department_id FROM employees WHERE first_name NOT LIKE 'X' ORDER BY department_id"
    )
}

question_mappings = {
    "什么是导师？": "什么导师至少有两个学生",
    "每个宿舍都有多少设施？": "每个能容纳超过100个学生的宿舍都有多少设施？",
    "“湖南大学”有多少本科生？": "“湖南大学”2004年有多少本科生？",
    "找出参与的组织的组织ID和详细信息": "找出总资助超过6000美元的组织的组织ID和详细信息",
    "每位发行了一首分辨率超过900的歌曲的艺术家的名字和原籍国是什么？": "每位发行了至少一首分辨率超过900的歌曲的艺术家的名字和原籍国是什么？",
    "按字母降序排列的所有30岁或30岁的飞行员的名字是什么？": "按字母降序排列的所有30岁及其以下的飞行员的名字是什么？",
    "工资在8000和120000范围内的雇员的电话号码是多少？": "工资在8000和12000范围内的雇员的电话号码是多少？",
    "显示最高工资在1200至1800范围内的职位名称和最低和最高工资之间的差额。": "显示最高工资在12000至18000范围内的职位名称和最低和最高工资之间的差额。",
    "最高工资在12000至1800之间的工作名称，以及工资范围是什么？": "最高工资在12000至18000之间的工作名称，以及工资范围是什么？",
    "那些没有佣金，工资在7000到120000之间，而且在部门50工作的员工的电子邮件是什么？": "那些没有佣金，工资在7000到12000之间，而且在部门50工作的员工的电子邮件是什么？",
    "你能返回目前工资在120000及以上的员工所从事的工作的详细信息吗？": "你能返回目前工资在12000及以上的员工所从事的工作的详细信息吗？",
    '有多少女生对“鸡蛋”或“Eggs"过敏？': '有多少女生对“鸡蛋”或“牛奶"过敏？',
    '找出对“Cat”或“猫”过敏的学生的不同名字和城市。': '找出对“牛奶”或“猫”过敏的学生的不同名字和城市。',
    "找出所有已经发布了900首以上分辨率歌曲的艺术家的名字和原籍国。": "找出所有至少已经发布了一首900以上分辨率歌曲的艺术家的名字和原籍国。",
    '2016 ACL论文的标题中包含“神经注意力”': '2016 ACL论文的标题中包含“neural attention”',
    '产品描述重包含汉字t”的产品其类别描述是什么？': '产品描述重包含汉字“强”的产品其类别描述是什么？',
    '作者“Olin Shivers”写了哪些论文？给我论文名。': '作者“曹”“子建”写了哪些论文？给我论文名。',
    '找到作者中有“Olin Shivers”的论文。': '找到作者中有“曹”“子建”的论文。',
    '去年ACL发出的关于“句法分析”的论文': '2012年ACL发出的关于“句法分析”的论文',
    '“克里斯蒂夫·达勒马斯”在去年论文中使用的关键词': '“克里斯蒂夫·达勒马斯”在2000年论文中使用的关键词',
    '今年在“多用户接收机的决策反馈领域”中写了多少篇论文？': '2016年在“多用户接收机的决策反馈领域”中写了多少篇论文？',
    '去年有多少论文发表在“自然通讯”上': '2015年有多少论文发表在“自然通讯”上',
    '关于“卷积神经网络”的方向在过去的一年里写了多少篇论文？': '关于“卷积神经网络”的方向在过去的2016年里写了多少篇论文？',
    '今年“卷积神经网络”方向是上写了多少篇论文？': '2016年“卷积神经网络”方向是上写了多少篇论文？',
    '在过去的一年里，已经发表了多少关于“卷积神经网络”领域的论文？': '在过去的2016年里，已经发表了多少关于“卷积神经网络”领域的论文？',
    '今年有关的“问答”方向写了哪些论文？': '2016年有关的“问答”方向写了哪些论文？',
    '今年有多少出版物被添加到“细胞”杂志上？': '2015年有多少出版物被添加到“细胞”杂志上？',
    '去年没有发表的论文': '2015年没有发表的论文',
    '今年CVPR会议最流行的论文是什么？': '2016年CVPR会议最流行的论文是什么？',
    "“2009-07-05”以后和“2009-07-05”之前雇佣的雇员的职位id和雇佣日期是什么？": '“2007-11-05”以后和“2009-07-05”之前雇佣的雇员的职位id和雇佣日期是什么？',
    "显示获得“Bob Fosse”或“金像奖”奖的被提名音乐人。": "显示获得“金鸡奖”或“金像奖”奖的被提名音乐人。",
    "谁被提名过“Bob Fosse”或“金像奖”奖项？": "谁被提名过“金鸡奖”或“金像奖”奖项？",
    '有多少客户有电子邮件包含"gmail.com"？': '有多少客户有电子邮件包含"qq.com"？',
    '计算包含"gmail.com"的电子邮件的客户数量。': '计算包含"qq.com"的电子邮件的客户数量。',
    '显示既有“banking”的公司也有“oil and gas”行业的公司的总部。': '显示既有“银行”的公司也有“天然气”行业的公司的总部。',
    "哪些论文的作者中有“Stephanie Weirich”？": '哪些论文的作者中有“史”“俊文”？',
    "找到作者“Stephanie Weirich”写的论文的标题。": '找到作者“史”“俊文”写的论文的标题。',
    "哪个国家参加过最多的比赛？": "哪个国家参加过最多的锦标赛？",
    "找到“课程ID、课程名称”课程的入学日期。": "找到“西班牙语”课程的入学日期。",
    "叫“Timothy Ward”的学生的手机号码是多少？": "叫“钟”“睿”的学生的手机号码是多少？",
    '每个州有多少发票？': "美国每个州有多少发票？",
    '发票最多的州是什么州？': '发票最多的州是美国什么州？',
    '关于“低频”的论文': 'NIPS关于“低频”的论文',
    '谁是“丽丽”的“医生”朋友？': '谁是“丽丽”的“医生”男性朋友？',
    '返回总部设在“东京”或“台湾”的公司的总收益。': '返回总部设在“日本”或“台湾”的公司的总收益。',
    '所有在“东京”或“台湾”设有总办事处的公司的总收益是多少？': '所有在“日本”或“台湾”设有总办事处的公司的总收益是多少？',
    '找出总部在“东京”或“北京”的制造商的数量。': '找出总部在“日本”或“北京”的制造商的数量。',
    '有多少制造商在“东京”或“北京”拥有总部？': "有多少制造商在“日本”或“北京”拥有总部？",
    '显示埃克森美孚公司的加油站的经理的名字。': '显示石化公司的加油站的经理的名字。',
    '埃克森美孚公司经营的加油站经理的名字是什么？': '石化公司经营的加油站经理的名字是什么？',
    '同一个姓克拉拉的员工的全名和聘用日期是什么？': '同“明明”在一个部门的员工的全名和聘用日期是什么？',
    '给出名字是刘易斯的司机参赛的比赛名称和比赛的年份。': '给出名字是“姚”的司机参赛的比赛名称和比赛的年份。',
    '姓氏是刘易斯的司机参加的所有比赛的名称和年份是什么？': '名字是"姚"的司机参加的所有比赛的名称和年份是什么？',
    '哪些客户使用了名为“取消保单”的服务和名为"Upgrade a policy"的服务？给我客户的名字。': '哪些客户使用了名为“取消保单”的服务和名为"新开保单"的服务？给我客户的名字。',
    '有过“Mortgage”贷款和“汽车”贷款的客户的名字是什么？': '有过“抵押”贷款和“汽车”贷款的客户的名字是什么？',
    '“厦门”最好的中餐厅是哪一家？': '“厦门”最好的“四川”餐厅是哪一家？',
    '“阿里·法拉迪”于2016年的论文': '“阿里·法拉迪”于2016年的eccv论文'
}

def fix_chinese_english_mismatch_values(ex, choice='train', verbose=True):
    if (choice == 'train' and ex['question_id'] in train_qids) or (choice == 'dev' and ex['question_id'] in dev_qids):
        question_wrappers = r'["“”\'](.*?)["“”\']'
        sql_wrappers = r'["\'](.*?)["\']'
        question_values = [span.group(1) for span in re.finditer(question_wrappers, ex['question'])]
        sql_values = [span.group(1) for span in re.finditer(sql_wrappers, ex['query'])]
        assert len(question_values) == len(sql_values)
        mappings = dict([(s, q) for q, s in zip(question_values, sql_values)])
        # mappings = dict([(q, s) for q, s in zip(question_values, sql_values)])
        # ex['question'] = re.sub(r'(["“”\'])(.*?)(["“”\'])', lambda match: match.group(1) + mappings[match.group(2)] + match.group(3), ex['question'])
        new_query = re.sub(r'("|\')(.*?)("|\')', lambda match: match.group(1) + mappings[match.group(2)] + match.group(3), ex['query'])
        if verbose:
            print('Question:', ex['question'])
            print('SQL:', ex['query'])
            print('SQL revised:', new_query)
        ex['query'] = new_query
        return True
    return False

def fix_good_restaurant(ex, verbose=True):
    qid = int(ex['question_id'].lstrip('qid'))
    if 4969 <= qid <= 5074 and '2.5' in ex['query'] and '2.5' not in ex['question']:
        # what is a "good" restaurant ? rating > 2.5
        new_question = ex['question'].replace('？', '（评分大于2.5分）？') if '？' in ex['question'] else ex['question'].rstrip() + '（评分大于2.5分）'
        if verbose:
            print('DB:', ex['db_id'])
            print('Question:', ex['question'])
            print('Question revised:', new_question)
            print('SQL:', ex['query'])
        ex['question'] = new_question
        return True
    return False

def fix_geo_major(ex, verbose=True):
    qid, flag = int(ex['question_id'].lstrip('qid')), False
    if 2840 <= qid <= 3349:
        if '150000' in ex['query'] and '150000' not in ex['question']:
            clues = r'主要城市|大城市|重要城市'
            new_question = re.sub(clues, lambda match_obj: match_obj.group(0) + '（人口大于150000）', ex['question'])
            if verbose:
                print('DB:', ex['db_id'])
                print('Question:', ex['question'])
                print('Question revised:', new_question)
                print('SQL:', ex['query'])
            ex['question'], flag = new_question, True
        if '750' in ex['query'] and '750' not in ex['question']:
            clues = r'主要河流|大河|主要湖泊'
            replacement = {'主要河流': '长度', '大河': '长度', '主要湖泊': '面积'}
            new_question = re.sub(clues, lambda match_obj: match_obj.group(0) + f'（{replacement[match_obj.group(0)]}大于750）', ex['question'])
            if verbose:
                print('DB:', ex['db_id'])
                print('Question:', ex['question'])
                print('Question revised:', new_question)
                print('SQL:', ex['query'])
            ex['question'], flag = new_question, True
    return flag

def add_wild_symbol_for_like(query):
    # add % for cmp_op LIKE
    if re.search(r'like\s+["\'](.*?)["\']', query, flags=re.I):
        matches = [m.group(1) for m in re.finditer(r'like\s+["\'](.*?)["\']', query, flags=re.I)]
        mappings = dict([(m, '%' + m.strip('%').strip() + '%') for m in matches])
        query = re.sub(r'(like\s+["\'])(.*?)(["\'])', lambda m: m.group(1) + mappings[m.group(2)] + m.group(3), query, flags=re.I)
    return query

def amend_examples_in_dataset(dataset, schemas, tables, choice='train', verbose=True):
    count = 0
    for ex in dataset:
        if ex['question'] in question_query_mappings:
            if verbose:
                print('DB:', ex['db_id'])
                print('Question:', ex['question'])
                print('SQL:', ex['query'])
                print('SQL revised:', question_query_mappings[ex['question']])
            ex['query'] = question_query_mappings[ex['question']]
        elif ex['question'] in question_query_replacement:
            if verbose:
                print('DB:', ex['db_id'])
                print('Question:', ex['question'])
                print('Question revised:', question_query_replacement[ex['question']][0])
                print('SQL:', ex['query'])
                print('SQL revised:', question_query_replacement[ex['question']][1])
            ex['query'] = question_query_replacement[ex['question']][1]
            ex['question'] = question_query_replacement[ex['question']][0]
        elif ex['query'] in query_mappings:
            if verbose:
                print('DB:', ex['db_id'])
                print('Question:', ex['question'])
                print('SQL:', ex['query'])
                print('SQL revised:', query_mappings[ex['query']])
            ex['query'] = query_mappings[ex['query']]
        elif ex['question'] in question_mappings:
            if verbose:
                print('DB:', ex['db_id'])
                print('Question:', ex['question'])
                print('Question revised:', question_mappings[ex['question']])
                print('SQL:', ex['query'])
            ex['question'] = question_mappings[ex['question']]
        elif fix_good_restaurant(ex, verbose): pass
        elif fix_geo_major(ex, verbose): pass
        elif fix_chinese_english_mismatch_values(ex, choice, verbose): pass
        else: count += 1
        ex['query'] = add_wild_symbol_for_like(ex['query'])
        # reparse the sql json, this is very important !!!
        db_id = ex['db_id']
        ex['sql'] = get_sql(Schema(schemas[db_id], tables[db_id]), ex['query'])
    print('Fix %d examples in the dataset' % (len(dataset) - count))
    return dataset

def fix_special_tables(tables):
    for db in tables:
        if db['db_id'] == 'scholar':
            db['table_names'] = ['venue', 'author', 'dataset', 'journal', 'key phrase', 'paper', 'cite', 'paper dataset', 'paper key phrase', 'writes']
            db['column_names'] = [[-1, '*'], [0, 'venue id'], [0, 'venue name'], [1, 'author id'], [1, 'author name'], [2, 'dataset id'], [2, 'dataset name'], [3, 'journal id'], [3, 'journal name'], [4, 'key phrase id'], [4, 'key phrase name'], [5, 'paper id'], [5, 'title'], [5, 'venue id'], [5, 'year'], [5, 'number citing'], [5, 'number cited by'], [5, 'journal id'], [6, 'citing paper id'], [6, 'cited paper id'], [7, 'paper id'], [7, 'dataset id'], [8, 'paper id'], [8, 'key phrase id'], [9, 'paper id'], [9, 'author id']]
        elif db['db_id'] == 'formula_1':
            db['table_names'] = ['circuits', 'races', 'drivers', 'status', 'seasons', 'constructors', 'constructor standings', 'results', 'driver standings', 'constructor results', 'qualifying', 'pitstops', 'laptimes']
            db['column_names'] = [[-1, '*'], [0, 'circuit id'], [0, 'circuit reference'], [0, 'name'], [0, 'location'], [0, 'country'], [0, 'latitude'], [0, 'longitude'], [0, 'altitude'], [0, 'url'], [1, 'race id'], [1, 'year'], [1, 'round'], [1, 'circuit id'], [1, 'name'], [1, 'date'], [1, 'time'], [1, 'url'], [2, 'driver id'], [2, 'driver reference'], [2, 'number'], [2, 'code'], [2, 'forename'], [2, 'surname'], [2, 'dob'], [2, 'nationality'], [2, 'url'], [3, 'status id'], [3, 'status'], [4, 'year'], [4, 'url'], [5, 'constructor id'], [5, 'constructor reference'], [5, 'name'], [5, 'nationality'], [5, 'url'], [6, 'constructor standings id'], [6, 'race id'], [6, 'constructor id'], [6, 'points'], [6, 'position'], [6, 'position text'], [6, 'wins'], [7, 'result id'], [7, 'race id'], [7, 'driver id'], [7, 'constructor id'], [7, 'number'], [7, 'grid'], [7, 'position'], [7, 'position text'], [7, 'position order'], [7, 'points'], [7, 'laps'], [7, 'time'], [7, 'milliseconds'], [7, 'fastest lap'], [7, 'rank'], [7, 'fastest lap time'], [7, 'fastest lap speed'], [7, 'status id'], [8, 'driver standings id'], [8, 'race id'], [8, 'driver id'], [8, 'points'], [8, 'position'], [8, 'position text'], [8, 'wins'], [9, 'constructor results id'], [9, 'race id'], [9, 'constructor id'], [9, 'points'], [9, 'status'], [10, 'qualify id'], [10, 'race id'], [10, 'driver id'], [10, 'constructor id'], [10, 'number'], [10, 'position'], [10, 'q1'], [10, 'q2'], [10, 'q3'], [11, 'race id'], [11, 'driver id'], [11, 'stop'], [11, 'lap'], [11, 'time'], [11, 'duration'], [11, 'milliseconds'], [12, 'race id'], [12, 'driver id'], [12, 'lap'], [12, 'position'], [12, 'time'], [12, 'milliseconds']]
        elif db['db_id'] == 'tracking_share_transactions':
            db['column_types'][13] = 'number'
        elif db['db_id'] == 'apartment_rentals':
            db['column_types'][14] = 'number'
        elif db['db_id'] == 'customers_and_addresses':
            db['column_types'][-1] = 'number'
        elif db['db_id'] == 'department_store':
            db['column_types'][50] = 'number'
        elif db['db_id'] == 'customers_and_products_contacts':
            db['column_types'][-1] = 'number'
    return tables


if __name__ == '__main__':

    data_dir, db_dir = DATASETS['cspider']['data'], DATASETS['cspider']['database']
    table_path = os.path.join(data_dir, 'tables.json')
    origin_table_path = os.path.join(data_dir, 'tables.original.json')
    update_table_path = origin_table_path if os.path.exists(origin_table_path) else table_path
    tables = fix_special_tables(json.load(open(update_table_path, 'r')))
    tables = amend_primary_keys(tables, verbose=False)
    tables = amend_foreign_keys(tables, verbose=False)
    tables = amend_boolean_types(tables, db_dir, verbose=False)
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
        dataset = amend_examples_in_dataset(dataset, schemas, tables, choice=data_split, verbose=False)
        json.dump(dataset, open(dataset_path, 'w'), indent=4, ensure_ascii=False)