#coding=utf8
import os, sys, json, re, shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.constants import DATASETS

train_mappings = {
    'qid003076': '在各出发机场的航班信息中，当机场的机位数量大于100时，给出航班信息的头等舱价格的平均值小于等于5000的那些机场的名称以及航班信息的机龄的最大值',
    'qid015679': '哪些公司赞助了5场峰会',
    'qid015724': '最低汽车价格总和不大于15000的公司及其最高价格是多少',
    'qid015725': '最低汽车价格不小于5万的公司及其最低价格总和',
    'qid015749': '高校平均独立科研机构数量至少200亿的国家及其最大重点学科数量',
    'qid018972': '哪些机场的机位多于一百个，且头等舱的平均价格不超过5千元，以及最大机龄是多少',
    'qid020775': '哪些菜系的饭店口味评分不超过3.5分，且平均人均价格最少100元，以及这些饭店服务评分总共多少',
    'qid017657': '公司item_product_10_83在中国建立的事业部中，在福建奔驰事业部投入资金比在北京奔驰事业部多投入了多少',
    'qid017660': '公司item_product_10_83在中国建立的事业部中，在福建奔驰事业部投入资金是在北京奔驰事业部投入的多少倍',
    'qid018007': '演员陈力比张译大几岁',
    'qid017944': '桂陵之战持续时间是长平之战的多少倍',
    'qid017673': '公司item_product_12_111在平台item_product_12_116和item_product_12_120上一共购买了多少个关键词',
    'qid017677': '公司item_product_12_111在平台item_product_12_120上的转化率比item_product_12_116上高了多少',
    'qid017707': '在第四季度，空调item_product_6_44型号售卖价格是item_product_6_41型号的多少倍',
    'qid017775': '在2016届，学校item_enterprise_10_102的本科毕业生是学校item_enterprise_10_101的多少倍',
    'qid017808': '宫保鸡丁在饭店item_task_dialogue_6_36的必点比例比饭店item_task_dialogue_6_39的高多少',
    'qid017857': '相亲软件item_software_9_109的银卡会员费是软件item_software_9_106银卡费用的多少倍',
    'qid017875': '针对市场营销专业，公司item_activity_14_123的基本工资比item_activity_14_121多多少钱',
    'qid018373': '第九届中国花卉园博会比第十一届多来了多少机构',
    'qid018374': '第九届中国花卉园博会开设的展园比第十一届多了多少个',
    'qid020641': '哪3个城市戏剧在2019年12月24号之前演出，总售票数最少',
    'qid013101': '当是在2019年12月24号之前演出时，售出的票数加起来最少的3个城市',
    'qid013116': '车展时间在2018年11月20号之后时，车展时间加起来最少的三个车展地点',
    'qid011771': '哪些出版社不是在上海成立的，且是在1990年6月23号之后成立的，也给出成立的地点',
    'qid008412': '成立时间超过1年，颁奖地点不是每届不同地点颁奖的是哪些华语电影奖',
    'qid017686': '快递公司item_enterprise_3_17每公斤收费标准在东北比在华北高多少',
    'qid017692': 'item_dynasties_words_10_93姓在中部地区的人口比例是南部地区的多少倍',
    'qid017579': '学校item_activity_14_130的应用数学专业的排名比软件工程专业高多少名',
    'qid017600': '作者item_book.2_6_58所写的贾平凹全集比其郭沫若全集多包含多少页',
    'qid008260': '年龄为74岁，出生地不为新疆乌鲁木齐的演员有哪些？',
    'qid011399': '哪些公司的哪些品牌的是在2012年1月11号之后成立的',
    'qid011434': '哪些期刊是在2000年3月24号之前创办的，以及期刊的语言是是什么',
    'qid012146': '开业时间2002年9月8号之后的酒店，按客房数量降序排列给出酒店的名称以及酒店地址',
    'qid017810': '在饭店item_task_dialogue_6_36，麻辣兔头每月销售量比拔丝苹果的多多少',
    'qid017851': '商家item_activity_13_118实体店分布的各个国家中，在韩国开设的实体店其年营业额占比是多少',
    'qid011854': '不在北京市东城区，且在2008年9月10号及之前开业的酒店有哪些，以及酒店的地址在哪里',
    'qid011737': '哪些演职员出生在1987年8月17号及之前，且不是出生在乌鲁木齐的，以及他们的职业是什么',
    'qid000053': '找出人数占比不到99%且熟识度不大于40%的奢侈品牌，以及品牌的国家个了解渠道',
    'qid003285': '在总假期不止19天或者带薪年假不少于20天的国家中，给出税率不少于5种的国家及其对应的最大年总税率',
    'qid013008': '当身高不止一米九一时，平均体重最小的国家',
    'qid001810': '2016届比人少于四千人，且2016届平均月薪不少于8千的学校有哪些，以及学校是什么类型和学历',
    'qid001770': '给出电视机型号的背光灯寿命不超过5万小时，且市场分不超过23%的电视品牌是哪些，以及属于哪个公司，电视机型号的产品定位是什么',
    'qid019045': '哪些基金公司注册资本多于21000万，且基金公司收入的总管理费共至少1万，以及总资产规模是多少',
    'qid008403': '成立年数不少于3年，或者每届都在不同地方举办的是哪些华语电影奖',
    'qid001842': '给出进口额不少于113亿，且城市GDP不超过8000亿的城市，以及属于哪个省，贸易产业属于什么行业',
    'qid002865': '找到GDP少于八千亿的城市及其所属省份，并给这些出城市金融产业的最大金融机构数',
    'qid003023': '在面积不超过25.8万平方米的2018年演唱会场馆中，给出拥有2018年演唱会不超过5场的体育场及所在城市',
    'qid007677': '在2004年后举办的世界经济论坛中，按着参与国家由少到多给出排序',
    'qid007679': '在2004年及其之后举办的世界经济论坛中，按着参与国家由多到少给出排序',
    'qid007680': '在2004年及其之后举办的世界经济论坛中，按着参与国家由少到多给出排序',
    'qid013653': '起始时间在618年及之后，并且人口少于8780万时，每个都城的最小国土面积是多少',
    'qid011841': '豆瓣评分不是9.3，一星占比不超过5%的书籍是哪些，以及作者是谁',
    'qid011225': '哪些平台入驻商家不超100000',
    'qid011692': '有哪些出境游路线给成人的价格不止10900，且门票超过了2500，是从哪个城市出发的',
    'qid011849': '恰好50岁，且作品不多于64个的诗人有哪些，以及性别是什么',
    "qid014293": "哪些城市出发的出境游路线成人价格18900元，超过25个景点，儿童价格总和不超100？",
    "qid013663": "年号不是乾隆，且子女不少于15个的皇帝中，每个朝代的这类皇帝平均有多少个妃子",
    'qid014632': "身高不低于1米91，或者体重不超过88千克时，哪些国籍的篮球运动员正好5个",
    'qid011391': "2020年八月十五之后还有哪些传统节日，以及是什么时候起源的",
    "qid007561": "给出距机场距离排名最高的5个火车站的名称或者距汽车站距离小于5km的火车站的名称"
}

dev_mappings = {
    'qid002204': '哪些综艺在各网站的总收视份额不超过0.3%？',
    'qid002150': '属性不止5个的高校有哪些？属于什么类别？',
    'qid002171': '哪些国家举办夏季奥运会时总共有不足100个大项项目？平均有几个国家参加？',
    'qid002300': '哪些球队出场过超过40次比赛且至少参加过5场？总篮板多少？',
    'qid002301': '哪些球队出场过不足40次且至少5次比赛？平均抢断多少？',
    'qid001979': '企业tem_enterprise_12_133在B轮比A轮多融资了多少金额',
    'qid001983': '参考试卷item_book.2_13_161在各省购买统计中，河北省售量是湖北省的多少倍',
    'qid001952': '品牌item_product_4_16在北京和昆明一共开设了多少家门店',
    'qid001942': '针对坚果item_animal_food_8_61，巴西生产的和东非生产的加起来市场占比是多少',
    'qid001253': '在场上的位置不是控球后卫，少于19岁的篮球运动员的中文名字，以及在场上的位置是什么',
    'qid000118': '给出售价不低于三千块，且市场份额不低于10.2%的洗衣机品牌，以及属于哪个所属公司，产品类别是什么',
    'qid000849': '成立年头超过了15年，公司的年营业额超过了2000万的公司是哪些',
    'qid000838': '成立时间不是12年，同时年营业额刚好是2000万的公司有哪些',
    'qid001054': '覆盖了400座城市的打车APP有哪些，给出它们所属的公司id 以及平均在每座城市服务多少用户',
    'qid002296': '人均摄入量大于0.05千克的坚果摄入量中，哪些国家的人均坚果摄入量数至少5？它们的世界人均摄入最少多少？'
}

def amend_examples_in_dataset(dataset: dict, choice: str = 'train', verbose: bool = False):
    count = 0
    if choice == 'train':
        for ex in dataset:
            if ex['question_id'] in train_mappings:
                count += 1
                ex['question'] = train_mappings[ex['question_id']]
            elif ex['question_id'] == 'qid017835':
                count += 1
                ex['question'] = '赛事item_software_1_12在平台item_software_1_1上的转播费是在平台item_software_1_2上的多少倍'
                ex['query'] = "select a.转播费 / b.转播费 from ( select 转播费 from 赛事转播 where 平台id == 'item_software_1_1' and 赛事id == 'item_software_1_12' ) a , ( select sum ( 转播费 ) from 赛事转播 where 平台id == 'item_software_1_2' and 赛事id == 'item_software_1_12' ) b"
                ex['sql']['from']['table_units'][1][1]['where'][0][3] = 'item_software_1_2'
                ex['sql']['from']['table_units'][1][1]['where'].append('and')
                ex['sql']['from']['table_units'][1][1]['where'].append([0, 2, [0, [0, 17, False], None], 'item_software_1_12', None])
            elif ex['question_id'] == 'qid014704':
                count += 1
                ex['query'] = "select 所属国家 from 代言明星 where 年龄 >= 40 or 性别 == '男' group by 所属国家 having count ( * ) < 5"
                ex['sql']['where'][2][3] = "男"
            elif ex['question_id'] == 'qid017858':
                count += 1
                ex['query'] = "select a.会费 / b.会费 from ( select 会费 from 相亲软件会费 where 软件id == 'item_software_9_106' and 会员类型 == '金卡' ) a , ( select 会费 from 相亲软件会费 where 软件id == 'item_software_9_106' and 会员类型 == '铜卡' ) b"
                ex['sql']['from']['table_units'][0][1]['where'][2][3] = '金卡'
                ex['sql']['from']['table_units'][1][1]['where'][2][3] = '铜卡'
            elif ex['question_id'] == 'qid003329':
                count += 1
                ex['query'] = "select T2.名称 , avg ( T1.2017年出货量 ) from 智能手机全球出货量 as T1 join 智能手机公司 as T2 on 智能手机全球出货量.公司id == 智能手机公司.词条id where T2.年营业额 <= 200000000000 or T2.年利润 >= 10000000000 group by T1.公司id having count ( * ) > 5"
                ex['sql']['where'][2][3] = 10000000000
            elif ex['question_id'] == 'qid013551':
                count += 1
                ex['query'] = "select min ( 获奖次数 ) , 导演 from 电影作品 where 提名次数 >= 1 group by 导演"
                ex['sql']['where'][0][3] = 1
            elif ex['question_id'] == 'qid005872':
                count += 1
                ex['query'] = "select T2.届数 from 春晚嘉宾 as T1 join 央视春节晚会 as T2 on 春晚嘉宾.春晚id == 央视春节晚会.词条id where T1.是否获奖 == '是' group by T1.春晚id order by count ( * ) asc limit 3"
                ex['sql']['limit'] = 3
            elif ex['question_id'] == 'qid020128':
                count += 1
                ex['query'] = "select T2.届数 from 春晚嘉宾 as T1 join 央视春节晚会 as T2 on 春晚嘉宾.春晚id == 央视春节晚会.词条id where T1.是否获奖 == '是' group by T1.春晚id order by count ( * ) asc limit 3"
                ex['sql']['limit'] = 3
            if '每届不同' in ex['query']: # for better recognition
                neq_prompt, eq_prompt = ['不同', '不一样'], ['同一个', '同一', '一样', '一个']
                for p in neq_prompt:
                    if p in ex['question']:
                        count += 1
                        ex['query'] = re.sub(r"每届不同", p, ex['query'])
                        ex['sql']['where'][-1][3] = p
                        break
                else:
                    for p in eq_prompt:
                        if p in ex['question']:
                            count += 1
                            ex['query'] = re.sub(r"!= '每届不同'", "== '{}'".format(p), ex['query'])
                            ex['sql']['where'][-1][1] = 2
                            ex['sql']['where'][-1][3] = p
                            break
    else:
        for ex in dataset:
            if ex['question_id'] in dev_mappings:
                count += 1
                ex['question'] = dev_mappings[ex['question_id']]
            elif ex['question_id'] == 'qid000817':
                count += 1
                ex['query'] = 'select app名称 from 打车APP where TIME_NOW - 上线时间 > 8 or 覆盖城市数 >= 100'
                ex['sql']['where'][0][3] = 8
            elif ex['question_id'] == 'qid001234':
                count += 1
                ex['query'] = "select 姓名 , 民族 from 明星 where 职业 != '演员'"
                ex['sql']['where'][-1][2][1][1] = 17
            elif ex['question_id'] == 'qid002228':
                count += 1
                ex['query'] = "select T2.节目名称 from 收视率 as T1 join 综艺节目 as T2 on 收视率.节目id == 综艺节目.词条id where T2.系列名 == '欢乐喜剧人' group by T1.节目id order by count ( * ) desc limit 3"
                ex['sql']['limit'] = 3
            elif ex['question_id'] == 'qid000591':
                count += 1
                ex['query'] = "select T2.节目名称 from 收视率 as T1 join 综艺节目 as T2 on 收视率.节目id == 综艺节目.词条id where T2.系列名 == '欢乐喜剧人' group by T1.节目id order by count ( * ) desc limit 3"
                ex['sql']['limit'] = 3
            elif ex['question_id'] == 'qid001512':
                count += 1
                ex['query'] = "select 所属公司 , avg ( 服务用户数量 ) from 打车APP where 上线时间 < 2014 or 覆盖城市数 <= 100 group by 所属公司"
            elif ex['question_id'] == 'qid000540':
                count += 1
                ex['question'] = '在企业融资的融资总额最多时，给出对应的企业的中文名以及企业融资的融资轮次'
                ex['query'] = 'select T2.中文名 , T1.融资轮次 from 企业融资 as T1 join 企业 as T2 on 企业融资.企业id == 企业.词条id order by T1.融资总额 desc'
                ex['sql']['select'] = ex['sql']['select'][1:]
                ex['sql']['from']['table_units'] = ex['sql']['from']['table_units'][:-1]
                ex['sql']['from']['conds'] = ex['sql']['from']['conds'][0:1]
            elif ex['question_id'] == 'qid000541':
                count += 1
                ex['question'] = '在企业融资的融资总额最少时，给出对应的企业的中文名以及企业融资的融资轮次'
                ex['query'] = 'select T2.中文名 , T1.融资轮次 from 企业融资 as T1 join 企业 as T2 on 企业融资.企业id == 企业.词条id order by T1.融资总额 asc'
                ex['sql']['select'] = ex['sql']['select'][1:]
                ex['sql']['from']['table_units'] = ex['sql']['from']['table_units'][:-1]
                ex['sql']['from']['conds'] = ex['sql']['from']['conds'][0:1]
            elif ex['question_id'] == 'qid000550':
                count += 1
                ex['question'] = '在企业融资的融资总额最少时，给出排名前3对应的企业的中文名以及企业融资的融资轮次'
                ex['query'] = 'select T2.中文名 , T1.融资轮次 from 企业融资 as T1 join 企业 as T2 on 企业融资.企业id == 企业.词条id order by T1.融资总额 asc limit 3'
                ex['sql']['select'] = ex['sql']['select'][1:]
                ex['sql']['from']['table_units'] = ex['sql']['from']['table_units'][:-1]
                ex['sql']['from']['conds'] = ex['sql']['from']['conds'][0:1]
            elif ex['question_id'] == 'qid000551':
                count += 1
                ex['question'] = '在企业融资的融资总额最多时，给出排名前3对应的企业的中文名以及企业融资的融资轮次'
                ex['query'] = 'select T2.中文名 , T1.融资轮次 from 企业融资 as T1 join 企业 as T2 on 企业融资.企业id == 企业.词条id order by T1.融资总额 desc limit 3'
                ex['sql']['select'] = ex['sql']['select'][1:]
                ex['sql']['from']['table_units'] = ex['sql']['from']['table_units'][:-1]
                ex['sql']['from']['conds'] = ex['sql']['from']['conds'][0:1]
    print('Fix %d examples in the %s dataset' % (count, choice))
    return dataset

def fix_tables(tables_list):
    tables = []
    for db in tables_list:
        if db['db_id'] not in tables:
            tables.append(db)
            if 'table_names_original' not in db or not db['table_names_original']:
                db['table_names_original'] = db['table_names']
            if 'column_names_original' not in db or not db['column_names_original']:
                db['column_names_original'] = db['column_names']
        if db['db_id'] == '智能手机全球占比':
            db['column_types'][-3] = 'text' # 部署国家
        elif db['db_id'] == '互联网企业':
            db['column_types'][6] = 'text' # 首席执行官
        elif db['db_id'] == '世界湖泊':
            db['column_types'][7] = 'text' # 属性
            db['column_types'][11] = 'text' # 接壤方式
        elif db['db_id'] == '酒店预订':
            db['column_types'][-3] = 'text' # 早餐
        elif db['db_id'] == '空调':
            db['column_types'][23] == 'number' # 季度
        elif db['db_id'] == '智能音箱':
            db['column_types'][13] == 'number' # 季度
        elif db['db_id'] == '中国文学奖':
            db['column_types'][13] == 'number' # 字数
    return tables

if __name__ == '__main__':

    data_dir = DATASETS['dusql']['data']
    table_path = os.path.join(data_dir, 'tables.json')
    origin_table_path = os.path.join(data_dir, 'tables.original.json')
    update_table_path = origin_table_path if os.path.exists(origin_table_path) else table_path
    tables = fix_tables(json.load(open(update_table_path, 'r')))
    if not os.path.exists(origin_table_path):
        shutil.copyfile(table_path, origin_table_path)
    json.dump(tables, open(table_path, 'w'), indent=4)

    for data_split in ['train', 'dev']:
        dataset_path = os.path.join(data_dir, data_split + '.json')
        origin_dataset_path = os.path.join(data_dir, data_split + '.original.json')
        if os.path.exists(origin_dataset_path):
            dataset = json.load(open(origin_dataset_path, 'r'))
        else:
            dataset = json.load(open(dataset_path, 'r'))
            shutil.copyfile(dataset_path, origin_dataset_path)
        dataset = amend_examples_in_dataset(dataset, data_split, verbose=True)
        json.dump(dataset, open(dataset_path, 'w'), indent=4, ensure_ascii=False)