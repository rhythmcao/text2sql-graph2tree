#coding=utf8
import os, sys, json, re, shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.constants import DATASETS

train_mappings = {
    'qid008023': '哪些皇帝在世时间不超过30年，或者不是“十全武功”的开创者',
    'qid012799': '预售价格不小于6500并且参团价格大于5000，景点数升序排列给出前3对应的国内游路线的路线名称以及出发城市',
    'qid014283': '起始时间是618年之前并且人口数量不是8600万的，朝代国土面积的平均值小于1000000的都城有哪些？',
    'qid014418': '哪些城市2014年及之前不超过100个员工的海外研究中心心数量大于5的那些所在城市',
    'qid017878': '在公司item_activity_14_121中，“金融会计专业”比“市场营销”专业多招聘多少人',
    'qid022422': '哪5个省的城市平均有最少三甲医院，以及最多有多少人',
    'qid016718': '拥有举重世界纪录的数量排名后3的项目类型，举重世界记录的记录成绩的平均值',
    'qid007561': '给出距机场距离排名最高的3个火车站的名称或者距汽车站距离小于5km的火车站的名称',
    'qid007118': '中国最早举办的3届国际进口博览会中，哪些博览会的参加企业数排名在倒数前5',
    'qid000208': '在奢侈品牌了解渠道的分布中，奢侈品渠道的奢侈品渠道的人数占比总和排名后3时给出奢侈品牌的名称和奢侈品牌的国家',
    'qid000646': '专辑数量小于10或歌曲数量不等于1000时，按粉丝总数升序排列给出前3的歌手的出生地及姓名',
    'qid003076': '在各出发机场的航班信息中，当机场的机位数量大于100时，给出航班信息的头等舱价格的平均值小于等于5000的那些机场的名称以及航班信息的机龄的最大值',
    'qid015679': '哪些公司赞助了5场峰会',
    'qid015724': '最低汽车价格总和不大于15000的公司及其最高价格是多少',
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
    'qid012364': '成人价格不小于1万8千9百的出境游路线按景点数升序给出前10对应的路线以及出发城市',
    "qid013663": "年号不是乾隆，且子女不少于15个的皇帝中，每个朝代的这类皇帝平均有多少个妃子",
    'qid014632': "身高不低于1米91，或者体重不超过88千克时，哪些国籍的篮球运动员正好5个",
    'qid011391': "2020年八月十五之后还有哪些传统节日，以及是什么时候起源的",
    'qid007374': "给出岗位不超过987个的省份，但是不包含合格人数最多的3个省",
    'qid007187': "起送价格最低的3个或者送达用时最多的5个外卖商家有哪些",
    'qid012363': '成人价格不等于1万8千九百的出境游路线按景点数升序排列给出前10的路线名称以及出发城市'
}

dev_mappings = {
    'qid000791': '给出GDP总计(亿)排名最后的3个省份或者财政预算同比增速小于-16%的各省财政收入的省份',
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
    'qid001054': '覆盖了400座城市的打车APP有哪些，给出它们所属的公司id以及平均在每座城市服务多少用户',
    'qid002296': '人均摄入量大于0.05千克的坚果摄入量中，哪些国家的人均坚果摄入量数至少5？它们的世界人均摄入最少多少？',
    'qid001310': '成立时间在15年之后的企业按注册资本逆序排列给出前10的企业中文名以及法定代表人',
    'qid001311': '成立时间在15年及其之后的企业按注册资本逆序排列给出前10的企业中文名以及法定代表人',
    'qid000896': '成立超两年注册资本不等于100万的企业都有什么？'
}

dev_limit_fix = {
    'qid000463': 5,
    'qid001329': 10,
    'qid001334': 10,
    'qid001336': 10,
    'qid001341': 10,
    'qid001843': 10,
    'qid001847': 10,
    'qid001853': 10,
    'qid001855': 10,
    'qid001856': 10
}

def revise_limit_number(ex):
    if ex['question_id'] in dev_limit_fix:
        ex['sql']['limit'] = dev_limit_fix[ex['question_id']]
        return True
    return False

train_limit_fix = {
    'qid020619': 'select 导演 from 行星相关电影 where 豆瓣评分 < 8.5 group by 导演 order by count ( * ) desc limit 1',
    'qid020595': 'select 主要成就 from 皇帝 where 出生时间 < 1456 group by 主要成就 order by count ( * ) desc limit 1',
    'qid012995': "select 国籍 from 名人 where 主要成就 != '苹果公司联合创办人' group by 国籍 order by count ( * ) desc limit 1",
    'qid007490': "( select 名称 from 影院 order by 与当前距离 asc limit 1 ) union ( select 名称 from 影院 where 用户评分 <= 4.7 )",
    'qid002357': "select T1.姓名 , T2.部门名称 , T2.职责 from 员工 as T1 join 部门 as T2 on 员工.所属部门id == 部门.词条id where T1.薪资 > 20000 order by T1.年龄 desc limit 1",
    'qid002329': "select T1.名称 , T2.名称 , T2.所属省份 from 医院 as T1 join 城市 as T2 on 医院.所属城市id == 城市.词条id where T1.职工数量 <= 1270 order by T1.重点专科数量 desc limit 1",
    'qid000090': "select T1.交通枢纽站 , T2.城市 , T2.城市面积 from 全国交通枢纽 as T1 join 城市 as T2 on 全国交通枢纽.所属城市id == 城市.词条id where T1.平均拥堵指数 >= 1.7 order by T1.周边路网平均速度(千米/时) desc limit 1"
}

def limit_one_error(ex):
    if ex['question_id'] in train_limit_fix:
        ex['query'] = train_limit_fix[ex['question_id']]
        ex['sql']['limit'] = 1
        return True
    return False

def incorrect_value(ex):
    if ex['question_id'] in ['qid015955', 'qid015957', 'qid015958', 'qid015960', 'qid021734', 'qid021736', 'qid021737', 'qid021739']:
        ex['query'] = ex['query'].replace('5', '50000')
        ex['sql']['having'][0][3] = 50000
    elif ex['question_id'] == 'qid015725':
        ex['question'] = '最低汽车价格不小于5万的公司及其最低价格总和'
        ex['query'] = ex['query'].replace('5', '50000')
        ex['sql']['having'][0][3] = 50000
    elif ex['question_id'] == 'qid016155':
        ex['query'] = ex['query'].replace('3000', '30000000')
        ex['sql']['having'][0][3] = 30000000
    else: return False
    return True

def amend_examples_in_dataset(dataset: dict, choice: str = 'train', verbose: bool = False):
    count = 0
    if choice == 'train':
        for ex in dataset:
            if ex['question_id'] in train_mappings: ex['question'] = train_mappings[ex['question_id']]
            elif incorrect_value(ex): pass
            elif limit_one_error(ex): pass
            elif ex['question_id'] in ['qid002671', 'qid002672', 'qid002673']:
                ex['question'] = ex['question'].replace('1亿万公里', '一亿公里')
            elif ex['question_id'] in ['qid018725', 'qid018726', 'qid018727', 'qid018728']:
                ex['question'] = ex['question'].replace('10公里', '10万')
                ex['query'] = ex['query'].replace('100000000', '100000')
                ex['sql']['where'][0][3] = 100000
            elif ex['db_id'] == '银行理财产品' and '万亿' in ex['question'] and '3000000000000' in ex['query']:
                ex['query'] = ex['query'].replace('3000000000000', '300000000000000')
                for cond in ex['sql']['where']:
                    if cond in ['and', 'or']: continue
                    if str(cond[3]) == '3000000000000':
                        cond[3] = 300000000000000
                        break
            elif ex['question_id'] == 'qid012157':
                ex['query'] = "select 姓名 , 性别 from 科学家 where 职业 != '小说家' order by 出生日期 asc"
                ex['sql']['where'][0][2][1][1] = 26
            elif ex['question_id'] == 'qid000710':
                ex['sql']['having'][0][1] = 5 
            elif ex['question_id'] == 'qid008111':
                ex['sql']['where'][0][1] = 5
                ex['sql']['where'][2][1] = 4
            elif ex['question_id'] == 'qid008323':
                ex['sql']['where'][0][1] = 3 
            elif ex['question_id'] == 'qid014669':
                ex['sql']['having'][0][1] = 4
            elif ex['question_id'] == 'qid010417':
                ex['query'] = "select 名称 , 类型 from 列车 where 名称 like 'Z' or 出发时间 > 13:30:00"
            elif ex['question_id'] == 'qid013475':
                ex['sql']['where'][0][1] = 3
            elif ex['question_id'] == 'qid013478':
                ex['sql']['where'][0][1] = 4
            elif ex['question_id'] == 'qid013480':
                ex['sql']['where'][0][1] = 3
            elif ex['question_id'] == 'qid013511':
                ex['sql']['where'][0][1] = 4
            elif ex['question_id'] == 'qid013516':
                ex['sql']['where'][0][1] = 4
            elif ex['question_id'] == 'qid013577':
                ex['sql']['where'][0][1] = 4
            elif ex['question_id'] == 'qid014411':
                ex['sql']['where'][0][1] = 5
            elif ex['question_id'] == 'qid014262':
                ex['sql']['having'][0][1] = 4
            elif ex['question_id'] == 'qid013740':
                ex['sql']['where'][0][1] = 3
                ex['sql']['where'][2][1] = 2
            elif ex['question_id'] == 'qid018736':
                ex['query'] = 'select T2.名称 , sum ( T1.人数 ) , T2.所属国家 from 高校获奖名单 as T1 join 高校 as T2 on 高校获奖名单.高校id == 高校.词条id where T2.独立科研机构数量 > 0 group by T1.高校id'
                ex['sql']['where'][0][1] = 3
            elif ex['question_id'] == 'qid018734':
                ex['query'] = 'select T2.名称 , avg ( T1.人数 ) , T2.所属国家 from 高校获奖名单 as T1 join 高校 as T2 on 高校获奖名单.高校id == 高校.词条id where T2.独立科研机构数量 > 0 group by T1.高校id'
                ex['sql']['where'][0][1] = 3
            elif ex['question_id'] == 'qid018733':
                ex['query'] = 'select T2.名称 , max ( T1.人数 ) , T2.所属国家 from 高校获奖名单 as T1 join 高校 as T2 on 高校获奖名单.高校id == 高校.词条id where T2.独立科研机构数量 > 0 group by T1.高校id'
                ex['sql']['where'][0][1] = 3
            elif ex['question_id'] == 'qid000505':
                ex['query'] = "( select 景点名称 from 旅游景点 where 平均拥堵指数 <= 3.2 ) except ( select 景点名称 from 旅游景点 order by 周边路网平均速度(千米/时) desc limit 1 )"
                ex['sql']['except']['limit'] = 1
            elif ex['question_id'] == 'qid003288':
                ex['query'] = 'select T2.名称 , max ( T1.人数 ) from 高校获奖名单 as T1 join 高校 as T2 on 高校获奖名单.高校id == 高校.词条id where T2.独立科研机构数量 == 0 or T2.重点学科数量 >= 50 group by T1.高校id having count ( * ) > 5'
                ex['sql']['where'][0][1] = 2
            elif ex['question_id'] == 'qid000334':
                ex['query'] = "select 获奖建筑名称 from 阿卡汗建筑奖获奖名单 where 位于城市 not in ( select 位于城市 from 阿卡汗建筑奖获奖名单 group by 位于城市 order by count ( * ) asc limit 1 )"
                ex['sql']['where'][0][3]['limit'] = 1
            elif ex['question_id'] == 'qid000348':
                ex['query'] = 'select 名称 from 酒店集团 where 总部所在省 not in ( select 总部所在省 from 酒店集团 group by 总部所在省 order by sum ( 酒店数量 ) asc limit 1 )'
                ex['sql']['where'][0][3]['limit'] = 1
            elif ex['question_id'] == 'qid006327':
                ex['query'] = 'select 名称 from 外文书籍 where 原著作者 in ( select 原著作者 from 外文书籍 group by 原著作者 order by count ( * ) desc limit 1 )'
                ex['sql']['where'][0][3]['limit'] = 1
            elif ex['question_id'] == 'qid006354':
                ex['query'] = 'select 名称 from 省份 where 南北区域 in ( select 南北区域 from 省份 group by 南北区域 order by count ( * ) desc limit 1 )'
                ex['sql']['where'][0][3]['limit'] = 1
            elif ex['question_id'] == 'qid017835':
                ex['question'] = '赛事item_software_1_12在平台item_software_1_1上的转播费是在平台item_software_1_2上的多少倍'
                ex['query'] = "select a.转播费 / b.转播费 from ( select 转播费 from 赛事转播 where 平台id == 'item_software_1_1' and 赛事id == 'item_software_1_12' ) a , ( select sum ( 转播费 ) from 赛事转播 where 平台id == 'item_software_1_2' and 赛事id == 'item_software_1_12' ) b"
                ex['sql']['from']['table_units'][1][1]['where'][0][3] = 'item_software_1_2'
                ex['sql']['from']['table_units'][1][1]['where'].append('and')
                ex['sql']['from']['table_units'][1][1]['where'].append([0, 2, [0, [0, 17, False], None], 'item_software_1_12', None])
            elif ex['question_id'] == 'qid014704':
                ex['query'] = "select 所属国家 from 代言明星 where 年龄 >= 40 or 性别 == '男' group by 所属国家 having count ( * ) < 5"
                ex['sql']['where'][2][1] = 2
                ex['sql']['where'][2][3] = "男"
            elif ex['question_id'] == 'qid017858':
                ex['query'] = "select a.会费 / b.会费 from ( select 会费 from 相亲软件会费 where 软件id == 'item_software_9_106' and 会员类型 == '金卡' ) a , ( select 会费 from 相亲软件会费 where 软件id == 'item_software_9_106' and 会员类型 == '铜卡' ) b"
                ex['sql']['from']['table_units'][0][1]['where'][2][3] = '金卡'
                ex['sql']['from']['table_units'][1][1]['where'][2][3] = '铜卡'
            elif ex['question_id'] == 'qid003329':
                ex['query'] = "select T2.名称 , avg ( T1.2017年出货量 ) from 智能手机全球出货量 as T1 join 智能手机公司 as T2 on 智能手机全球出货量.公司id == 智能手机公司.词条id where T2.年营业额 <= 200000000000 or T2.年利润 >= 10000000000 group by T1.公司id having count ( * ) > 5"
                ex['sql']['where'][2][3] = 10000000000
            elif ex['question_id'] == 'qid013551':
                ex['query'] = "select min ( 获奖次数 ) , 导演 from 电影作品 where 提名次数 >= 1 group by 导演"
                ex['sql']['where'][0][3] = 1
            elif ex['question_id'] == 'qid005872':
                ex['query'] = "select T2.届数 from 春晚嘉宾 as T1 join 央视春节晚会 as T2 on 春晚嘉宾.春晚id == 央视春节晚会.词条id where T1.是否获奖 == '是' group by T1.春晚id order by count ( * ) asc limit 3"
                ex['sql']['limit'] = 3
            elif ex['question_id'] == 'qid020128':
                ex['query'] = "select T2.届数 from 春晚嘉宾 as T1 join 央视春节晚会 as T2 on 春晚嘉宾.春晚id == 央视春节晚会.词条id where T1.是否获奖 == '是' group by T1.春晚id order by count ( * ) asc limit 3"
                ex['sql']['limit'] = 3
            elif ex['question_id'] == 'qid007427':
                ex['query'] = '( select 名称 from 智能手机公司 where 年营业额 >= 1000000 ) except ( select 名称 from 智能手机公司 order by 年利润 desc limit 3 )'
                ex['sql']['where'][0][3] = 1000000
            elif '每届不同' in ex['query']: # for better value recognition
                if ex['question_id'] == 'qid008403':
                    ex['question'] = '成立年数不少于3年，或者每届都在不同地方举办的是哪些华语电影奖'
                elif ex['question_id'] == 'qid008412':
                    ex['question'] = '成立时间超过1年，颁奖地点每届相同的是哪些华语电影奖'
                elif ex['question_id'] == 'qid008414':
                    ex['question'] = '成立时间不到19年，或者颁奖地点每次不同的是哪些华语电影奖'
                elif ex['question_id'] == 'qid008416':
                    ex['question'] = '哪些华语电影奖的成立时间不少于13年，且颁奖地点不是每届不同'
                elif ex['question_id'] == 'qid008419':
                    ex['question'] = '设立已经不止8年了，或者颁奖地每届不同的是哪些华语电影奖'
                elif ex['question_id'] == 'qid008405':
                    ex['question'] = '成立时间不是7年，且颁奖地每次不同的是哪些华语电影奖'
                elif ex['question_id'] == 'qid008408':
                    ex['question'] = '成立时间是6年，且颁奖地点每届不同的华语电影奖是哪些'
                elif ex['question_id'] == 'qid011348':
                    ex['question'] = '哪些话语电影奖每届都在相同地方颁奖，以及颁奖地点在哪'
                neq_prompt, eq_prompt = ['不同地方', '不同地点', '每次不同'], ['同一地点', '每届一样', '每届相同', '相同地方']
                for p in neq_prompt:
                    if p in ex['question']:
                        ex['query'] = re.sub(r"每届不同", p, ex['query'])
                        ex['sql']['where'][-1][3] = p
                        break
                else:
                    for p in eq_prompt:
                        if p in ex['question']:
                            ex['query'] = re.sub(r"!= '每届不同'", "== '{}'".format(p), ex['query'])
                            ex['sql']['where'][-1][1] = 2
                            ex['sql']['where'][-1][3] = p
                            break
            else: count += 1
    else:
        for ex in dataset:
            if ex['question_id'] in dev_mappings: ex['question'] = dev_mappings[ex['question_id']]
            elif revise_limit_number(ex): pass
            elif ex['question_id'] == 'qid002425':
                ex['query'] = 'select 营养成分 from 每100克坚果营养成分 group by 营养成分 order by sum ( 含量 ) asc limit 1'
                ex['sql']['limit'] = 1
            elif ex['question_id'] == 'qid000817':
                ex['query'] = 'select app名称 from 打车APP where TIME_NOW - 上线时间 > 8 or 覆盖城市数 >= 100'
                ex['sql']['where'][0][3] = 8
            elif ex['question_id'] == 'qid001234':
                ex['query'] = "select 姓名 , 民族 from 明星 where 职业 != '演员'"
                ex['sql']['where'][-1][2][1][1] = 17
            elif ex['question_id'] == 'qid002228':
                ex['query'] = "select T2.节目名称 from 收视率 as T1 join 综艺节目 as T2 on 收视率.节目id == 综艺节目.词条id where T2.系列名 == '欢乐喜剧人' group by T1.节目id order by count ( * ) desc limit 3"
                ex['sql']['limit'] = 3
            elif ex['question_id'] == 'qid000591':
                ex['query'] = "select T2.节目名称 from 收视率 as T1 join 综艺节目 as T2 on 收视率.节目id == 综艺节目.词条id where T2.系列名 == '欢乐喜剧人' group by T1.节目id order by count ( * ) desc limit 3"
                ex['sql']['limit'] = 3
            elif ex['question_id'] == 'qid001512':
                ex['query'] = "select 所属公司 , avg ( 服务用户数量 ) from 打车APP where 上线时间 < 2014 or 覆盖城市数 <= 100 group by 所属公司"
            elif ex['question_id'] == 'qid000540':
                ex['question'] = '在企业融资的融资总额最多时，给出对应的企业的中文名以及企业融资的融资轮次'
                ex['query'] = 'select T2.中文名 , T1.融资轮次 from 企业融资 as T1 join 企业 as T2 on 企业融资.企业id == 企业.词条id order by T1.融资总额 desc'
                ex['sql']['select'] = ex['sql']['select'][1:]
                ex['sql']['from']['table_units'] = ex['sql']['from']['table_units'][:-1]
                ex['sql']['from']['conds'] = ex['sql']['from']['conds'][0:1]
            elif ex['question_id'] == 'qid000541':
                ex['question'] = '在企业融资的融资总额最少时，给出对应的企业的中文名以及企业融资的融资轮次'
                ex['query'] = 'select T2.中文名 , T1.融资轮次 from 企业融资 as T1 join 企业 as T2 on 企业融资.企业id == 企业.词条id order by T1.融资总额 asc'
                ex['sql']['select'] = ex['sql']['select'][1:]
                ex['sql']['from']['table_units'] = ex['sql']['from']['table_units'][:-1]
                ex['sql']['from']['conds'] = ex['sql']['from']['conds'][0:1]
            elif ex['question_id'] == 'qid000550':
                ex['question'] = '在企业融资的融资总额最少时，给出排名前3对应的企业的中文名以及企业融资的融资轮次'
                ex['query'] = 'select T2.中文名 , T1.融资轮次 from 企业融资 as T1 join 企业 as T2 on 企业融资.企业id == 企业.词条id order by T1.融资总额 asc limit 3'
                ex['sql']['select'] = ex['sql']['select'][1:]
                ex['sql']['from']['table_units'] = ex['sql']['from']['table_units'][:-1]
                ex['sql']['from']['conds'] = ex['sql']['from']['conds'][0:1]
            elif ex['question_id'] == 'qid000551':
                ex['question'] = '在企业融资的融资总额最多时，给出排名前3对应的企业的中文名以及企业融资的融资轮次'
                ex['query'] = 'select T2.中文名 , T1.融资轮次 from 企业融资 as T1 join 企业 as T2 on 企业融资.企业id == 企业.词条id order by T1.融资总额 desc limit 3'
                ex['sql']['select'] = ex['sql']['select'][1:]
                ex['sql']['from']['table_units'] = ex['sql']['from']['table_units'][:-1]
                ex['sql']['from']['conds'] = ex['sql']['from']['conds'][0:1]
            else: count += 1
    print('Fix %d examples in the %s dataset' % (len(dataset) - count, choice))
    return dataset

def amend_tables(tables_list):
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
            db['column_types'][23] = 'number' # 季度
        elif db['db_id'] == '智能音箱':
            db['column_types'][13] = 'number' # 季度
        elif db['db_id'] == '中国文学奖':
            db['column_types'][10] = 'text' # 作者
            db['column_types'][13] = 'number' # 字数
        elif db['db_id'] == '中国菜系':
            db['column_types'][6] = 'binary' # 是否是四大菜系
            db['column_types'][7] = 'binary' # 是否是八大菜系
        elif db['db_id'] == '教材辅助参考书':
            db['column_types'][10] = 'number' # 套数
        elif db['db_id'] == '澳网公开赛':
            db['column_types'][9] = 'time' # 夺冠年份
        elif db['db_id'] == '植物经济价值':
            db['column_types'][17] = 'text' # 地区
        elif db['db_id'] == '笔记本电脑':
            db['column_types'][25] = 'text' # 最好评价
        elif db['db_id'] == '诺贝尔奖项':
            db['column_types'][18] = 'text' # 国家
        elif db['db_id'] == '中国高校':
            db['column_types'][10] = 'text' # 职业
        elif db['db_id'] == '大洲与国家':
            db['column_types'][32] = 'text' # 源大洲名称
            db['column_types'][33] = 'text' # 目标大洲名称
        elif db['db_id'] == '各城市院士情况':
            db['column_types'][2] = 'text' # 省份名称
        elif db['db_id'] == '城市拥堵':
            db['column_types'][13] = 'text' # 所属省份
        elif db['db_id'] == '医院':
            db['column_types'][20] = 'text' # 科室
        elif db['db_id'] == '地震':
            db['column_types'][24] = 'text' # 地点
        elif db['db_id'] == '中国演员和电影':
            db['column_types'][14] = 'text' # 导演
        elif db['db_id'] == '中国城市潜力':
            db['column_types'][18] = 'text' # 名称
        elif db['db_id'] == '中国宜居城市': # 空气指数, 蓝天数量
            db['column_types'][4] = 'text'
            db['column_types'][5] = 'text'
            db['column_types'][9] = 'text'
            db['column_types'][10] = 'text'
            db['column_types'][14] = 'text'
            db['column_types'][15] = 'text'
    return tables

if __name__ == '__main__':

    data_dir = DATASETS['dusql']['data']
    table_path = os.path.join(data_dir, 'tables.json')
    origin_table_path = os.path.join(data_dir, 'tables.original.json')
    update_table_path = origin_table_path if os.path.exists(origin_table_path) else table_path
    tables = amend_tables(json.load(open(update_table_path, 'r')))
    if not os.path.exists(origin_table_path):
        shutil.copyfile(table_path, origin_table_path)
    json.dump(tables, open(table_path, 'w'), indent=4, ensure_ascii=False)

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