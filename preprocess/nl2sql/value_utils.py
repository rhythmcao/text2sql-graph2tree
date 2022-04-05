#coding=utf8
import re, os, sys, copy, math
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import editdistance as edt
from LAC import LAC
from decimal import Decimal
from itertools import combinations
from asdl.transition_system import SelectValueAction
from preprocess.nl2sql.input_utils import NORM
from preprocess.process_utils import ValueCandidate, State, SQLValue
from preprocess.process_utils import is_number, is_int, search_for_longest_substring
from preprocess.process_utils import ZH_NUM2WORD, ZH_WORD2NUM, ZH_NUMBER, ZH_UNIT, DIGIT_ALIAS

AGG_OP = ('none', 'avg', 'max', 'min', 'count', 'sum')
CMP_OP = ('>', '<', '==', '!=')
PLACEHOLDER = '|' # | never appears in the dataset

ABBREV_SET = [
    set({'重点项目', '重点研究', '重点工程', '重点课题', '重点建设项目'}), set({'自取', '自行拿取', '自己来取'}), set({'普通类本科及以上学历', '本科或者本科以上学历'}), set({'教辅人员', '教学辅导员'}), set({'取消岗位', '撤销'}),
    set({'缺技术', '没有技术', '缺乏技术'}), set({'35以下', '不足35', '35岁以下'}), set({'中国', '我国', '国内', '我们国'}), set({'教师', '老师'}), set({'1610年德国银币', '德国1610年的银币'}),
    set({'湖南卫视', '湖南台', '芒果台', '芒果tv', '芒果', '马桶台'}), set({'每批6个(满足实验和留样需求)', '6个'}), set({'2学分', '2个学分', '2分'}), set({'专技', '专业技术'}), set({'小学音乐(女生)', '女音乐'}),
    set({'梦想的声音3', '梦想的声音第三季'}), set({'暂时停止', '暂停'}), set({'2个20吨', '20*2'}), set({'建设完成', '完成建设'}), set({'七年级', '初一'}), set({'理工科', '理工类', '理科类', '理科'}), set({'工学', '工程学科'}),
    set({'一年约400小时', '每年大概400小时'}), set({'大专及以上', '专科毕业及以上'}), set({'2017年版的徙', '徙.2017'}), set({'日', '一天'}), set({'电视剧', '剧'}), set({'上海源本食品质量检验有限公司', '上海市的源本食品质检公司'}),
    set({'测量', '测量人员'}), set({'多平台', '多个平台'}), set({'增持', '增加持有', '买入', '加仓'}), set({'日报', '每天', '每日'}), set({'全年', '一年', '一整年'}), set({'优秀', '优等品'}), set({'高', '非常'}), 
    set({'初版', '初次'}), set({'强烈推荐-a', 'a'}), set({'梅观高速', '梅观'}), set({'玉米', '包谷'}), set({'北京市大兴区', '大兴区'}), set({'临床医疗(内科)', '内科'}), set({'临床医疗(心电)', '心电'}), set({'进口关税', '关税'}),
    set({'省级会议', '省级'}), set({'一般项目', '一般建设项目', '次关键'}), set({'必修的专业课程', '专业必修课'}), set({'一楼贵宾室', '一层的贵宾室'}), set({'本科', '学士'}), set({'硕士', '研究生'}), set({'建行', '建设银行'}),
    set({'高二理科', '高二年级理科'}), set({'散装', '散称', '没有进行包装'}), set({'b级', '等级b'}), set({'中医师', '中医医师'}), set({'a类企业', 'a类'}), set({'嘉应学院', '嘉大'}), set({'蜜蜂', '蜂产品'}), 
    set({'一楼报告厅', '一层的报告厅'}), set({'延期申请书', '申请延期'}), set({'科研', '科学研究', '科技研究'}), set({'思品', '思想品德'}), set({'女经理', '经理女'}), set({'市保', '市级'}), set({'正常', '没毛病'}),
    set({'亳州市第五中学', '亳州五中'}), set({'宝坻二中', '宝坻区第二中学'}), set({'一次性', '一次缴', '趸缴'}), set({'硼硅类玻璃', '硼硅'}), set({'初二', '八年级'}), set({'2014年4季度', '14年第四季度'}), set({'无合同', '没有合同'}),
    set({'船用燃油硫排放', 'seca', '硫氧化物排放控制区'}), set({'春城', '昆明'}), set({'鹿城', '三亚'}), set({'腾讯', '企鹅', '鹅厂'}), set({'2kg', '两千克', '2千克'}), set({'牡丹江市', '牡丹江市交通运输局网站'}),
    set({'液晶电视-海尔', '海尔牌液晶电视'}), set({'粤语', '广东话'}), set({'卫生间', '洗手间'}), set({'3a', 'aaa', '三a', '三个a', '3个a'}), set({'5a', 'aaaaa', '五a', '五个a', '5个a'}), set({'矢高(h)', '矢量高度'}),
    set({'元/生、年', '元', '生', '年'}), set({'蓝莓台', '蓝鲸台', '浙江台', '浙江卫视'}), set({'江苏卫视', '红台', '江苏台', '荔枝台'}), set({'攀枝花学院', '攀大'}), set({'广东卫视', '广州电视台'}), set({'同意', '通过', '过'}),
    set({'红利分发', '分红型'}), set({'流通', '销售', '售卖'}), set({'新办', '第一次', '首次'}), set({'a级别', 'a'}), set({'b级', 'b'}), set({'沃尔玛(珠海)商业零售有限公司', '珠海沃尔玛'}), set({'无证办学', '没有证件就办学'}),
    set({'不合格', '不达标', '不符合标准', '不符合规定', '达不到标准', '没达到标准'}), set({'合格', '合格品', '及格', '达标', '符合规定', '达到标准'}), set({'40岁或者40岁以下', '40周岁及以下'}), 
    set({'军用航空飞机', '军用飞机'}), set({'北京卫视', '北京台', '北京电视台', 'btv'}), set({'江西卫视', '江西台'}), set({'港币', '香港货币'}), set({'口咽通气管/进口', '进口的口咽通气管'}), set({'2001年6月第1版', '01年6月第1版'}), 
]

TLDR = {
    '直径约6mm,长约260mm': ['直径约为6毫米，长度大概有260毫米', '直径大约为6毫米，长度大约有260毫米', '直径大概是6mm,长度大概有260mm'], 'web应用高级开发人员': ['web应用高级开发员', '前端高级开发员', '高级前端开发员'], '河北小五台国家级自然保护区': ['小五台山保护区', '河北省小五台山自然保护区'], '专职教师': ['专职的老师', '专职老师', '教师', '老师'],
    '卡乐星(上海)餐饮管理有限公司': ['上海卡乐星', '上海的卡乐星'], '聘用后须在招聘单位服务满3周年': ['工作满三年', '干满3年'], '统招本科及以上': ['最低需要全国统一招生本科', '全国统一招生的本科以上'], '65克×6支/盒': ['每盒有6支，一支是65克'], '研究生(硕士)及以上': ['研究生及以上', '硕士或者硕士以上', '研究生或者研究生以上'],
    '(300×450×9.5)mm': ['厚度是9.5毫米，面积是300×450平方毫米', '面积为300×450平方毫米，厚度为9.5毫米'], '18周岁以上、35周岁以下': ['18到35周岁之间', '18至35周岁之间', '大于18岁小于35岁'], '20kg/桶/-': ['20千克一桶', '每桶是20千克'], '佛山市风行家电有限公司': ['佛山风行家电', '风行家电', '风行'], '专任教师': ['专任的教师', '专任老师', '教师', '老师'],
    '滨潍高速铁路': ['从滨州到潍坊的高速铁路', '滨州到潍坊的高铁', '滨州-潍坊的高速铁路'], 'a、b、c为三角形的三条边长。': ['三边长分别为a、b、c', '三边长 a、b、c是已知的', 'a、b、c为三条边长'], '彪马(上海)商贸有限公司': ['上海彪马', '上海市彪马', '上海的彪马'], '上海好孩子儿童服饰有限公司': ['上海市的好孩子儿'],
    '2012-2015': ['12年到15年', '12年开始，15年完成'], '插画入门(2班)': ['插画2班', '插画二班'], '600×240×200': ['长宽高分别是600mm，240mm和200mm', '长宽高分别为的600mm，240mm和200mm'], '250ml*1': ['一瓶250ml', '每瓶250ml'], '11*4.5*2cm，黑色': ['11*4.5*2cm'], '内页黑白，封彩': ['封面是彩色的，内页是黑白的', '内页黑白，封面彩色'],
    '350*350': ['长宽都是350毫米', '350毫米长宽', '三百五乘三百五'], '海拔1200-1500m林中': ['海拔1200-1500m的森林', '海拔高度为1200到1500m的森林中'], '保障经费；场馆需重新改造': ['需要重新改造场馆', '场馆需要重新改造', '场馆重新进行改造'], '10克*1支': ['每支是10克', '1支10克'], '房山区妇幼保健院': ['妇幼保健医院', '妇幼医院'],
    '全日制本科及以上': ['本科或者是本科以上', '本科或者本科以上', '本科学历及以上'], '18旅游管理1班(4)': ['18级旅游管理一班', '18级旅管1班'], '18金融学1班(4)': ['18级金融学一班', '18级金融1班', '18级金融一班'], '绿点(苏州)科技有限公司': ['苏州绿点科技', '苏州市绿点科技'], '已知弧长l，求面积a与圆心角α': ['面积a与圆心角α'],
    '终结者1': ['终结者第一部', '第一部终结者'], '亲爱的客栈第2季': ['第二季亲爱的客栈', '亲爱的客栈的他的第二季', '第二季的客栈'], '终结者2': ['终结者第二部', '第二部终结者'], '华夏人寿保险股份有限公司': ['华夏人寿', '华夏鸿利两全保险分红型c款'], '儿科临床医生': ['儿科的临床医生', '儿科'], 'feb2000': ['2000年2月', '00年的2月', '两千年二月'],
    '研究生/硕士及以上学历': ['研究生学历及以上', '硕士或者硕士以上'], '研究生/硕士及以上': ['研究生或者是以上', '硕士或者硕士学历以上'], '中国大陆/香港': ['中国大陆或者在香港', '中国大陆或在香港'], 'cny28.00': ['二十八元', '二十八块'], '35周岁以下': ['不超过三十五岁', '低于三十五岁'], '30周岁以下': ['30岁以下', '不超过30岁'],
    '1000-1120': ['10:00到11:20', '10:00至11:20', '10点起飞，11:20降落'], '4kg/瓶-35号乙二醇型轻负荷': ['一瓶4千克的35号乙二醇型轻负荷', '35号乙二醇型轻负荷型并且每瓶是4kg'], '器乐演奏': ['拉小提琴演奏或者吹笛子', '拉小提琴'], '电子座便器(智能坐便器)': ['智能电子座便器', '智能型电子座便器'], 'mar1969': ['1969年3月', '1969年三月'],
    '2017年4季度': ['2017年9~12月', '2017年第4季度', '17年的第四季度'], '建筑类、机械类': ['建筑专业，机械专业', '建筑专业和机械专业'], '日上上海': ['上海市日上','上海日上', '上海的日上'], '50000-60000元/年': ['5到6万', '5万到6万'], '35周岁以下,1978年8月1日以后出生': ['不超过三十五周岁', '年龄在三十五岁以下', '1978年8月1号之后出生'],
    '硕士研究生或以上': ['研究生或者研究生以上', '研究生及以上', '硕士或是硕士以上'], '博士或副高可放宽到40岁及以下，免笔试': ['博士或者副教授职称的可以放宽年龄的，还免笔试', '博士还可以放宽到40岁以下，并且还免笔试', '博士或者副教授职称的话，就可以免笔试'], '北滘镇党建工作指导员': ['北滘镇'], '佳驿酒店燕子山路店': ['燕子山路上佳驿酒店'], 
    '双语教学培训项目': ['双语教学'], '文化发展专项资金': ['文化发展'], '农业管理培训项目': ['农业管理培训', '农业管理'], '15枚/盒(748g/盒)': ['一盒总的有748g，也就是每盒有15个', '一盒是15个，即每盒总重748g', '每盒有748g，也可以说每盒有15个'], '中国平安人寿保险股份有限公司': ['平安人寿保险公司', '平安人寿'],
}

EN_NE = ['intel', 'samsung', 'youku', 'aframax', 'suezmax', 'vlcc', 'singapore', 'sd卡', 'cnfia', 'supor', 'duke university press', 'au9999', 'au999', 'seg', 'ge', 'lawson', 'citvc', 'kg', 'mg', 'ml', 'l', 'g']
ZH_NE = ['英特尔', '三星', '优酷', '阿芙拉型', '苏伊士型', '超大型', '新加坡', '内存卡', '中国食品工业协会', '苏泊尔', '杜克大学出版社', '万足金', '千足金', '国际勘探地球物理学家学会', '美国通用电气', '罗森', '中国国际电视总公司', '千克', '毫克', '毫升', '升', '克']
EN2ZH_NE = dict(zip(EN_NE, ZH_NE))
ZH2EN_NE = dict(zip(ZH_NE, EN_NE))

class ValueExtractor():

    def __init__(self) -> None:
        tool = LAC(mode='seg')
        self.nlp = lambda s: tool.run(s)

    def extract_values(self, entry: dict, db: dict, verbose=False):
        """ Extract values(class SQLValue) which will be used in AST construction,
        we need question_toks for comparison, entry for char/word idx mapping.
        The matched_index field in the ValueCandidate stores the index_pair of uncased_question_toks, not question.
        `track` is used to record the traverse clause path for disambiguation.
        """
        result = { 'values': [], 'entry': entry, 'question_toks': copy.deepcopy(entry['uncased_question_toks']) , 'db': db}
        result = self.extract_values_from_sql(entry['sql'], result)
        entry = self.assign_values(entry, result['values'])
        if verbose and len(entry['values']) > 0:
            print('Question:', ' '.join(entry['uncased_question_toks']))
            print('SQL:', entry['query'])
            print('Values:', ' ; '.join([repr(val) for val in entry['values']]), '\n')
        return entry

    def assign_values(self, entry, values):
        # take set because some SQLValue may use the same ValueCandidate
        cased_question_toks = entry['cased_question_toks']
        candidates = set([val.candidate for val in values if isinstance(val.candidate, ValueCandidate)])
        candidates = sorted(candidates)
        offset = SelectValueAction.size('nl2sql')
        for idx, val in enumerate(candidates):
            val.set_value_id(idx + offset)
            cased_value = ' '.join(cased_question_toks[val.matched_index[0]:val.matched_index[1]])
            val.set_matched_cased_value(cased_value)
        entry['values'], entry['candidates'] = set(values), candidates
        return entry

    def use_extracted_values(self, sqlvalue, values, fuzzy=False):
        """ When value can not be found in the question,
        one opportunity is to resort to extracted values for reference
        """
        val = sqlvalue.real_value
        old_candidates = [v.candidate for v in values if v.real_value.lower().strip() == val.lower().strip() and v.candidate is not None]
        if len(old_candidates) > 0: # use the last one
            sqlvalue.add_candidate(old_candidates[-1])
            return True
        if fuzzy and not is_number(val): # allow fuzzy match and not number
            val = val.lower()
            for v in values:
                matched_val = v.matched_value.replace(' ', '')
                if all(c in matched_val for c in val) or len(set(matched_val) & set(val)) >= 2:
                    sqlvalue.add_candidate(v.candidate)
                    return True
        return False

    def extract_values_from_sql(self, sql: dict, result: dict):
        conds = sql['conds']
        func = { str: self.extract_string_val, float: self.extract_float_val, int: self.extract_int_val }
        for idx, (col_id, cmp_id, val) in enumerate(conds):
            state = State(str(idx), 'none', CMP_OP[cmp_id], 'none', col_id)
            sqlvalue = SQLValue(str(val), state)
            if sqlvalue in result['values']: continue
            result['values'].append(sqlvalue)
            if is_int(val) and 'e' not in val.lower():
                val = int(float(val))
                result = func[int](val, result, sqlvalue)
            elif is_number(val) and 'e' not in val.lower():
                val = float(val)
                result = func[float](val, result, sqlvalue)
            else:
                result = func[str](NORM(val.strip().lower()), result, sqlvalue)
        return result
    
    def extract_int_val(self, num, result, sqlvalue):
        values, question_toks, entry = result['values'], result['question_toks'], result['entry']
        question = ''.join(question_toks)

        def parse_year(val):
            val = str(val)
            match = re.search(r'^\d{4}$', val)
            if not match: return False
            short_val = val[-2:]
            zh_val, zh_short_val = ZH_NUM2WORD(val, 'direct'), ZH_NUM2WORD(short_val, 'direct')
            match = re.search(fr'({zh_val}|{zh_short_val}|{short_val})[级年]', question)
            if match:
                add_value_from_char_idx((match.start(), match.end()), question_toks, sqlvalue, entry)
                return True
            return False

        def is_num_variants(val):
            if str(val) + '.0' in question: span = str(val) + '.0'
            elif re.search(r'[^\d]0' + str(val), question): span = '0' + str(val)
            elif val == 10 and re.search(r'十年', question): span = '十年'
            elif val < 10 and re.search(r'[^\d]' + str(val) + '年', question): span = str(val) + '年'
            elif str(val) in question: span = str(val)
            else: return False
            start_id = question.index(span)
            add_value_from_char_idx((start_id, start_id + len(span)), question_toks, sqlvalue, entry)
            return True

        def ignore_metric(val):
            pos = [(span.start(0), span.end(0)) for span in re.finditer(r'([0-9%s十百千万亿]+)' % (ZH_NUMBER), question)]
            candidates = []
            for s, e in pos:
                word = question[s:e]
                if s > 0 and re.search(r'[a-z年月]', question[s - 1], flags=re.I): continue
                if e < len(question) and re.search(fr'[ae\-\._年月日些批部层楼下手共月周年行线个]', question[e], flags=re.I): continue
                if re.search(r'^[百千万亿]+$', word): word = '一' + word # add prefix 一
                try:
                    num = ZH_WORD2NUM(word.rstrip('百千万亿'))
                    if str(val).startswith(str(num)): candidates.append((s, e))
                except: pass
            if len(candidates) > 0:
                add_value_from_char_idx(candidates[0], question_toks, sqlvalue, entry)
                return True
            return False            

        def is_ranking(val):
            col_id = sqlvalue.state.col_id
            if '排名' not in result['db']['column_names'][col_id][1]: return False
            zh_val, val = ZH_NUM2WORD(val - 1, 'low').replace('二', '[两二]'), str(val - 1)
            match =  re.search(fr'前({zh_val}|{val})', question)
            if match:
                start_id, end_id = match.start(1), match.end(1)
                add_value_from_char_idx((start_id, end_id), question_toks, sqlvalue, entry)
                return True
            return False

        def is_production_date(val): # no hyphen -
            col_id = sqlvalue.state.col_id
            col_name = result['db']['column_names'][col_id][1]
            match = re.search(r'(\d{4})(\d{2})(\d{2})', str(val))
            if match and re.search(r'生产.*日期', col_name):
                y, m, d = match.group(1), match.group(2), match.group(3)
                pattern = rf"{y}[/\-年]({m}|{m.lstrip('0')})[/\-月]({d}|{d.lstrip('0')})[日号]?"
                match = re.search(pattern, question)
                if match:
                    add_value_from_char_idx((match.start(0), match.end(0)), question_toks, sqlvalue, entry)
                    return True
            return False

        if is_ranking(num): return result
        start_ids = extract_number_occurrences(num, question, exclude_prefix='~', exclude_suffix='年')
        if len(start_ids) > 0:
            start_id = start_ids[0] # directly use the first one
            add_value_from_char_idx((start_id, start_id + len(str(num))), question_toks, sqlvalue, entry)
        elif try_number_to_word(num, question, question_toks, sqlvalue, entry): pass
        elif num == 0:
            if '负' in question or (not self.use_extracted_values(sqlvalue, values)):
                add_value_from_reserved('0', sqlvalue)
        elif parse_year(num): pass
        elif self.use_extracted_values(sqlvalue, values): pass
        elif is_production_date(num): pass
        elif is_num_variants(num): pass
        elif ignore_metric(num): pass
        elif num == 1: add_value_from_reserved('1', sqlvalue)
        else: raise ValueError('Unrecognized int value %s' % (num))
        return result

    def extract_float_val(self, num, result, sqlvalue):
        values, question_toks, entry = result['values'], result['question_toks'], result['entry']
        question = ''.join(question_toks)

        def parse_date(val):
            val = str(val).strip()
            match = re.search(r'^(\d{4})\.(\d{1,2})$', val)
            if match:
                y, m = match.group(1), match.group(2)
                short_y, short_m = y[-2:], m.lstrip('0')
                zh_y, short_zh_y, zh_m = ZH_NUM2WORD(y, 'direct'), ZH_NUM2WORD(short_y, 'direct'), ZH_NUM2WORD(short_m, 'low')
                y_pattern, m_pattern = f'({y}|{zh_y}|{short_y}|{short_zh_y})', f'({m}|{short_m}|{zh_m})'
                match = re.search(y_pattern + '年的?' + m_pattern + '月?', question)
                if match:
                    add_value_from_char_idx((match.start(), match.end()), question_toks, sqlvalue, entry)
                    return True
            return False
        
        def round_number(val):
            rval = str(round(val))
            start_ids = extract_number_occurrences(rval, question)
            if len(start_ids) > 0:
                start_id = start_ids[0]
                add_value_from_char_idx((start_id, start_id + len(rval)), question_toks, sqlvalue, entry)
                return True
            if math.floor(val) != round(val):
                rval = str(math.floor(val))
            else: rval = str(math.ceil(val))
            start_ids = extract_number_occurrences(rval, question)
            if len(start_ids) > 0:
                start_id = start_ids[0]
                add_value_from_char_idx((start_id, start_id + len(rval)), question_toks, sqlvalue, entry)
                return True
            return False

        def parse_rmb(val):
            rmb_pattern = []
            word = ZH_NUM2WORD(val, 'low')
            if '点' in word or '二' in word:
                rmb_pattern.append(word.replace('点', '[元块点]').replace('二', '[两二]'))
            val_10 = float(Decimal(str(val)) * Decimal('10'))
            val_100 = float(Decimal(str(val)) * Decimal('100'))
            if 0 < val < 1 and is_int(val_10): # 八毛, 八角
                rmb = int(val_10)
                rmb_pattern.append('[' + str(rmb) + DIGIT_ALIAS(rmb) + '][毛角]')
            elif 0 < val < 1 and is_int(val_100): # 八毛一, 八点一毛
                zh_val_10 = ZH_NUM2WORD(val_10, 'low')
                rmb_pattern.append(str(val_10).replace('.', '点') + '[毛角]')
                rmb_pattern.append(zh_val_10.replace('点', '[点毛角]').replace('二', '[两二]') + '[毛角]?')
            if len(rmb_pattern) == 0: return False
            match = re.search('(' + '|'.join(rmb_pattern) + ')', question)
            if match:
                add_value_from_char_idx((match.start(), match.end()), question_toks, sqlvalue, entry)
                return True
            return False
        
        def try_simple_number_to_word(num):
            for mode in ['low', 'direct']:
                try:
                    word = ZH_NUM2WORD(num, mode)
                    if word in question:
                        start_id = question.index(word)
                        add_value_from_char_idx((start_id, start_id + len(word)), question_toks, sqlvalue, entry)
                        return True
                except: pass
            return False

        start_ids = extract_number_occurrences(num, question)
        if len(start_ids) > 0:
            start_id = start_ids[0] # directly use the first one
            add_value_from_char_idx((start_id, start_id + len(str(num))), question_toks, sqlvalue, entry)
        elif try_simple_number_to_word(num): pass
        elif str(num) in question:
            start_id = question.index(str(num))
            add_value_from_char_idx((start_id, start_id + len(str(num))), question_toks, sqlvalue, entry)
        elif self.use_extracted_values(sqlvalue, values): pass
        elif parse_date(num): pass
        elif round_number(num): pass
        elif parse_rmb(num): pass
        elif try_percentage_variants(num, question, question_toks, sqlvalue, entry): pass
        else: raise ValueError('[ERROR]: Unrecognized float value %s' % (num))
        return result

    def extract_string_val(self, val, result, sqlvalue):
        values, question_toks, entry = result['values'], result['question_toks'], result['entry']
        question = ''.join(question_toks)

        def transform_number(val, reverse=False): # number and words transformation
            match = re.search(r'[0-9\.]+', val.replace(',', ''))
            if match:
                try:
                    span = match.group(0)
                    num = int(float(span)) if is_int(span) else float(span)
                    s = ZH_NUM2WORD(num, 'low')
                    val_num = val.replace(',', '').replace(span, s)
                    return val_num
                except: pass
            if not reverse: return val
            match = re.search(fr'[{ZH_NUMBER}{ZH_UNIT}]+', val)
            if match:
                try:
                    span = match.group(0)
                    s = str(ZH_WORD2NUM(num))
                    val_word = re.sub(span, s, val)
                    return val_word
                except: pass
            return val

        val_num = transform_number(val)

        def parse_reserved(val):
            if val in ['无', '不限', '没有限制', '无限制', '不限制', '不限专业', '专业不限', '规格不限', '免笔试', '免费', '/']:
                add_value_from_reserved('无', sqlvalue)
            elif val in ['是', '有', '√', '否', '否no', '未公布', '不适用', '未检出']:
                if val in ['是', '有', '√']: add_value_from_reserved('是', sqlvalue)
                else: add_value_from_reserved('否', sqlvalue)
            else: return False
            return True

        def parse_teachers(val):
            level, subject, teacher = '(小学|初中|高中|中学)?', '(语文|数学|英语|音乐|美术|体育|物理|生物|化学|政治|历史|地理|计算机)', '(教师|老师)?'
            match = re.search(r'^(.{0,2})' + level + r'?' + subject + teacher + r'$', val)
            if not match: return False
            prefix, level, subject, suffix = match.groups()
            if prefix and level: pattern = '(' + prefix + ')?' + '(' + level + '.{0,3})?'
            elif prefix and not level: pattern = '(' + prefix + '.{0,2})?'
            else: pattern = ''
            teacher = r'(.{0,2}教师|.{0,2}老师)?' if suffix else ''
            match = re.search(pattern + subject + teacher, question)
            if match:
                add_value_from_char_idx((match.start(), match.end()), question_toks, sqlvalue, entry)
                return True
            return False

        def try_very_long_text(val):
            if val in TLDR:
                candidates = TLDR[val]
                for cand in candidates:
                    if cand in question:
                        start_id = question.index(cand)
                        add_value_from_char_idx((start_id, start_id + len(cand)), question_toks, sqlvalue, entry)
                        return True
            return False

        def try_abbreviation_mapping(val):
            for s in ABBREV_SET:
                if val in s:
                    for v in s:
                        if v in question:
                            start_id = question.index(v)
                            add_value_from_char_idx((start_id, start_id + len(v)), question_toks, sqlvalue, entry)
                            return True
            return False

        def process_item_and_above(val):
            if '相关工作经验' in val:
                candidates = ['得有工作经验一年以上', '一年以上工作的经验', '工作经历得超过一年', '有三年工作经验', '有三年或者三年以上的相关工作经验', '三年相关工作经验']
                for span in candidates:
                    if span in question:
                        start_id = question.index(span)
                        add_value_from_char_idx((start_id, start_id + len(span)), question_toks, sqlvalue, entry)
                        return True
            match = re.search(r'(.+?)(\(含\)|及)?(以|之)上', val)
            if match:
                if len(match.group(1)) > 10: return False
                num = transform_number(match.group(1), reverse=True)
                pattern = r'((%s|%s).{1,4}(%s|%s)?[以之]上|(至少|最低|最少|最小|高于|大于|多于|超过|不止|不低于|不小于|不少于).{0,5}(%s|%s))' % (match.group(1), num, match.group(1), num, match.group(1), num)
                match_obj = re.search(pattern, question)
                if match_obj:
                    add_value_from_char_idx((match_obj.start(), match_obj.end()), question_toks, sqlvalue, entry)
                    return True
            match = re.search(r'(.+?)(\(含\)|及)?(以|之)下', val)
            if match:
                if len(match.group(1)) > 10: return False
                num = transform_number(match.group(1), reverse=True)
                pattern = r'((%s|%s).{1,4}(%s|%s)?[以之]下|(至多|最高|最多|最大|低于|小于|少于|不及|不足|不高于|不大于|不多于|不超过).{0,5}(%s|%s))' % (match.group(1), num, match.group(1), num, match.group(1), num)
                match_obj = re.search(pattern, question)
                if match_obj:
                    add_value_from_char_idx((match_obj.start(), match_obj.end()), question_toks, sqlvalue, entry)
                    return True
            return False

        def process_brackets(val):
            match = re.search(r'^(.+?)\((.+?)\)(.*?)$', val)
            if not match: return False
            val_ = re.sub(r'[、，,\(\)]', '', val)
            if val_ in question: # remove brackets
                start_id = question.index(val_)
                add_value_from_char_idx((start_id, start_id + len(val_)), question_toks, sqlvalue, entry)
                return True
            pattern = []
            m1, m2, m3 = match.group(1), match.group(2).replace('%vol', '度'), re.sub(r'[、，,\(\)]', '', match.group(3))
            if re.search(r'第(.)版', m2): # 第三版, 第3版
                version = re.search(r'第(.)版', m2).group(1)
                alias = '[' + DIGIT_ALIAS(int(version)) + f'{version}]' if is_int(version) else '[' + str(ZH_WORD2NUM(version)) + f'{version}]'
                m2 = re.sub(version, alias, m2)
            if m3:
                pattern += [m3 + r'.{0,1}' + m2 + r'.{0,1}' + m1, m2 + r'.{0,1}' + m3 + r'.{0,1}' + m1]
                if len(m3) > 1: pattern += [m3[:-1] + r'.{0,1}' + m2 + r'.{0,1}' + m1, m2 + r'.{0,1}' + m3[:-1] + r'.{0,1}' + m1]
            else:
                pattern += [m1 + r'.{0,1}' + m2, m2 + r'.{0,1}' + m1]
                if len(m2) > 1: pattern += [m2[:-1] + r'.{0,1}' + m1, m2[1:] + r'.{0,1}' + m1]
            pattern = '(' + '|'.join(pattern + [m2 + m3 if m1 in m2 else m1 + m3]) + ')'
            if val == '冰糖糯米(糯米口味冰棍)': pattern = r'冰糖糯米(口?味的?)?冰棍'
            match = re.search(pattern, question)
            if match:
                add_value_from_char_idx((match.start(), match.end()), question_toks, sqlvalue, entry)
                return True
            return False

        def ignore_suffix(val):
            if val == '九州出版': val = '九州出版社'
            match = re.search(r'(.+)(公安分局人口管理科|/好奇心小百科|有限责任公司|股份有限公司|客票代售点|营业室|营业处|农药厂|出版社|支行|培训中心|枸杞酒|显示器)', val)
            if match:
                val_ = match.group(1)
                if val_ in question:
                    start_id = question.index(val_)
                    add_value_from_char_idx((start_id, start_id + len(val_)), question_toks, sqlvalue, entry)
                    return True
            return False

        def reorder_words(val):
            toks = self.nlp(val)
            if len(toks) <= 3:
                val_ = ''.join(toks[::-1])
                if val_ in question:
                    start_id = question.index(val_)
                    add_value_from_char_idx((start_id, start_id + len(val_)), question_toks, sqlvalue, entry)
                    return True
            if len(val) == 3:
                val_ = val[1:] + val[0:1]
                if val_ in question:
                    start_id = question.index(val_)
                    add_value_from_char_idx((start_id, start_id + len(val_)), question_toks, sqlvalue, entry)
                    return True
                val_ = val[-1] + val[:-1]
                if val_ in question:
                    start_id = question.index(val_)
                    add_value_from_char_idx((start_id, start_id + len(val_)), question_toks, sqlvalue, entry)
                    return True
            if len(val) == 4:
                val_ = val[2:] + val[0:2]
                if val_ in question:
                    start_id = question.index(val_)
                    add_value_from_char_idx((start_id, start_id + len(val_)), question_toks, sqlvalue, entry)
                    return True
            match = re.search(r'\((.+?)\)', val)
            if match and val.count('(') == 1:
                val_ = match.group(1) + re.sub(r'\(.+?\)', '', val)
                if val_ in question:
                    start_id = question.index(val_)
                    add_value_from_char_idx((start_id, start_id + len(val_)), question_toks, sqlvalue, entry)
                    return True
            return False

        def parse_date(val):
            if val.count('-') > 2 or re.search(r'(元|季度|版本)', val) or re.search(r'\d{4}-\d{4}', val): return False
            if re.search(r'[年月日号]', val) and re.search(r'[到至\-]', val) and not re.search(r'(周|星期)', val):
                pattern = fr'([日号至到\-\d{ZH_NUMBER}和份，]|年的?|月的?)+'
                match = re.search('^' + pattern + '$', val)
                if val != '购进日期2018-07-09' and not match: return False
                longest = [0, [0, 0]]
                for match in re.finditer(pattern, question):
                    start_id, end_id = match.start(), match.end()
                    if question[end_id - 1] == '，': end_id -= 1
                    if end_id - start_id > longest[0]:
                        longest = [end_id - start_id, (start_id, end_id)]
                if longest[0] > 0:
                    add_value_from_char_idx(longest[1], question_toks, sqlvalue, entry)
                    return True
                return False
            match = re.search(r'^[\d\.]{4,}-[\d\.]{4,}$', val)
            if match:
                pattern = r'[年月日号至到\-\d份]+'
                longest = [0, [0, 0]]
                for match in re.finditer(pattern, question):
                    start_id, end_id = match.start(), match.end()
                    if end_id - start_id > longest[0]:
                        longest = [end_id - start_id, (start_id, end_id)]
                if longest[0] > 0:
                    add_value_from_char_idx(longest[1], question_toks, sqlvalue, entry)
                    return True
                return False
            match = re.search(r'(\d{4})[/\-年\.](\d{1,2})[/\-月\.](\d{1,2})[日号]?', val) # year-month-day
            if match:
                y, m, d = match.group(1), match.group(2), match.group(3)
                short_y, short_m, short_d = y[-2:], m.lstrip('0'), d.lstrip('0')
                zh_y, short_zh_y = ZH_NUM2WORD(y, 'direct'), ZH_NUM2WORD(short_y, 'direct')
                zh_m, zh_d = ZH_NUM2WORD(short_m, 'low'), ZH_NUM2WORD(short_d, 'low')
                pattern = fr'(({y}|{short_y}|{zh_y}|{short_zh_y}).+?({m}|{short_m}|{zh_m}).+?({d}|{short_d}|{zh_d})|({y}|{short_y}|{zh_y}|{short_zh_y}).+?({m}|{short_m}|{zh_m}).)'
                match = re.search(pattern, question)
                if match:
                    add_value_from_char_idx((match.start(), match.end()), question_toks, sqlvalue, entry)
                    return True
            match = re.search(r'(\d{4})[/\-年\.](\d{1,2})[/\-月\.]?', val) # year-month
            if match:
                y, m = match.group(1), match.group(2)
                short_y, short_m = y[-2:], m.lstrip('0')
                zh_y, short_zh_y, zh_m = ZH_NUM2WORD(y, 'direct'), ZH_NUM2WORD(short_y, 'direct'), ZH_NUM2WORD(short_m, 'low')
                pattern = fr'({y}|{short_y}|{zh_y}|{short_zh_y})年的?第?({m}|{short_m}|{zh_m})个?月?'
                match = re.search(pattern, question)
                if match:
                    add_value_from_char_idx((match.start(), match.end()), question_toks, sqlvalue, entry)
                    return True
            match = re.search(r'(\d{1,2})月(\d{1,2})[日号]', val) # month-day
            if match:
                m, d = match.group(1), match.group(2)
                short_m, short_d = m.lstrip('0'), d.lstrip('0')
                zh_m, zh_d = ZH_NUM2WORD(short_m, 'low'), ZH_NUM2WORD(short_d, 'low')
                pattern = fr'({m}|{short_m}|{zh_m})月的?({d}|{short_d}|{zh_d})[日号]?'
                match = re.search(pattern, question)
                if match:
                    add_value_from_char_idx((match.start(), match.end()), question_toks, sqlvalue, entry)
                    return True
            match = re.search(r'(\d{4}|\d{2})年.{0,1}$', val) # 94年
            if match:
                y = match.group(1)
                short_y = y[-2:]
                zh_y, short_zh_y = ZH_NUM2WORD(y, 'direct'), ZH_NUM2WORD(short_y, 'direct')
                pattern = fr'({y}|{short_y}|{zh_y}|{short_zh_y})年'
                match = re.search(pattern, question)
                if match:
                    add_value_from_char_idx((match.start(), match.end()), question_toks, sqlvalue, entry)
                    return True
            return False

        def parse_metric(val):
            match = re.search(r'^([\d\.]+)([^\-/]+)/([^\-/]+)$', val)
            if not match or '×' in val: return False
            num, metric1, metric2 = match.groups()
            if metric2[0] == '瓶' and len(metric2) > 1:
                if '%vol' in metric2: metric2 = '瓶'
                else: metric1, metric2 = metric1 + '.{0,1}' + metric2[1:].strip('()') + '.', '瓶'
            metric1_en = ['g', 'kg', 'l', 'ml', 'mg', '吨']
            metric1_zh = ['克', '千克', '升', '毫升', '毫克', '吨']
            try:
                num_word = ZH_NUM2WORD(num, 'low').replace('二', '[两二]')
                num += '|' + num_word
            except: pass
            num_p = '(' + num + ')'
            metric1 = metric1 + '|' + metric1_zh[metric1_en.index(metric1)] if metric1 in metric1_en else \
                metric1 + '|' + metric1_en[metric1_zh.index(metric1)] if metric1 in metric1_zh else metric1
            m1_p = '(' + metric1 + ')'
            m2_p = '[一每].{0,1}' + metric2
            p = '(' + m2_p + '.{0,2}' + num_p + m1_p + '|' + num_p + m1_p + m2_p + '|' + num_p + m1_p + '|' + m2_p + '.{0,2}' + num_p + ')'
            match = re.search(p, question)
            if match:
                add_value_from_char_idx((match.start(), match.end()), question_toks, sqlvalue, entry)
                return True
            return False

        def fuzzy_match(val):
            dist, length = [], len(val)
            index_pairs = list(filter(lambda x: min([1, length - 4]) <= x[1] - x[0] <= length + 4, combinations(range(len(question) + 1), 2)))
            index_pairs = sorted(index_pairs, key=lambda x: x[1] - x[0])
            for i, j in index_pairs:
                span = question[i:j]
                score, score_num = float(edt.eval(val, span)) / len(val), float(edt.eval(val_num, span)) / len(val_num)
                dist.append(min([score, score_num]))
            min_dist = sorted(dist)[0]
            threshold = math.floor(len(val) / 2.0) / len(val) if len(val) < 6 else 0.43
            if min_dist <= threshold:
                index_pair = index_pairs[dist.index(min_dist)]
                if PLACEHOLDER in question[index_pair[0]: index_pair[1]]:
                    return False
                add_value_from_char_idx((index_pair[0], index_pair[1]), question_toks, sqlvalue, entry)
                return True
            return False

        def parse_outliers(val):
            mappings = {'福建省三明市梅列区玫瑰新村86幢负一层至负二层': '玫瑰新村86幢的负一层和负二层', '世界儿童共享的经典丛书:格林童话': '格林童话', '四川理工学院': '川理', '恒大人寿保险有限公司': '恒大人寿',
            '郑州星光车业有限公司': '星光车业', '广州市天谱电器有限公司': '广州天谱电器', '永吉县万昌镇': '永吉县名叫万昌的镇子', '护理专业(助产方向)': '助产护理', '北京统一饮品有限公司': '北京统一',
            '智能马桶盖(智能温水洗净便座)': '智能型马桶盖', '悠游堂海盗主题乐园北京祥云小镇店': '北京祥云小镇的悠游堂海盗主题乐园', '定量:16.5g/㎡2层': '2层每平方米16.5g', '本书适用于大众读者': '大众阅读', '英语专任教师': '英语老师',
            '2019-02-14': '19年情人节', '惊悚/悬疑': '悬疑还有惊悚', '上海盈泰塑胶有限公司': '上海市盈泰塑胶', '时时乐万柳店': '万柳的时时乐', '中影数字院线(北京)有限公司': '北京中影', '动作/惊悚': '惊悚类、动作类',
            '沈阳南站-长白山站': '沈阳到长白山', '市职业中等专业学校': '市职业中专', '浙江苏泊尔股份有限公司': '苏泊尔公司', '省工商局统一安排2018年度抽查工作计划': '省工商局', '北京市朝阳区': '朝阳区', '焦作市温县': '温县',
            '山西省太原市娄烦县初中信息技术教师-尖山学校-乡镇九年制': '初中信息技术老师', '市经济和信息化委员会': '市经信委', '晏子春秋／陈涛译注': '晏子春秋', '苏州三星电子电脑有限公司': 'samsung', 'canon相机': '佳能相机',
            'nikon相机': '尼康相机', 'iphone手机': '苹果手机', '工业和信息化部电子第五研究所华东分所': '中国赛宝(华东)实验室', '劳资管理岗位': '劳动资源管理', '锦屏-苏南直流800千伏': '锦屏县到苏南地区的特高压建设',
            '2013-01-01': '13年元旦', '高铁站-深渡': '深渡', '18旅游管理2班(4)': '二班', '插画入门(1班)': '插画一班', '时时乐望京店': '望京的时时乐', '中国人民大学': '人大', '外勤，适宜男性。': '外勤',
            '江西上饶市广丰县花炮厂“11.6”事故': '11月6日广丰县花炮厂', '硕士学位及以上': '最低学位需要硕士', '全日制大学本科及以上': '本科或者是本科以上', '大学本科及以上': '最低学历需要本科', '统一冰红茶(柠檬味红茶饮料)': '冰红茶',
            '安魂曲.2017': '2017版安魂曲', 'skinfood黑豆保湿乳液': '思亲肤的黑豆保湿乳液', '东北大米(珍珠米)': '东北珍珠大米', '中国农业银行江苏省分行营业部虹桥支行': '虹桥支行', '花旗参(原枝)四两裝': '花旗参',
            '硕士研究生及以上': '硕士或者硕士以上', '炒货食品及坚果制品': '炒货或是坚果类', '个人理财规划(尔雅网络课)': '个人规划理财课', '诸神纪.2017': '17版的诸神纪', '高粱原浆(50%vol)(浓香型)': '50度高粱原浆',
            '帝企鹅日记2:召唤': '帝企鹅日记第二部',}
            if val in mappings and mappings[val] in question:
                span = mappings[val]
                start_id = question.index(span)
                add_value_from_char_idx((start_id, start_id + len(span)), question_toks, sqlvalue, entry)
                return True
            return False

        def parse_substring(val):
            pos = search_for_longest_substring(question, val, ignore_chars='的个省市区')
            if pos[1] - pos[0] >= 2 and question[pos[0]:pos[1]] not in ['出版', '食品', '医院', '水泥', '软件', '公司', '电器']:
                add_value_from_char_idx((pos[0], pos[1]), question_toks, sqlvalue, entry)
                return True
            return False

        def english_chinese_transform(val):
            if val in EN2ZH_NE:
                val = EN2ZH_NE[val]
                if val in question:
                    start_id = question.index(val)
                    add_value_from_char_idx((start_id, start_id + len(val)), question_toks, sqlvalue, entry)
                    return True
            if val in ZH_NE:
                val = ZH2EN_NE[val].replace(' ', '')
                if val in question:
                    start_id = question.index(val)
                    add_value_from_char_idx((start_id, start_id + len(val)), question_toks, sqlvalue, entry)
                    return True
            return False

        def parse_time(val):
            is_week = re.search(r'(周[一二三四五六七日天]|星期[一二三四五六七日天]|工作日|休息日|周末)', val) and ':' not in val
            is_time = re.search(r'(早上|晚上|中午|晚上|上午|下午|[\d一二三四五六七八九十]点)', entry['question']) and (':' in val or re.search(r'\d点', val))
            if not is_time and not is_week: return False
            if val == '周一至周五' and '工作日' in question:
                start_id = question.index('工作日')
                add_value_from_char_idx((start_id, start_id + len('工作日')), question_toks, sqlvalue, entry)
                return True
            if val == '周一至周日' and re.search(r'每一?天', question):
                match = re.search(r'每一?天', question)
                add_value_from_char_idx((match.start(), match.end()), question_toks, sqlvalue, entry)
                return True
            if val == '10月24日周三' and '十月二十四日星期三' in question:
                start_id = question.index('十月二十四日星期三')
                add_value_from_char_idx((start_id, start_id + len('十月二十四日星期三')), question_toks, sqlvalue, entry)
                return True
            if is_week:
                val_norm = re.sub(r'\(.+?\)', lambda m: m.group() + '?', val)
                val_norm = re.sub(r'[至到]', '[至到]', val_norm)
                val_norm = re.sub(r'[日号天]', '[日号天]', val_norm)
                week_pattern = re.sub(r'(星期|周)', '(星期|周)', val_norm).replace('2', '2?').replace('0', '0?')
                match = re.search(week_pattern, question)
                if match:
                    add_value_from_char_idx((match.start(), match.end()), question_toks, sqlvalue, entry)
                    return True
            if val == '早8点30南湖校区大成教学馆302室' and '早上8点30南湖校区大成教学馆302室' in question:
                start_id = question.index('早上8点30南湖校区大成教学馆302室')
                add_value_from_char_idx((start_id, start_id + len('早上8点30南湖校区大成教学馆302室')), question_toks, sqlvalue, entry)
                return True
            if is_time:
                longest = [0, (0, 0)]
                time_pattern = fr'([\d{ZH_NUMBER}十年月周日号的:]|星期[一二三四五六七日天]|考?到|点半?|早上|晚上|上午|中午|下午|那天|开始，?|开考，|结束)+'
                for match in re.finditer(time_pattern, question):
                    start_id, end_id = match.start(), match.end()
                    span = question[start_id: end_id]
                    if span.startswith('的') or span.startswith('到'): start_id += 1
                    if span.endswith('的') or span.endswith('到'): end_id -= 1
                    if span.startswith('开始') or span.startswith('结束'): start_id += 2
                    if span.endswith('开始') or span.endswith('结束'): end_id -= 2
                    if end_id - start_id > longest[0]:
                        longest = [end_id - start_id, (start_id, end_id)]
                if longest[0] > 0:
                    add_value_from_char_idx(longest[1], question_toks, sqlvalue, entry)
                    return True
            return False

        if parse_reserved(val): pass
        elif val in question:
            start_id = question.index(val)
            add_value_from_char_idx((start_id, start_id + len(val)), question_toks, sqlvalue, entry)
        elif val_num in question:
            start_id = question.index(val_num)
            add_value_from_char_idx((start_id, start_id + len(val_num)), question_toks, sqlvalue, entry)
        elif parse_teachers(val): pass
        elif try_very_long_text(val): pass
        elif parse_time(val): pass
        elif parse_date(val): pass
        elif english_chinese_transform(val): pass
        elif try_abbreviation_mapping(val): pass
        elif process_item_and_above(val): pass
        elif process_brackets(val): pass
        elif ignore_suffix(val): pass
        elif reorder_words(val): pass
        elif parse_metric(val): pass
        elif fuzzy_match(val): pass
        elif parse_outliers(val): pass
        elif parse_substring(val): pass
        elif self.use_extracted_values(sqlvalue, values, fuzzy=True): pass
        else: raise ValueError('[ERROR]: Unrecognized string value %s' % (val))
        return result


def extract_number_occurrences(num, question, exclude_prefix='', exclude_suffix=''):
    """ Extract all occurrences of num in the questioin, return the start char ids.
    But exclude those with specified prefixes or suffixes. In these cases, num may only be part of the exact number, e.g. 10 in 100.
    """
    pos = [(span.start(0), span.end(0)) for span in re.finditer(str(num), question)]
    char_ids = []
    for s, e in pos:
        if s > 0 and re.search(fr'[0-9a-z\-\._{ZH_NUMBER}{ZH_UNIT}{exclude_prefix}年月]', question[s - 1], flags=re.I): continue
        if e < len(question) - 1 and re.search(r'千米|千克|千瓦|千卡|千斤|kg|kw|cm|km|k|w|g|m|l', question[e:e+2], flags=re.I):
            char_ids.append(s)
            continue
        if e < len(question) and re.search(fr'[a-z0-9\-\._{ZH_NUMBER}{exclude_suffix}月日]', question[e], flags=re.I): continue
        char_ids.append(s)
    return char_ids


def try_number_to_word(num, question, question_toks, sqlvalue, entry):
    for mode in ['low', 'direct']:
        try:
            word = ZH_NUM2WORD(num, mode)
            if word in question:
                if question.count(word) == 1:
                    start_id = question.index(word)
                    end_id = start_id + len(word)
                    if start_id > 0 and re.search(rf'[{ZH_NUMBER}{ZH_UNIT}0-9]', question[start_id - 1]): continue
                    if end_id < len(question) and num < 100 and re.search(rf'[些部层楼下手共月日号周年股线{ZH_NUMBER}ae]', question[end_id]): continue
                    add_value_from_char_idx((start_id, end_id), question_toks, sqlvalue, entry)
                    return True
        except Exception as e: pass
    candidates = []
    pos = [(span.start(0), span.end(0)) for span in re.finditer(r'([0-9\.点%s十百千万亿]+)' % (ZH_NUMBER), question)]
    for s, e in pos:
        word = question[s: e]
        if e < len(question) and re.search(rf'[些批部层楼下手共月日周年号股线{ZH_NUMBER}ae]', question[e]): continue
        try:
            parsed_num = ZH_WORD2NUM(word)
            if parsed_num == num:
                candidates.append((s, e))
                continue
        except Exception as e: pass
        word_ = word.rstrip('万亿')
        if word_ == word: continue
        try:
            parsed_num = ZH_WORD2NUM(word_)
            if parsed_num == num:
                candidates.append((s, e))
        except Exception as e: pass
    if len(candidates) > 0:
        add_value_from_char_idx(candidates[0], question_toks, sqlvalue, entry)
        return True
    return False


def try_percentage_variants(num, question, question_toks, sqlvalue, entry):
    num_pattern = '|'.join([str(num).replace('.', '点'), ZH_NUM2WORD(num, 'low'), ZH_NUM2WORD(num, 'direct')])
    num_100 = float(Decimal(str(num)) * Decimal('100'))
    if is_int(num_100):
        num_100 = int(num_100)
        num_100_pattern = '|'.join([str(num_100), ZH_NUM2WORD(num_100, 'low'), ZH_NUM2WORD(num_100, 'direct')])
    else: num_100_pattern = '|'.join([str(num_100), str(num_100).replace('.', '点'), ZH_NUM2WORD(num_100, 'low'), ZH_NUM2WORD(num_100, 'direct')])
    percentage = f'(百分之)?({num_pattern}|{num_100_pattern})%?'
    match_obj = re.search(percentage, question)
    if match_obj:
        start_id, end_id = match_obj.start(0), match_obj.end(0)
        add_value_from_char_idx((start_id, end_id), question_toks, sqlvalue, entry)
        return True
    return False


def add_value_from_reserved(val, sqlvalue):
    sqlvalue.add_candidate(SelectValueAction.vocab('nl2sql')[val])


def add_value_from_char_idx(index_pairs, question_toks, sqlvalue, entry):
    start_id, end_id = index_pairs
    start = entry['char2word_id_mapping'][start_id]
    end = entry['char2word_id_mapping'][end_id - 1] + 1
    value = ' '.join(question_toks[start: end])
    candidate = ValueCandidate(matched_index=(start, end), matched_value=value)
    sqlvalue.add_candidate(candidate)
    question_toks[start: end] = [PLACEHOLDER * len(question_toks[idx]) for idx in range(start, end)]
    return question_toks