#coding=utf8
import re, os, sys, copy, math
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import editdistance as edt
from LAC import LAC
from decimal import Decimal
from itertools import combinations
from asdl.transition_system import SelectValueAction
from preprocess.process_utils import ValueCandidate, State, SQLValue
from preprocess.process_utils import is_number, is_int, search_for_longest_substring
from preprocess.process_utils import ZH_NUM2WORD, ZH_WORD2NUM, ZH_NUMBER, ZH_UNIT, DIGIT_ALIAS

AGG_OP = ('none', 'avg', 'max', 'min', 'count', 'sum')
CMP_OP = ('>', '<', '==', '!=')
PLACEHOLDER = '|' # | never appears in the dataset

ABBREV_SET = [
    set({'重点项目', '重点研究', '重点工程', '重点课题'}), set({'自取', '自行拿取', '自己来取'}), set({'普通类本科及以上学历', '本科或者本科以上学历'}), set({'教辅人员', '教学辅导员'}), set({'取消岗位', '撤销'}),
    set({'缺技术', '没有技术', '缺乏技术'}), set({'35以下', '不足35', '35岁以下'}), set({'小学部的音乐教师', '小学音乐'}), set({'中国', '我国', '国内', '我们国'}), set({'教师', '专任教师', '专职教师', '老师'}),
    set({'湖南卫视', '湖南台', '芒果台', '芒果tv', '芒果', '马桶台'}), set({'每批6个（满足实验和留样需求）', '6个'}), set({'免笔试', '没有笔试', '不用笔试', '不需要笔试'}), set({'2学分', '2个学分', '2分'}),
    set({'梦想的声音3', '梦想的声音第三季'}), set({'暂时停止', '暂停'}), set({'2个20吨', '20*2'}), set({'建设完成', '完成建设'}), set({'七年级', '初一'}), set({'理工科', '理工类', '理科类', '理科'}),
    set({'一年约400小时', '每年大概400小时'}), set({'大专及以上', '专科毕业及以上'}), set({'2017年版的徙', '徙.2017'}), set({'专技', '专业技术'}), set({'终结者第二部', '终结者2', '第二部终结者'}), set({'日', '一天'}),
    set({'测量', '测量人员'}), set({'免费', '免'}), set({'多平台', '多个平台'}), set({'增持', '增加持有'}), set({'未公布', '没有公布'}), set({'日报', '每天', '每日'}), set({'全年', '一年', '一整年'}),
    set({'初版', '初次'}), set({'强烈推荐-a', 'a'}), set({'梅观高速', '梅观'}), set({'第二季亲爱的客栈', '亲爱的客栈第2季', '第二季的客栈'}), set({'终结者第一部', '第一部终结者', '终结者1'}), set({'玉米', '包谷'}),
    set({'省级会议', '省级'}), set({'一般项目', '一般建设项目', '次关键'}), set({'必修的专业课程', '专业必修课'}), set({'一楼贵宾室', '一层的贵宾室'}), set({'合格', '合格品', '及格', '达标', '符合规定', '达到标准'}),
    set({'高二理科', '高二年级理科'}), set({'散装', '散称', '没有进行包装'}), set({'b级', '等级b'}), set({'中医师', '中医医师'}), set({'无合同', '没有合同'}), set({'a类企业', 'a类'}), set({'嘉应学院', '嘉大'}),
    set({'一楼报告厅', '一层的报告厅'}), set({'延期申请书', '申请延期'}), set({'科研', '科学研究', '科技研究'}), set({'思品', '思想品德'}), set({'女经理', '经理女'}), set({'市保', '市级'}), set({'正常', '没毛病'}),
    set({'亳州市第五中学', '亳州五中'}), set({'宝坻二中', '宝坻区第二中学'}), set({'一次性', '一次缴', '趸缴'}), set({'硼硅类玻璃', '硼硅'}), set({'日上上海', '上海日上', '上海的日上'}), set({'初二', '八年级'}),
    set({'50000-60000元/年', '5到6万', '5万到6万'}), set({'硫氧化物', 'seca', '硫排放'}), set({'春城', '昆明'}), set({'鹿城', '三亚'}), set({'腾讯', '企鹅', '鹅厂'}), set({'3a', 'aaa', '三a', '三个a', '3个a'}),
    set({'5a', 'aaaaa', '五a', '五个a', '5个a'}), set({'不合格', '不达标', '不符合标准', '不符合规定', '达不到标准', '没达到标准'}), set({'本科', '学士'}), set({'硕士', '研究生'}), set({'电视剧', '剧'}),
    set({'液晶电视-海尔', '海尔牌液晶电视'}), set({'粤语', '广东话'}), set({'蜜蜂', '蜂产品'}), set({'优秀', '优等品'}), set({'高', '非常'}), set({'买入', '增持', '加仓'}), set({'卫生间', '洗手间'}),
    set({'元/生、年', '元', '生', '年'}), set({'蓝莓台', '蓝鲸台', '浙江卫视'}), set({'江苏卫视', '红台', '荔枝台'}), set({'同意', '通过', '过'}), set({'攀枝花学院', '攀大'}), set({'广东卫视', '广东电视台'}),
    set({'矢高（h）', '矢量高度'}), set({'红利分发', '分红型'}), set({'流通', '销售', '售卖'}), set({'cny28.00', '二十八元', '二十八块'}), set({'新办', '第一次', '首次'}), set({'2kg', '两千克', '2千克'})
]

EN_NE = ['intel', 'samsung', 'youku', 'aframax', 'suezmax', 'vlcc', 'singapore', 'sd卡', 'cnfia', 'supor', 'duke university press', 'au9999', 'au999', 'seg', 'ge', 'lawson', 'citvc', 'btv']
ZH_NE = ['英特尔', '三星', '优酷', '阿芙拉型', '苏伊士型', '超大型', '新加坡', '内存卡', '中国食品工业协会', '苏泊尔', '杜克大学出版社', '万足金', '千足金', '国际勘探地球物理学家学会', '美国通用电气', '罗森', '中国国际电视总公司', '北京电视台']
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
        # entry = self.assign_values(entry, result['values'])
        # if verbose and len(entry['values']) > 0:
            # print('Question:', ' '.join(entry['uncased_question_toks']))
            # print('SQL:', entry['query'])
            # print('Values:', ' ; '.join([repr(val) for val in entry['values']]), '\n')
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
        the last chance is to resort to extracted values for reference
        """
        val = sqlvalue.real_value
        old_candidates = [v.candidate for v in values if v.real_value == val and v.candidate is not None]
        if len(old_candidates) > 0: # use the last one
            sqlvalue.add_candidate(old_candidates[-1])
            return True
        if fuzzy:
            val = val.lower()
            for v in values:
                matched_val = v.matched_value.replace(' ', '')
                if all(c in matched_val for c in val):
                    sqlvalue.add_candidate(v.candidate)
                    return True
                index_pairs = list(filter(lambda x: 1 < x[1] - x[0] <= len(val), combinations(range(len(matched_val) + 1), 2)))
                for s, e in index_pairs:
                    if edt.eval(val, matched_val[s:e]) / float(len(val)) < 0.5:
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
                result = func[str](val.strip().lower(), result, sqlvalue)
        return result
    
    def extract_int_val(self, num, result, sqlvalue):
        values, question_toks, entry = result['values'], result['question_toks'], result['entry']
        question = ''.join(question_toks)

        def parse_year(val):
            val = str(val)
            match = re.search(r'^\d{4}$', val)
            if not match: return False
            span = val[-2:]
            long_zh_span = ZH_NUM2WORD(val, 'direct').replace('一', '[一幺]')
            zh_span = ZH_NUM2WORD(span, 'direct').replace('一', '[一幺]')
            match = re.search(fr'({long_zh_span}|{zh_span}|{span})[级年]', question)
            if match:
                start_id, end_id = match.start(), match.end()
                add_value_from_char_idx((start_id, end_id), question_toks, sqlvalue, entry)
                return True
            return False

        def fix_num_in_question(val):
            if str(val) + '.0' in question:
                span = str(val) + '.0'
            elif re.search(r'[^\d]0' + str(val), question):
                span = '0' + str(val)
            elif val <= 10 and re.search(r'[^\d]' + str(val) + '年', question):
                span = str(val) + '年'
            elif str(val) in question:
                span = str(val)
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
                if re.search(r'^[百千万亿]+$', word):
                    word = '一' + word # add prefix 一
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

        def is_production_date(val):
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

        start_ids = extract_number_occurrences(num, question, exclude_prefix='~', exclude_suffix='年')
        if len(start_ids) > 0:
            start_id = start_ids[0] # directly use the first one
            add_value_from_char_idx((start_id, start_id + len(str(num))), question_toks, sqlvalue, entry)
        elif try_number_to_word(num, question, question_toks, sqlvalue, entry):  pass
        elif num == 0:
            if '负' in question or (not self.use_extracted_values(sqlvalue, values)):
                add_value_from_reserved('0', sqlvalue)
        elif parse_year(num): pass
        elif self.use_extracted_values(sqlvalue, values): pass
        elif is_ranking(num): pass
        elif fix_num_in_question(num): pass
        elif ignore_metric(num): pass
        elif is_production_date(num): pass
        elif num == 1: add_value_from_reserved('1', sqlvalue)
        else: pass
            # raise ValueError('Unrecognized int value %s' % (num))
        return result

    def extract_float_val(self, num, result, sqlvalue):
        values, question_toks, entry = result['values'], result['question_toks'], result['entry']
        question = ''.join(question_toks)

        def parse_date(val):
            val = str(val).strip()
            match = re.search(r'^(\d{4})\.(\d{1,2})$', val)
            if match:
                y, m = match.group(1), match.group(2)
                short_y, zh_y = y[-2:], ZH_NUM2WORD(y, 'direct')
                short_zh_y = ZH_NUM2WORD(short_y, 'direct')
                y_pattern = f'({y}|{zh_y}|{short_y}|{short_zh_y})'
                short_m = m.lstrip('0')
                zh_m = ZH_NUM2WORD(short_m, 'low')
                m_pattern = f'({m}|{short_m}|{zh_m})'
                date_pattern = y_pattern + '年的?' + m_pattern + '月?'
                match = re.search(date_pattern, question)
                if match:
                    start_id, end_id = match.start(), match.end()
                    add_value_from_char_idx((start_id, end_id), question_toks, sqlvalue, entry)
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
            if 0 < val < 1 and is_int(val_10):
                rmb = int(val_10)
                rmb_pattern.append('[' + str(rmb) + DIGIT_ALIAS(rmb) + '][毛角]')
            elif 0 < val < 1 and is_int(val_100):
                zh_val_10 = ZH_NUM2WORD(val_10, 'low')
                rmb_pattern.append(str(val_10).replace('.', '点') + '[毛角]')
                rmb_pattern.append(zh_val_10.replace('点', '[点毛角]').replace('二', '[两二]') + '[毛角]?')
            if len(rmb_pattern) == 0: return False
            rmb_pattern = '(' + '|'.join(rmb_pattern) + ')'
            match = re.search(rmb_pattern, question)
            if match:
                start_id, end_id = match.start(), match.end()
                add_value_from_char_idx((start_id, end_id), question_toks, sqlvalue, entry)
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
        else: pass
            # raise ValueError('[ERROR]: Unrecognized float value %s' % (num))
        return result

    def extract_string_val(self, val, result, sqlvalue):
        values, question_toks, entry = result['values'], result['question_toks'], result['entry']
        question = ''.join(question_toks)

        def transform_number(val): # transform number in val into chinese chars
            match = re.search(r'[0-9]+', val)
            if match and len(match.group(0)) < len(val):
                try:
                    span = match.group(0)
                    num = int(float(span)) if is_int(span) else float(span)
                    s = ZH_NUM2WORD(num, 'low')
                    val_num = re.sub(span, s, val)
                    return val_num
                except: pass
            return val

        val_num = transform_number(val)

        def parse_reserved(val):
            if val in ['是', '√', '否', '否no']:
                if val in ['是', '√']:
                    add_value_from_reserved('是', sqlvalue)
                else: add_value_from_reserved('否', sqlvalue)
            elif val in ['无', '不限', '没有限制', '无限制', '不限制', '不限专业', '专业不限', '规格不限', '/', '不适用', '未检出']:
                add_value_from_reserved('无', sqlvalue)
            else: return False
            return True

        def remove_bracket(val):
            val = re.sub(r'[\(（].*?[\)）]', '', val)
            if val in question: # contents out of the brackets are more important
                start_id = question.index(val)
                add_value_from_char_idx((start_id, start_id + len(val)), question_toks, sqlvalue, entry)
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

        def item_and_above_item(val):
            match = re.search(r'(.*?)及以上', val)
            if match:
                pattern = r'(%s.{1,4}%s[以之]上|(至少|最低|不止|高于)%s)' % (match.group(1), match.group(1), match.group(1))
                match_obj = re.search(pattern, question)
                if match_obj:
                    start_id = question.index(match_obj.group(0))
                    add_value_from_char_idx((start_id, start_id + match_obj.end(0) - match_obj.start(0)), question_toks, sqlvalue, entry)
                    return True
            return False

        def ignore_suffix(val):
            match = re.search(r'(.*?)(出版社|支行|培训中心)', val)
            if match:
                val = match.group(1)
                if val in question:
                    start_id = question.index(val)
                    add_value_from_char_idx((start_id, start_id + len(val)), question_toks, sqlvalue, entry)
                    return True
            return False

        def reorder_words(val):
            toks = self.nlp(val)
            if len(toks) <= 3:
                val = ''.join(toks[::-1])
                if val in question:
                    start_id = question.index(val)
                    add_value_from_char_idx((start_id, start_id + len(val)), question_toks, sqlvalue, entry)
                    return True
            return False

        def parse_date(val):
            if val.count('-') > 2: return False
            match = re.search(r'(\d{4})[/\-年\.](\d{1,2})[/\-月\.](\d{1,2})[日号]?', val) # year-month-day
            if match:
                y, m, d = match.group(1), match.group(2), match.group(3)
                short_y, short_m, short_d = y[-2:], m.lstrip('0'), d.lstrip('0')
                long_zh_y = ZH_NUM2WORD(y, 'direct')
                zh_y, zh_m, zh_d = ZH_NUM2WORD(short_y, 'direct'), ZH_NUM2WORD(short_m, 'low'), ZH_NUM2WORD(short_d, 'low')
                pattern = fr'(({y}|{short_y}|{long_zh_y}|{zh_y}).+?({m}|{short_m}|{zh_m}).+?({d}|{short_d}|{zh_d})|({y}|{short_y}|{long_zh_y}|{zh_y}).+?({m}|{short_m}|{zh_m}).)'
                match = re.search(pattern, question)
                if match:
                    start_id, end_id = match.start(), match.end()
                    add_value_from_char_idx((start_id, end_id), question_toks, sqlvalue, entry)
                    return True
            match = re.search(r'(\d{4})[/\-年\.](\d{1,2})[/\-月\.]?', val) # year-month
            if match:
                y, m = match.group(1), match.group(2)
                short_y, short_m = y[-2:], m.lstrip('0')
                long_zh_y = ZH_NUM2WORD(y, 'direct')
                zh_y, zh_m = ZH_NUM2WORD(short_y, 'direct'), ZH_NUM2WORD(short_m, 'low')
                pattern = fr'({y}|{short_y}|{long_zh_y}|{zh_y})年的?第?({m}|{short_m}|{zh_m})个?月?'
                match = re.search(pattern, question)
                if match:
                    start_id, end_id = match.start(), match.end()
                    add_value_from_char_idx((start_id, end_id), question_toks, sqlvalue, entry)
                    return True
            match = re.search(r'(\d{1,2})月(\d{1,2})[日号]', val) # month-day
            if match:
                m, d = match.group(1), match.group(2)
                short_m, short_d = m.lstrip('0'), d.lstrip('0')
                zh_m, zh_d = ZH_NUM2WORD(short_m, 'low'), ZH_NUM2WORD(short_d, 'low')
                pattern = fr'({m}|{short_m}|{zh_m})月的?({d}|{short_d}|{zh_d})[日号]?'
                match = re.search(pattern, question)
                if match:
                    start_id, end_id = match.start(), match.end()
                    add_value_from_char_idx((start_id, end_id), question_toks, sqlvalue, entry)
                    return True
            match = re.search(r'(\d{4}|\d{2})年', val) # 94年
            if match:
                y = match.group(1)
                short_y = y[-2:]
                zh_y, zh_short_y = ZH_NUM2WORD(y, 'direct'), ZH_NUM2WORD(short_y, 'direct')
                pattern = fr'({y}|{short_y}|{zh_y}|{zh_short_y})年'
                match = re.search(pattern, question)
                if match:
                    start_id, end_id = match.start(), match.end()
                    add_value_from_char_idx((start_id, end_id), question_toks, sqlvalue, entry)
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

        def parse_substring(val):
            pos = search_for_longest_substring(question, val, ignore_words='的')
            if pos[1] - pos[0] >= 2:
                add_value_from_char_idx((pos[0], pos[1]), question_toks, sqlvalue, entry)
                return True
            return False

        def english_chinese_transform(val):
            if val in EN2ZH_NE: val = EN2ZH_NE[val]
            if val in ZH_NE: val = ZH2EN_NE[val].replace(' ', '')
            if val in question:
                start_id = question.index(val)
                add_value_from_char_idx((start_id, start_id + len(val)), question_toks, sqlvalue, entry)
                return True
            return False

        if val in question:
            start_id = question.index(val)
            add_value_from_char_idx((start_id, start_id + len(val)), question_toks, sqlvalue, entry)
        elif val_num in question:
            start_id = question.index(val_num)
            add_value_from_char_idx((start_id, start_id + len(val_num)), question_toks, sqlvalue, entry)
        elif parse_date(val): pass
        elif parse_reserved(val): pass
        elif remove_bracket(val): pass
        elif item_and_above_item(val): pass
        elif reorder_words(val): pass
        elif ignore_suffix(val): pass
        elif try_abbreviation_mapping(val): pass
        elif fuzzy_match(val): pass
        elif parse_substring(val): pass
        elif self.use_extracted_values(sqlvalue, values, fuzzy=True): pass
        elif english_chinese_transform(val): pass
        else:
            pass
            # raise ValueError('[ERROR]: Unrecognized string value %s' % (val))
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