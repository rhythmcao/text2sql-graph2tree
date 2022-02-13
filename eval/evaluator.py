#coding=utf8
import json, pickle
import torch
from functools import partial


def build_foreign_key_map(entry):
    """
    Args:
    Returns:
    """
    cols_orig = entry.get("column_names_original", "column_names")
    tables_orig = entry.get("table_names_original", "table_names")

    # rebuild cols corresponding to idmap in Schema
    cols = []
    for col_orig in cols_orig:
        if col_orig[0] >= 0:
            t = tables_orig[col_orig[0]]
            c = col_orig[1]
            cols.append("__" + t.lower() + "." + c.lower() + "__")
        else:
            cols.append("__all__")

    def keyset_in_list(k1, k2, k_list):
        """keyset_in_list"""
        for k_set in k_list:
            if k1 in k_set or k2 in k_set:
                return k_set
        new_k_set = set()
        k_list.append(new_k_set)
        return new_k_set

    foreign_key_list = []
    foreign_keys = entry["foreign_keys"]
    for fkey in foreign_keys:
        key1, key2 = fkey
        key_set = keyset_in_list(key1, key2, foreign_key_list)
        key_set.add(key1)
        key_set.add(key2)

    foreign_key_map = {}
    for key_set in foreign_key_list:
        sorted_list = sorted(list(key_set))
        midx = sorted_list[0]
        for idx in sorted_list:
            foreign_key_map[cols[idx]] = cols[midx]

    return foreign_key_map


def build_foreign_key_map_from_json(table):
    """
    Args:
    Returns:
    """
    with open(table) as f:
        data = json.load(f)
    tables = {}
    for entry in data:
        tables[entry['db_id']] = build_foreign_key_map(entry)
    return tables


class Evaluator():

    def __init__(self, transition_system, table_path, db_dir):
        super(Evaluator, self).__init__()
        self.transition_system = transition_system
        self.table_path, self.db_dir = table_path, db_dir
        self.kmaps = build_foreign_key_map_from_json(self.table_path)
        self.schemas = self._load_schemas(self.table_path)
        self.engine, self.surface_checker, self.exec_checker = None, None, None
        self.acc_dict = {
            "sketch": self.sketch_acc, # ignore tables, columns and values, use special placeholders
            "sql": self.sql_acc, # directly use top of beam predicted sql
            "beam": self.beam_acc, # if the correct answer exist in the beam, assume the result is true
            "eg-sql": partial(self.sql_acc, checker=True), # execution guided, use the first sql in the beam without error
        }


    def _load_schemas(self, table_path):
        raise NotImplementedError


    def change_database(self, db_dir):
        """ Obtaining testsuite execution accuracy is too expensive if evaluation every epoch,
        thus just evaluate on testsuite database after training is finished
        """
        self.db_dir = db_dir
        self.exec_checker.db_dir = db_dir


    @classmethod
    def get_class_by_dataset(self, dataset):
        if dataset == 'spider':
            from eval.spider.evaluator import SpiderEvaluator
            return SpiderEvaluator
        elif dataset == 'dusql':
            from eval.dusql.evaluator import DuSQLEvaluator
            return DuSQLEvaluator
        elif dataset == 'cspider':
            from eval.cspider.evaluator import CSpiderEvaluator
            return CSpiderEvaluator
        elif dataset == 'cspider-raw':
            from eval.cspider_raw.evaluator import CSpiderRawEvaluator
            return CSpiderRawEvaluator
        else:
            raise ValueError(f'[Error]: Unrecognized dataset "{dataset}" for evaluator')


    def value_metric(self, vals, dataset, metric='fscore'):
        """
        @metric: by default, fscore
            fscore: dataset-level metric, compare value indexes and accumulate error items
            acc: instance-level metric, one sample is true if and only if
                all predicted value indexes are correct and complete compared to the golden indexes set
            acc-recall: instance-level metric, we pay more attention to recall and ignore precision,
                one sample is true as long as no index pair in golden labels is missing
        """
        pred = [[val.matched_index for val in candidates] for candidates in vals]
        gold = [[val.matched_index for val in sample.ex['candidates']] for sample in dataset]
        assert len(pred) == len(gold)

        tp, fp, fn, err, missing = 0, 0, 0, 0, 0
        for p, g in zip(pred, gold):
            p_fp, p_fn = fp, fn # record previous number
            for item in p:
                if item in g:
                    tp += 1
                else:
                    fp += 1
            for item in g:
                if item not in p:
                    fn += 1
            if fp > p_fp or fn > p_fn:
                err += 1
            if fn > p_fn:
                missing += 1
        acc = (len(pred) - err) / float(len(pred))
        acc_recall = (len(pred) - missing) / float(len(pred))
        fscore = 2 * float(tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0
        if metric == 'acc':
            return acc
        elif metric == 'acc-recall':
            return acc_recall
        elif metric == 'fscore':
            return fscore
        else:
            return acc, acc_recall, fscore


    def schema_metric(self, gates, dataset, metric='acc'):
        """
        @metric: by default, acc
            fscore: dataset-level metric, compare each schema item and accumulate error items
            acc: instance-level metric, one sample is true if and only if
                labels for all schema items are correct compared to the golden labels
            acc-recall: instance-level metric, we pay more attention to recall and ignore precision,
                one sample is true as long as all True schema items in golden labels are distinguished
        """
        gold = [ex.graph.schema_label for ex in dataset]
        num = [ex.size(0) for ex in gold]
        pred = torch.cat(gates, dim=0) # first concatenate all batches
        pred = pred.split(num) # then split into instances

        tp, fp_fn, correct, full_recall = 0, 0, 0, 0
        for p, g in zip(pred, gold):
            if torch.equal(p, g):
                correct += 1
            if torch.equal(g, torch.logical_and(p, g)):
                full_recall += 1
            tp += torch.logical_and(p, g).sum().item()
            fp_fn += torch.logical_xor(p, g).sum().item()

        acc = correct / float(len(pred))
        acc_recall = full_recall / float(len(pred))
        fscore = 2 * float(tp) / (2 * tp + fp_fn) if 2 * tp + fp_fn > 0 else 0.0
        if metric == 'acc':
            return acc
        elif metric == 'acc-recall':
            return acc_recall
        elif metric == 'fscore':
            return fscore
        else:
            return acc, acc_recall, fscore


    def acc(self, pred_hyps, values, dataset, output_path=None, acc_type='sql', etype='exec'):
        assert len(pred_hyps) == len(dataset) and acc_type in self.acc_dict and etype in ['match', 'exec', 'all']
        acc_method = self.acc_dict[acc_type]
        return acc_method(pred_hyps, values, dataset, output_path, etype)


    def beam_acc(self, pred_hyps, values, dataset, output_path, etype):
        scores, results = {}, []
        for each in ['easy', 'medium', 'hard', 'extra', 'all']:
            scores[each] = [0, 0.] # first is count, second is total score
        for idx, pred in enumerate(pred_hyps):
            entry = dataset[idx]
            question, gold_sql, db = entry.ex['question'], entry.query, entry.db
            gold_sql = gold_sql.replace('==', '=')
            hardness, pred_sqls = self.engine.eval_hardness(entry.ex['sql']), []
            for b_id, hyp in enumerate(pred):
                pred_sql, flag = self.transition_system.ast_to_surface_code(hyp.tree, db, values[idx], entry)
                pred_sqls.append(pred_sql)
                if not flag: continue
                score = self.evaluate_with_adaptive_interface(pred_sql, gold_sql, db['db_id'], etype)
                if int(score) == 1:
                    scores[hardness][0] += 1
                    scores[hardness][1] += 1.0
                    scores['all'][0] += 1
                    scores['all'][1] += 1.0
                    results.append((hardness, question, gold_sql, b_id, pred_sql, True))
                    break
            else:
                scores[hardness][0] += 1
                scores['all'][0] += 1
                results.append((hardness, question, gold_sql, 0, pred_sqls[0], False))
        for each in scores:
            accuracy = scores[each][1] / float(scores[each][0]) if scores[each][0] != 0 else 0.
            scores[each].append(accuracy)
        if output_path is not None:
            with open(output_path, 'w', encoding='utf8') as of:
                for item in results:
                    of.write(f'Level: {item[0]}\n')
                    of.write(f'Question: {item[1]}\n')
                    of.write(f'Gold SQL: {item[2]}\n')
                    of.write(f'Pred SQL ({item[3]}): {item[4]}\n')
                    of.write(f'Correct: {item[5]}\n\n')
                of.write(f'Overall {etype} accuracy:\n')
                for each in scores:
                    of.write(f'Level {each}: {scores[each][2]}\n')
        return scores['all'][2]


    def sql_acc(self, pred_hyps, values, dataset, output_path, etype, checker=False):
        pred_sqls, ref_sqls = [], [ex.query for ex in dataset]
        dbs = [ex.db for ex in dataset]
        for idx, hyp in enumerate(pred_hyps):
            pred_sql = self.obtain_sql(hyp, dbs[idx], values[idx], dataset[idx].ex, checker=checker)
            pred_sqls.append(pred_sql)
        result = self.evaluate_with_official_interface(pred_sqls, ref_sqls, dbs, dataset, output_path, etype)

        if etype == 'match':
            return float(result['exact'])
        elif etype == 'exec':
            return float(result['exec'])
        else:
            return float(result['exact']), float(result['exec'])


    def sketch_acc(self, pred_hyps, values, dataset, output_path, etype):
        dbs = [ex.db for ex in dataset]
        pred_sqls = [self.transition_system.ast_to_surface_code(hyp[0].tree, dbs[idx], values[idx], dataset[idx].ex, sketch=True) for idx, hyp in enumerate(pred_hyps)]
        ref_asts = [ex.ast for ex in dataset]
        ref_sqls = [self.transition_system.ast_to_surface_code(hyp, dbs[idx], values[idx], dataset[idx].ex, sketch=True) for idx, hyp in enumerate(ref_asts)]
        results = self.evaluate_with_official_interface(pred_sqls, ref_sqls, dbs, dataset, output_path=None, etype='match')
        return float(results['exact'])


    def evaluate_with_official_interface(self, *args, **kargs):
        raise NotImplementedError


    def evaluate_with_adaptive_interface(self, *args, **kargs):
        raise NotImplementedError


    def obtain_sql(self, hyps, db, vals, entry, checker=False):
        pred_sqls = []
        for hyp in hyps:
            cur_ast = hyp.tree
            pred_sql, flag = self.transition_system.ast_to_surface_code(cur_ast, db, vals, entry)
            pred_sqls.append(pred_sql)
            if not flag: continue
            if not checker: return pred_sql
            if self.surface_checker.validity_check(pred_sql, db) and self.exec_checker.validity_check(pred_sql, db):
                return pred_sql
        return pred_sqls[0]
