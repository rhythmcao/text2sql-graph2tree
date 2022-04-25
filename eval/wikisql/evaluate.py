#!/usr/bin/env python
import json, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from argparse import ArgumentParser
from tqdm import tqdm
from eval.lib.dbengine import DBEngine
from eval.lib.query import Query
from eval.lib.common import count_lines

def evaluate(source_file, db_file, pred_file, ordered=False):
    engine = DBEngine(db_file)
    result = {'exec': [], 'exact': [], 'sel': [], 'agg': [], 'conds': []}
    with open(source_file) as fs, open(pred_file) as fp:
        for ls, lp in tqdm(zip(fs, fp), total=count_lines(source_file)):
            eg = json.loads(ls)
            ep = json.loads(lp)
            qg = Query.from_dict(eg['sql'], ordered=ordered)
            gold = engine.execute_query(eg['table_id'], qg, lower=True)
            pred = ep.get('error', None)
            qp = None
            if not ep.get('error', None):
                try:
                    qp = Query.from_dict(ep, ordered=ordered)
                    pred = engine.execute_query(eg['table_id'], qp, lower=True)
                except Exception as e:
                    pred = repr(e)
            correct = pred == gold
            if not correct:
                print('Error Table id: %s' % (eg['table_id']))
                print('Gold: %s' % (json.dumps(eg, ensure_ascii=False)))
                print('Pred: %s\n' % (json.dumps(ep, ensure_ascii=False)))

            match = qp == qg
            sel = qp.sel_index == qp.sel_index
            agg = qp.agg_index == qg.agg_index
            if ordered: cond = [(col, op, str(cond).lower()) for col, op, cond in qp.conditions] == [(col, op, str(cond).lower()) for col, op, cond in qg.conditions]
            else: cond = set([(col, op, str(cond).lower()) for col, op, cond in qp.conditions]) == set([(col, op, str(cond).lower()) for col, op, cond in qg.conditions])

            result['exec'].append(correct)
            result['exact'].append(match)
            result['sel'].append(sel)
            result['agg'].append(agg)
            result['conds'].append(cond)

        for k in result:
            result[k] = sum(result[k]) / len(result[k])
            print(k, ': %.4f' % (result[k]))

    return result

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('source_file', help='source file for the prediction')
    parser.add_argument('db_file', help='source database for the prediction')
    parser.add_argument('pred_file', help='predictions by the model')
    parser.add_argument('--ordered', action='store_true', help='whether the exact match should consider the order of conditions')
    args = parser.parse_args()

    evaluate(args.source_file, args.db_file, args.pred_file, ordered=args.ordered)