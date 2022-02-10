#coding=utf8
import sys, os, gc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.example import Example
from utils.batch import Batch
from model.model_constructor import *
from torch.utils.data import DataLoader


def decode(model, evaluator, dataset, output_path, batch_size=64, beam_size=5, ts_order='controller',
        acc_type='sql', etype='exec', value_metric=None, schema_metric=None, device=None):
    assert acc_type in ['sql', 'sketch', 'beam', 'eg-sql'] and etype in ['match', 'exec', 'all']
    model.eval()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=Example.collate_fn)
    results, all_hyps, all_vals, all_gates = {'sql': None, 'value': None, 'schema': None}, [], [], []
    with torch.no_grad():
        for cur_batch in data_loader:
            cur_batch = Batch.from_example_list(cur_batch, device, train=False)
            hyps, vals, gates = model.parse(cur_batch, beam_size=beam_size, ts_order=ts_order)
            all_hyps.extend(hyps)
            all_vals.extend(vals)
            if schema_metric:
                all_gates.append(gates.cpu())
        results['sql'] = evaluator.acc(all_hyps, all_vals, dataset, output_path, acc_type=acc_type, etype=etype)
        if value_metric:
            results['value'] = evaluator.value_metric(all_vals, dataset, metric=value_metric) if Example.predict_value else 0.
        if schema_metric:
            results['schema'] = evaluator.schema_metric(all_gates, dataset, metric=schema_metric)
    torch.cuda.empty_cache()
    gc.collect()
    return results


def gather_ts_order_from_dataset(model, controller, dataset, order_path, batch_size=64, beam_size=5, device=None):
    model.train()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=Example.collate_fn)
    with torch.no_grad():
        for cur_batch in data_loader:
            cur_batch = Batch.from_example_list(cur_batch, device, train=True, ts_order='enum', uts_order='enum')
            outputs = model(cur_batch, sample_size=0, gtl_size=beam_size, n_best=1, ts_order='enum', uts_order='enum')
            controller.record_ast_generation_order(outputs['hyps'], cur_batch.ids, epoch=-1)
        canonical_ts_order = controller.accumulate_canonical_ts_order(reference_epoch=-1, save_path=order_path, verbose=True)
        controller.load_and_set_ts_order(canonical_ts_order)
    torch.cuda.empty_cache()
    gc.collect()
    return canonical_ts_order