#coding=utf8
import sys, os, json, argparse, time, torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from argparse import Namespace
from preprocess.process_input import get_input_processor, process_tables, process_dataset_input
from preprocess.spider.fix_error import amend_primary_keys, amend_foreign_keys, amend_boolean_types
from utils.spider.example import Example
from utils.spider.batch import Batch
from model.model_utils import Registrable
from model.model_constructor import *

def preprocess_database_and_dataset(db_dir='database/', table_path='data/tables.json', dataset_path='data/dev.json', method='lgesql'):
    tables = json.load(open(table_path, 'r'))
    # tables = amend_primary_keys(tables)
    # tables = amend_foreign_keys(tables)
    # tables = amend_boolean_types(tables, db_dir)
    dataset = json.load(open(dataset_path, 'r'))
    processor = get_input_processor(dataset='spider', method=method, db_dir=db_dir, db_content=True, bridge=True)
    output_tables = process_tables(processor, tables)
    output_dataset = process_dataset_input(processor, dataset, output_tables)
    return output_dataset, output_tables

def load_examples(dataset, tables):
    return [Example(ex, tables[ex['db_id']]) for ex in dataset]

parser = argparse.ArgumentParser()
parser.add_argument('--db_dir', default='database', help='path to db dir')
parser.add_argument('--table_path', default='data/tables.json', help='path to tables json file')
parser.add_argument('--dataset_path', default='data/dev.json', help='path to raw dataset json file')
parser.add_argument('--saved_model', default='saved_models/glove42B', help='path to saved model path, at least contain param.json and model.bin')
parser.add_argument('--output_path', default='predicted_sql.txt', help='output predicted sql file')
parser.add_argument('--batch_size', default=20, type=int, help='batch size for evaluation')
parser.add_argument('--beam_size', default=5, type=int, help='beam search size')
parser.add_argument('--order_method', default='controller', choices=['all', 'controller'], help='order method for evaluation')
parser.add_argument('--use_gpu', action='store_true', help='whether use gpu')
args = parser.parse_args(sys.argv[1:])

params = json.load(open(os.path.join(args.saved_model, 'params.json'), 'r'), object_hook=lambda d: Namespace(**d))
params.lazy_load = True # load PLM from AutoConfig instead of AutoModel.from_pretrained(...)
dataset, tables = preprocess_database_and_dataset(db_dir=args.db_dir, table_path=args.table_path, dataset_path=args.dataset_path, method=params.model)
Example.configuration(plm=params.plm, method=params.model, tables=tables, table_path=args.table_path, db_dir=args.db_dir, order_path=args.saved_model)
dataset = load_examples(dataset, tables)

device = torch.device("cuda:0") if torch.cuda.is_available() and args.use_gpu else torch.device("cpu")
model = Registrable.by_name('text2sql')(params, Example.trans).to(device)
check_point = torch.load(open(os.path.join(args.saved_model, 'model.bin'), 'rb'), map_location=device)['model']
if list(check_point.keys())[0].startswith('module.'):
    check_point, old = {}, check_point
    for k in old:
        new_k = k[len('module.'):]
        check_point[new_k] = old[k]
model.load_state_dict(check_point)

start_time = time.time()
print('Start evaluating ...')
model.eval()
all_hyps, all_vals = [], []
with torch.no_grad():
    for i in range(0, len(dataset), args.batch_size):
        current_batch = Batch.from_example_list(dataset[i: i + args.batch_size], device, train=False)
        hyps, vals, _ = model.parse(current_batch, args.beam_size, order_method=args.order_method)
        all_hyps.extend(hyps)
        all_vals.extend(vals)

print('Start writing predicted sqls to file %s' % (args.output_path))
with open(args.output_path, 'w', encoding='utf8') as of:
    evaluator = Example.evaluator
    for idx in range(len(dataset)):
        pred_sql = evaluator.obtain_sql(all_hyps[idx], dataset[idx].db, all_vals[idx], checker=True)
        of.write(pred_sql + '\n')
print('Evaluation costs %.4fs .' % (time.time() - start_time))
