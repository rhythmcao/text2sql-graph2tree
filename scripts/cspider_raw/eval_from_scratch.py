#coding=utf8
import sys, os, json, argparse, time, torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from argparse import Namespace
from preprocess.process_input import get_input_processor, process_tables, process_dataset_input
from preprocess.cspider_raw.fix_error import amend_primary_keys, amend_foreign_keys, amend_boolean_types
from utils.initialization import set_torch_device
from utils.constants import DEBUG, TEST
from utils.example import Example
from utils.batch import Batch
from torch.utils.data import DataLoader
from model.model_utils import Registrable
from model.model_constructor import *

def preprocess_database_and_dataset(db_dir='database/', table_path='data/tables.json', dataset_path='data/dev.json', encode_method='lgesql'):
    tables = json.load(open(table_path, 'r'))
    # tables = amend_primary_keys(tables)
    # tables = amend_foreign_keys(tables)
    # tables = amend_boolean_types(tables, db_dir)
    dataset = json.load(open(dataset_path, 'r'))
    processor = get_input_processor(dataset='cspider_raw', encode_method=encode_method, db_dir=db_dir, db_content=True, bridge=False)
    output_tables = process_tables(processor, tables)
    output_dataset = process_dataset_input(processor, dataset, output_tables)
    return output_dataset, output_tables

parser = argparse.ArgumentParser()
parser.add_argument('--db_dir', default='data/database', help='path to db dir')
parser.add_argument('--table_path', default='data/tables.json', help='path to tables json file')
parser.add_argument('--dataset_path', default='data/dev.json', help='path to raw dataset json file')
parser.add_argument('--read_model_path', default='saved_models/glove42B', help='path to saved model path, at least contain param.json and model.bin')
parser.add_argument('--output_path', default='predicted_sql.txt', help='output predicted sql file')
parser.add_argument('--batch_size', default=20, type=int, help='batch size for evaluation')
parser.add_argument('--beam_size', default=5, type=int, help='beam search size')
parser.add_argument('--ts_order', default='controller', choices=['enum', 'controller'], help='order method for evaluation')
parser.add_argument('--deviceId', type=int, default=-1, help='-1 -> CPU ; GPU index o.w.')
args = parser.parse_args(sys.argv[1:])

assert TEST and not DEBUG
params = json.load(open(os.path.join(args.read_model_path, 'params.json'), 'r'), object_hook=lambda d: Namespace(**d))
params.lazy_load = True # load PLM from AutoConfig instead of AutoModel.from_pretrained(...)
dataset, tables = preprocess_database_and_dataset(db_dir=args.db_dir, table_path=args.table_path, dataset_path=args.dataset_path, encode_method=params.encode_method)
Example.configuration('cspider_raw', plm=params.plm, encode_method=params.encode_method, tables=tables, table_path=args.table_path, db_dir=args.db_dir, ts_order_path=os.path.join(args.read_model_path, 'order.bin'))
dataset = Example.load_dataset(dataset=dataset)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=Example.collate_fn)

device = set_torch_device(args.deviceId)
model = Registrable.by_name('text2sql')(params, Example.trans).to(device)
check_point = torch.load(open(os.path.join(args.read_model_path, 'model.bin'), 'rb'), map_location=device)['model']
model.load_state_dict(check_point)

model.eval()
start_time = time.time()
print(f'Start evaluating with {args.ts_order} method ...')
all_hyps, all_vals = [], []
with torch.no_grad():
    for cur_batch in dataloader:
        cur_batch = Batch.from_example_list(cur_batch, device, train=False)
        hyps, vals, _ = model.parse(cur_batch, args.beam_size, ts_order=args.ts_order)
        all_hyps.extend(hyps)
        all_vals.extend(vals)

print('Start writing predicted sqls to file %s' % (args.output_path))
with open(args.output_path, 'w', encoding='utf8') as of:
    evaluator = Example.evaluator
    for idx in range(len(dataset)):
        # execution-guided unparsing
        pred_sql = evaluator.obtain_sql(all_hyps[idx], dataset[idx].db, all_vals[idx], dataset[idx].ex, checker=True)
        of.write(pred_sql + '\n')
print('Evaluation costs %.4fs .' % (time.time() - start_time))
