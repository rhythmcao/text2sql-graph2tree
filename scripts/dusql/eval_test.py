#coding=utf8
import sys, os, json, pickle, argparse, time, torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from argparse import Namespace
import numpy as np
from utils.constants import DEBUG
from utils.dusql.example import Example
from utils.dusql.batch import Batch
from model.model_utils import Registrable
from model.model_constructor import *

def load_test_dataset(dataset, tables, order_seed=1024, shuffle=False):
    state = np.random.get_state()
    np.random.seed(order_seed)
    if shuffle:
        Example.grammar.order_controller.shuffle_order()
    raw_dataset = pickle.load(open(dataset, 'rb'))
    dataset = [Example(ex, tables[ex['db_id']]) for ex in raw_dataset]
    np.random.set_state(state)
    return dataset

parser = argparse.ArgumentParser()
parser.add_argument('--db_dir', default='data/dusql/db_content.json', help='path to db content file')
parser.add_argument('--raw_table_path', default='data/dusql/tables.json', help='path to raw table json file')
parser.add_argument('--table_path', default='data/dusql/tables.bin', help='path to tables file')
parser.add_argument('--dataset_path', default='data/dusql/test.bin', help='path to dataset file')
parser.add_argument('--saved_model', default='saved_models/chinese-macbert-base', help='path to saved model path, at least contain param.json and model.bin')
parser.add_argument('--output_path', default='saved_models/chinese-macbert-base/predicted_sql.txt', help='output predicted sql file')
parser.add_argument('--batch_size', default=20, type=int, help='batch size for evaluation')
parser.add_argument('--beam_size', default=5, type=int, help='beam search size')
parser.add_argument('--use_gpu', action='store_true', help='whether use gpu')
args = parser.parse_args(sys.argv[1:])

assert not DEBUG
params = json.load(open(os.path.join(args.saved_model, 'params.json'), 'r'), object_hook=lambda d: Namespace(**d))
params.lazy_load = True # load PLM from AutoConfig instead of AutoModel.from_pretrained(...)
# order_path = params.read_order_path if hasattr(params, 'read_order_path') else None
Example.configuration(plm=params.plm, method=params.model, raw_table_path=args.raw_table_path, table_path=args.table_path, db_dir=args.db_dir, order_path=args.saved_model)
dataset = load_test_dataset(args.dataset_path, Example.tables, order_seed=params.order_seed, shuffle=False)

device = torch.device("cuda:0") if torch.cuda.is_available() and args.use_gpu else torch.device("cpu")
model = Registrable.by_name('text2sql')(params, Example.trans).to(device)
check_point = torch.load(open(os.path.join(args.saved_model, 'model.bin'), 'rb'), map_location=device)
if list(check_point['model'].keys())[0].startswith('module.'):
    # remove unnecessary module prefix for parameter loading
    param_dict = {}
    for k in check_point['model']:
        v = check_point['model'][k]
        k = k[len('module.'):]
        param_dict[k] = v
    model.load_state_dict(param_dict)
else:
    model.load_state_dict(check_point['model'])

start_time = time.time()
print('Start evaluating ...')
model.eval()
all_hyps, all_vals = [], []
with torch.no_grad():
    for i in range(0, len(dataset), args.batch_size):
        current_batch = Batch.from_example_list(dataset[i: i + args.batch_size], device, train=False)
        hyps, vals, _ = model.parse(current_batch, args.beam_size, order_method='controller')
        all_hyps.extend(hyps)
        all_vals.extend(vals)

print('Start writing predicted sqls to file %s' % (args.output_path))
with open(args.output_path, 'w', encoding='utf8') as of:
    evaluator = Example.evaluator
    for idx in range(len(dataset)):
        pred_sql = evaluator.obtain_sql(all_hyps[idx], dataset[idx].db, all_vals[idx], dataset[idx].ex, checker=True)
        of.write(dataset[idx].ex['question_id'] + '\t' + pred_sql + '\n')
print('Evaluation costs %.4fs .' % (time.time() - start_time))
