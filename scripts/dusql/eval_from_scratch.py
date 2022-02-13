#coding=utf8
import sys, os, json, argparse, time, torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from argparse import Namespace
from torch.utils.data import DataLoader
from utils.initialization import set_torce_device
from utils.constants import DEBUG, DATASETS
from utils.example import Example
from utils.batch import Batch
from model.model_utils import Registrable
from model.model_constructor import *

parser = argparse.ArgumentParser()
parser.add_argument('--read_model_path', type=str, required=True, help='path to saved model, at least containing model.bin, params.json, order.bin')
parser.add_argument('--output_file', default='dusql.sql', help='output predicted sql file')
parser.add_argument('--batch_size', default=20, type=int, help='batch size for evaluation')
parser.add_argument('--beam_size', default=5, type=int, help='beam search size')
parser.add_argument('--ts_order', choices=['controller', 'enum'], default='controller', help='input node selection method')
parser.add_argument('--deviceId', type=int, default=0, help='-1 -> CPU ; GPU index o.w.')
args = parser.parse_args(sys.argv[1:])

assert not DEBUG
# load model params
params = json.load(open(os.path.join(args.read_model_path, 'params.json'), 'r'), object_hook=lambda d: Namespace(**d))
params.lazy_load = True # load PLM from AutoConfig instead of AutoModel.from_pretrained(...)
# Example configuration
Example.configuration('dusql', plm=params.plm, encode_method=params.encode_method, ts_order_path=os.path.join(args.read_model_path, 'order.bin'))
# load test dataset
dataset = Example.load_dataset('test')
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=Example.collate_fn)
# set torch device
device = set_torch_device(args.deviceId)
# model init
model = Registrable.by_name('text2sql')(params, Example.trans).to(device)
check_point = torch.load(open(os.path.join(args.read_model_path, 'model.bin'), 'rb'), map_location=device)
model.load_state_dict(check_point['model'])

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

output_path = os.path.join(args.read_model_path, args.output_file)
print('Start writing predicted sqls to file %s' % (output_path))
with open(output_path, 'w', encoding='utf8') as of:
    evaluator = Example.evaluator
    for idx in range(len(dataset)):
        # execution-guided unparsing
        pred_sql = evaluator.obtain_sql(all_hyps[idx], dataset[idx].db, all_vals[idx], dataset[idx].ex, checker=True)
        of.write(dataset[idx].ex['question_id'] + '\t' + pred_sql + '\n')
print('Evaluation costs %.4fs .' % (time.time() - start_time))
