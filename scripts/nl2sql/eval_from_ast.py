#coding=utf8
import sys, os, json, argparse, time, torch, pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from argparse import Namespace
from utils.example import Example

parser = argparse.ArgumentParser()
parser.add_argument('--read_model_path', type=str, required=True, help='path to saved model, at least containing model.bin, params.json, order.bin')
parser.add_argument('--output_file', default='nl2sql.sql', help='output predicted sql file')
args = parser.parse_args(sys.argv[1:])

# Example configuration
Example.configuration('nl2sql', plm='chinese-macbert-large', encode_method='lgesql', ts_order_path=os.path.join(args.read_model_path, 'order.bin'))
# load test dataset
dataset = Example.load_dataset('test')

start_time = time.time()
all_hyps = pickle.load(open(os.path.join(args.read_model_path, 'hyps.ast'), 'rb'))
all_vals = pickle.load(open(os.path.join(args.read_model_path, 'hyps.val'), 'rb'))

output_path = os.path.join(args.read_model_path, args.output_file)
print('Start writing predicted sqls to file %s' % (output_path))
with open(output_path, 'w', encoding='utf8') as of:
    evaluator = Example.evaluator
    for idx in range(len(dataset)):
        # execution-guided unparsing
        pred_sql = evaluator.obtain_sql(all_hyps[idx], dataset[idx].db, all_vals[idx], dataset[idx].ex, checker=True)
        of.write(dataset[idx].ex['question_id'] + '\t' + pred_sql + '\n')
print('Evaluation costs %.4fs .' % (time.time() - start_time))
