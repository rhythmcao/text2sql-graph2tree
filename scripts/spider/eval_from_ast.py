#coding=utf8
import sys, os, argparse, time, pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.example import Example

parser = argparse.ArgumentParser()
parser.add_argument('--read_model_path', type=str, required=True, help='path to saved model, at least containing model.bin, params.json, order.bin')
parser.add_argument('--db_dir', default='data/database', help='path to db dir')
parser.add_argument('--table_path', default='data/tables.json', help='path to tables json file')
parser.add_argument('--output_path', default='predicted_sql.txt', help='output predicted sql file')
args = parser.parse_args(sys.argv[1:])

# Example configuration
tables = pickle.load(open(os.path.join(args.read_model_path, 'tables.bin'), 'rb'))
Example.configuration('spider', plm='electra-large-discriminator', encode_method='lgesql', tables=tables, table_path=args.table_path, db_dir=args.db_dir, ts_order_path=os.path.join(args.read_model_path, 'order.bin'))

start_time = time.time()
dataset = pickle.load(open(os.path.join(args.read_model_path, 'dataset.bin'), 'rb'))
all_hyps = pickle.load(open(os.path.join(args.read_model_path, 'hyps.ast'), 'rb'))
all_vals = pickle.load(open(os.path.join(args.read_model_path, 'hyps.val'), 'rb'))

output_path = args.output_path
print('Start writing predicted sqls to file %s' % (output_path))
with open(output_path, 'w', encoding='utf8') as of:
    evaluator = Example.evaluator
    for idx in range(len(dataset)):
        # execution-guided unparsing
        pred_sql = evaluator.obtain_sql(all_hyps[idx], dataset[idx].db, all_vals[idx], dataset[idx].ex, checker=True)
        of.write(pred_sql + '\n')
print('Evaluation costs %.4fs .' % (time.time() - start_time))