#coding=utf8
import sys, os, time, json, gc, pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from argparse import Namespace
from utils.args import init_args
from utils.initialization import *
from utils.spider.example import Example
from utils.spider.batch import Batch
from model.model_utils import Registrable
from model.model_constructor import *

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
exp_path = args.read_model_path
logger = set_logger(exp_path, True)
set_random_seed(args.seed)
device = set_torch_device(args.device)
logger.info("Initialization finished ...")
logger.info("Output path is %s" % (exp_path))
logger.info("Random seed is set to %d" % (args.seed))
logger.info("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

# load dataset and vocabulary
start_time = time.time()
params = json.load(open(os.path.join(args.read_model_path, 'params.json')), object_hook=lambda d: Namespace(**d))
params.lazy_load = True
# set up the grammar, transition system, evaluator, etc.
read_order_path = params.read_order_path if hasattr(params, 'read_order_path') else None
Example.configuration(plm=params.plm, method=params.model, db_dir=args.db_dir, order_path=read_order_path)

if False: # for fixed order models
    state = np.random.get_state()
    np.random.seed(params.order_seed)
    Example.grammar.order_controller.shuffle_order()
    np.random.set_state(state)

dev_dataset = Example.load_dataset('dev')
logger.info("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
logger.info("Dataset size: dev -> %d" % (len(dev_dataset)))
sql_trans, evaluator = Example.trans, Example.evaluator
args.word_vocab, args.relation_num = len(Example.word_vocab), len(Example.relation_vocab)

# model init, set optimizer
if not hasattr(params, 'struct_feeding'):
    params.struct_feeding = False
if not hasattr(params, 'no_share_clsfy'):
    params.no_share_clsfy = False
model = Registrable.by_name('text2sql')(params, sql_trans).to(device)
check_point = torch.load(open(os.path.join(args.read_model_path, 'model.bin'), 'rb'), map_location=device)['model']
if list(check_point.keys())[0].startswith('module.'):
    check_point, old = {}, check_point
    for k in old:
        new_k = k[len('module.'):]
        check_point[new_k] = old[k]
model.load_state_dict(check_point)
logger.info("Load saved model from path: %s" % (args.read_model_path))
# logger.info(str(model))

def decode(output_path, order_method='controller', add_auxiliary=True):
    model.eval()
    all_hyps, all_vals, all_gates = [], [], []
    with torch.no_grad():
        for i in range(0, len(dev_dataset), args.batch_size):
            current_batch = Batch.from_example_list(dev_dataset[i: i + args.batch_size], device, train=False)
            hyps, vals, gates = model.parse(current_batch, args.beam_size, order_method=order_method)
            all_hyps.extend(hyps)
            all_vals.extend(vals)
            all_gates.append(gates.cpu())
    match_acc, exec_acc = evaluator.acc(all_hyps, all_vals, dev_dataset, output_path, acc_type='sql', etype='all')
    em, ex = evaluator.acc(all_hyps, all_vals, dev_dataset, output_path, acc_type='eg-sql', etype='all')
    if add_auxiliary:
        value_fscore = evaluator.value_fscore(all_vals, dev_dataset)
        schema_acc = evaluator.schema_acc(all_gates, dev_dataset)
        return match_acc, exec_acc, em, ex, value_fscore, schema_acc
    else:
        return match_acc, exec_acc, em, ex

logger.info('Start evaluating ......')
start_time = time.time()
match_acc, exec_acc, em_acc, ex_acc, value_fscore, schema_acc = decode(output_path=os.path.join(args.read_model_path, 'dev.eval' + str(args.beam_size)), order_method=args.order_method)
logger.info("Evaluation costs %.2fs ; Dev dataset exact match/execution acc is %.4f/%.4f ." % (time.time() - start_time, match_acc, exec_acc))
logger.info("Evaluation costs %.2fs ; Dev dataset exact match/execution execution-guided acc is %.4f/%.4f ." % (time.time() - start_time, em_acc, ex_acc))
logger.info("Evaluation costs %.2fs ; Dev dataset value recognition fscore/graph pruning acc is %.4f/%.4f ." % (time.time() - start_time, value_fscore, schema_acc))

if args.order_method == 'all':

    train_dataset = Example.load_dataset('train', shuffle=False)

    def gather_orders_from_asts(output_path):
        nsamples, step_size = len(train_dataset), args.batch_size
        results = None
        model.train()
        with torch.no_grad():
            for j in range(0, nsamples, step_size):
                current_batch = Batch.from_example_list(train_dataset[j: j + step_size], device, train=True, method='gtol')
                (_, hyps), _, _ = model(current_batch, sample_size=0, gtol_size=args.beam_size, n_best=1, order_method='all', method='gtol')
                results = sql_trans.grammar.order_controller.order_statistics(hyps, results)
        prod2fields = sql_trans.grammar.order_controller.compute_best_order(results, save_path=output_path, verbose=True)
        return prod2fields

    if os.path.exists(os.path.join(args.read_model_path, 'order.bin')):
        results = pickle.load(open(os.path.join(args.read_model_path, 'order.bin'), 'rb'))
        prod2fields = { p: list(results[p].most_common(1)[0][0]) for p in results  }
        logger.info('Load order from path: %s' % (os.path.join(args.read_model_path, 'order.bin')))
    else:
        start_time = time.time()
        prod2fields = gather_orders_from_asts(os.path.join(args.read_model_path, 'order.bin'))
        logger.info("Order accumulation costs %.2fs ; Written to: %s" % (time.time() - start_time, os.path.join(args.read_model_path, 'order.bin')))
    sql_trans.grammar.order_controller.set_order(prod2fields)
    start_time = time.time()
    match_acc, exec_acc, em_acc, ex_acc = decode(output_path=os.path.join(args.read_model_path, 'dev.best.eval'), order_method='controller', add_auxiliary=False)
    logger.info("Evaluation costs %.2fs ; Dev dataset exact match/execution acc is %.4f/%.4f ." % (time.time() - start_time, match_acc, exec_acc))
    logger.info("Evaluation costs %.2fs ; Dev dataset exact match/execution execution-guided acc is %.4f/%.4f ." % (time.time() - start_time, em_acc, ex_acc))
