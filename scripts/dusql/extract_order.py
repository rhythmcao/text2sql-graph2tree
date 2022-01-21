#coding=utf8
import sys, os, time, json, pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from argparse import Namespace
from utils.args import init_args
from utils.initialization import *
from utils.dusql.example import Example
from utils.dusql.batch import Batch
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

train_dataset = Example.load_dataset('train', shuffle=False)
logger.info("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
logger.info("Dataset size: train -> %d" % (len(train_dataset)))
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

def generate_new_dataset(output_path):
    model.train()
    all_actions = []
    with torch.no_grad():
        for i in range(0, len(train_dataset), args.batch_size):
            current_batch = Batch.from_example_list(train_dataset[i: i + args.batch_size], device, train=True, method='gtol')
            (_, hyps), _, _ = model(current_batch, sample_size=0, gtol_size=args.beam_size, n_best=1, order_method='controller', method='gtol')
            actions = [list(hyp[0].actions) for hyp in hyps]
            all_actions.extend(actions)
    assert len(all_actions) == len(train_dataset)
    new_dataset = []
    for ex, actions in zip(train_dataset, all_actions):
        ex = ex.ex
        ex['actions'] = actions
        new_dataset.append(ex)
    pickle.dump(new_dataset, open(output_path, 'wb'))
    logger.info('New dataset written to path:', output_path)

logger.info('Start traversing the train dataset ......')
start_time = time.time()
gather_orders_from_asts(output_path=os.path.join(args.read_model_path, 'order.bin'))
# generate_new_dataset(output_path=os.path.join(args.read_model_path, 'train_best.bin'))
logger.info("Traversing costs %.2fs ;" % (time.time() - start_time))
