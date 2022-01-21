#coding=utf8
import sys, os, time, json, gc
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from argparse import Namespace
from utils.args import init_args
from utils.hyperparams import hyperparam_path_from_pretrained
from utils.initialization import *
from utils.dusql.example import Example
from utils.dusql.batch import Batch
from utils.optimization import set_optimizer
from model.model_utils import Registrable
from model.model_constructor import *

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
params_path = args.read_model_path if args.read_model_path else args.read_pretrained_model_path
# reinforcement learning needs pretraining
params = json.load(open(os.path.join(params_path, 'params.json')), object_hook=lambda d: Namespace(**d))
params.lazy_load = True

exp_path = hyperparam_path_from_pretrained(args)
logger = set_logger(exp_path, args.testing)
set_random_seed(args.seed)
device = set_torch_device(args.device)
logger.info("Initialization finished ...")
logger.info("Output path is %s" % (exp_path))
logger.info("Random seed is set to %d" % (args.seed))
logger.info("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

# load dataset and vocabulary
start_time = time.time()
# set up the grammar, transition system, evaluator, etc.
Example.configuration(plm=params.plm, method=params.model, db_dir=args.db_dir)#, order_path=params.read_order_path)
train_dataset, dev_dataset = Example.load_dataset('train', params.order_seed, True), Example.load_dataset('dev')
logger.info("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
logger.info("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))
sql_trans, evaluator = Example.trans, Example.evaluator
args.word_vocab, args.relation_num = len(Example.word_vocab), len(Example.relation_vocab)

# model init
model = Registrable.by_name('text2sql')(params, sql_trans).to(device)
check_point = torch.load(open(os.path.join(params_path, 'model.bin'), 'rb'), map_location=device)
model.load_state_dict(check_point['model'])
logger.info("Load saved model from path: %s" % (params_path))
if not args.read_model_path:
    json.dump(vars(params), open(os.path.join(exp_path, 'params.json'), 'w'), indent=4)
# logger.info(str(model))

def decode(choice, output_path, acc_type='sql', value_fscore=False, schema_acc=False, order_method='controller'):
    assert acc_type in ['sql', 'beam', 'eg-sql'] and choice in ['train', 'dev']
    model.eval()
    dataset = train_dataset if choice == 'train' else dev_dataset
    all_hyps, all_vals, all_gates = [], [], []
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            current_batch = Batch.from_example_list(dataset[i: i + args.batch_size], device, train=False)
            hyps, vals, gates = model.parse(current_batch, args.beam_size, order_method=order_method)
            all_hyps.extend(hyps)
            all_vals.extend(vals)
            if schema_acc:
                all_gates.append(gates.cpu())
        acc = evaluator.acc(all_hyps, all_vals, dataset, output_path, acc_type=acc_type)
        if value_fscore:
            value_fscore = evaluator.value_fscore(all_vals, dataset)
        if schema_acc:
            schema_acc = evaluator.schema_acc(all_gates, dataset)
    torch.cuda.empty_cache()
    gc.collect()
    if value_fscore and schema_acc:
        return acc, value_fscore, schema_acc
    elif value_fscore:
        return acc, value_fscore
    elif schema_acc:
        return acc, schema_acc
    else:
        return acc

if not args.testing:
    num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    warmup_epoch = int(args.max_epoch * args.warmup_ratio)
    logger.info('Total training steps: %d;\t Warmup steps: %d' % (num_training_steps, num_warmup_steps))
    optimizer, scheduler = set_optimizer(model, args, num_warmup_steps, num_training_steps)
    start_epoch, nsamples, best_result = 0, len(train_dataset), {'dev_acc': 0.}
    train_index, step_size = np.arange(nsamples), args.batch_size // args.grad_accumulate
    if args.read_model_path and args.load_optimizer:
        optimizer.load_state_dict(check_point['optim'])
        scheduler.load_state_dict(check_point['scheduler'])
        start_epoch = check_point['epoch'] + 1
    logger.info('Start training ......')
    if args.supervised:
        sup_train_index = np.arange(nsamples)
    for i in range(start_epoch, args.max_epoch):
        start_time = time.time()
        epoch_rl_loss, epoch_loss, epoch_vr_loss, epoch_gp_loss, count = 0, 0, 0, 0, 0
        np.random.shuffle(train_index)
        if args.supervised:
            np.random.shuffle(sup_train_index)
        model.train()
        for j in range(0, nsamples, step_size):
            count += 1
            cur_dataset = [train_dataset[k] for k in train_index[j: j + step_size]]
            current_batch = Batch.from_example_list(cur_dataset, device, train=True, method='rl', smoothing=args.smoothing)
            hyps, _, _ = model.parse(current_batch, args.rl_size, order_method='controller')
            rewards = sql_trans.compute_reward(hyps, current_batch.asts, args.rl_method)
            if rewards != []:
                loss = - torch.stack(rewards).sum()
                epoch_rl_loss += loss.item()
                (args.lam * loss).backward()

            if args.supervised:
                cur_dataset = [train_dataset[k] for k in sup_train_index[j: j + step_size]]
                current_batch = Batch.from_example_list(cur_dataset, device, train=True, smoothing=args.smoothing)
                sample_size = 0 if i < warmup_epoch else 1 + args.beam_size * (i - warmup_epoch) // (args.max_epoch - warmup_epoch)
                loss, vr_loss, gp_loss = model(current_batch, sample_size, method='fixed') # see utils/batch.py for batch elements
                epoch_loss += loss.item()
                epoch_vr_loss += vr_loss.item()
                epoch_gp_loss += gp_loss.item()
                loss = loss + vr_loss + gp_loss
                loss.backward()

            if count == args.grad_accumulate or j + step_size >= nsamples:
                count = 0
                model.pad_embedding_grad_zero()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        logger.info('Training: \tEpoch: %d\tTime: %.4f\tTraining loss: %.4f/%.4f/%.4f/%.4f' % (i, time.time() - start_time, epoch_rl_loss, epoch_loss, epoch_vr_loss, epoch_gp_loss))
        torch.cuda.empty_cache()
        gc.collect()

        # fine-tune: need evaluation after each epoch
        start_time = time.time()
        dev_acc = decode('dev', os.path.join(exp_path, 'dev.iter' + str(i)), acc_type='sql', order_method='controller')
        logger.info('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.4f' % (i, time.time() - start_time, dev_acc))
        if dev_acc > best_result['dev_acc']:
            best_result['dev_acc'], best_result['iter'] = dev_acc, i
            torch.save({
                'epoch': i, 'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, open(os.path.join(exp_path, 'model.bin'), 'wb'))
            logger.info('NEW BEST MODEL: \tEpoch: %d\tDev acc: %.4f' % (i, dev_acc))

    logger.info('FINAL BEST RESULT: \tEpoch: %d\tDev acc: %.4f' % (best_result['iter'], best_result['dev_acc']))
else:
    logger.info('Start evaluating ......')
    start_time = time.time()
    acc, value_fscore, schema_acc = decode('dev', output_path=os.path.join(args.read_model_path, 'dev.eval' + str(args.beam_size)), acc_type='eg-sql', value_fscore=True, schema_acc=True)
    logger.info("Evaluation costs %.2fs ; Dev dataset execution execution-guided acc is %.4f ." % (time.time() - start_time, acc))
    logger.info("Evaluation costs %.2fs ; Dev dataset value recognition fscore/graph pruning acc is %.4f/%.4f ." % (time.time() - start_time, value_fscore, schema_acc))
    # start_time = time.time()
    # ex_acc_beam = decode('dev', output_path=os.path.join(args.read_model_path, 'dev.eval.beam' + str(args.beam_size)), acc_type='beam')
    # logger.info("Evaluation costs %.2fs ; Dev dataset execution beam acc (provided any one in the beam is correct) is %.4f ." % (time.time() - start_time, ex_acc_beam))
