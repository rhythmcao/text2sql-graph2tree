#coding=utf8
import sys, os, time, json, gc, pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from argparse import Namespace
from utils.args import init_args
from utils.hyperparams import hyperparam_path
from utils.initialization import *
from utils.dusql.example import Example, DatasetWrapper
from utils.dusql.batch import Batch
from utils.optimization import set_optimizer
from model.model_utils import Registrable
from model.model_constructor import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
exp_path = hyperparam_path(args, create=False)

# communicate with common file system
# sync_file = os.path.join(os.path.abspath(exp_path), 'synchronized')
# rank, world_size = args.rank, args.world_size
# if rank == 0 and os.path.exists(sync_file):
    # os.remove(sync_file)
# dist.init_process_group("nccl", init_method="file://" + sync_file, rank=rank, world_size=world_size)
# local_rank = args.local_rank
# device = set_torch_device_ddp(local_rank)
host_addr = get_master_node_addr()
local_rank = int(os.environ['SLURM_LOCALID'])
rank = int(os.environ['SLURM_PROCID'])
world_size = int(os.environ['SLURM_NTASKS'])
device = distributed_init(host_addr, rank, local_rank, world_size, port='2' + os.environ['SLURM_JOBID'][-4:])

def is_master():
    return rank == 0

if is_master():
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    logger = set_logger(exp_path, args.testing)
    logger.info("Initialization finished ...")
    logger.info("Output path is %s" % (exp_path))
    logger.info("Random seed is set to %d" % (args.seed))

# load dataset and vocabulary
start_time = time.time()
if args.read_model_path:
    params = json.load(open(os.path.join(args.read_model_path, 'params.json')), object_hook=lambda d: Namespace(**d))
    params.lazy_load = True
else:
    params = args
# set up the grammar, transition system, evaluator, etc.
Example.configuration(plm=params.plm, method=params.model, db_dir=params.db_dir, order_path=args.read_order_path)
train_dataset, dev_dataset = DatasetWrapper(Example.load_dataset('train', shuffle=False)), Example.load_dataset('dev')
if is_master():
    logger.info("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
    logger.info("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))
sql_trans, evaluator = Example.trans, Example.evaluator
args.word_vocab, args.relation_num = len(Example.word_vocab), len(Example.relation_vocab)

# model init, set optimizer
model = Registrable.by_name('text2sql')(params, sql_trans).to(device)
if is_master():
    # logger.info(str(model))
    if args.read_model_path:
        check_point = torch.load(open(os.path.join(args.read_model_path, 'model.bin'), 'rb'), map_location=device)
        model.load_state_dict(check_point['model'])
        logger.info("Load saved model from path: %s" % (args.read_model_path))
    else:
        json.dump(vars(params), open(os.path.join(exp_path, 'params.json'), 'w'), indent=4)
model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

def decode(choice, output_path, acc_type='sql', value_fscore=False, schema_acc=False, order_method='all'):
    assert acc_type in ['sql', 'beam', 'eg-sql'] and choice in ['train', 'dev']
    model.eval()
    dataset = train_dataset if choice == 'train' else dev_dataset
    all_hyps, all_vals, all_gates = [], [], []
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            current_batch = Batch.from_example_list(dataset[i: i + args.batch_size], device, train=False)
            hyps, vals, gates = model.module.parse(current_batch, args.beam_size, order_method=order_method)
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
    train_sampler = DistributedSampler(train_dataset)
    assert args.batch_size % (world_size * args.grad_accumulate) == 0
    batch_size = args.batch_size // (world_size * args.grad_accumulate)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, collate_fn=lambda x: list(x))
    size_per_world = (len(train_dataset) + world_size - 1) // world_size
    last_iter = (size_per_world - 1) // batch_size
    total_size = size_per_world * world_size
    num_training_steps = ((total_size + args.batch_size - 1) // args.batch_size) * args.max_epoch
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    warmup_epoch = int(args.max_epoch * args.warmup_ratio)
    optimizer, scheduler = set_optimizer(model.module, args, num_warmup_steps, num_training_steps)
    start_epoch, best_result = 0, { 'dev_acc': 0. }
    if args.read_model_path and args.load_optimizer:
        check_point = torch.load(open(os.path.join(args.read_model_path, 'model.bin'), 'rb'), map_location=device)
        optimizer.load_state_dict(check_point['optim'])
        scheduler.load_state_dict(check_point['scheduler'])
        start_epoch = check_point['epoch'] + 1

    if is_master():
        logger.info('Total training steps: %d;\t Warmup steps: %d' % (num_training_steps, num_warmup_steps))
        logger.info('Start training ......')
    all_results, all_prod2fields = None, []
    for i in range(start_epoch, args.max_epoch):
        start_time = time.time()
        train_loader.sampler.set_epoch(i)
        epoch_loss, epoch_vr_loss, epoch_gp_loss, count, prod_results = 0, 0, 0, 0, None
        model.train()
        for j, cur_dataset in enumerate(train_loader):
            count += 1
            sample_size = 0 if i < warmup_epoch else 1 + args.beam_size * (i - warmup_epoch) // (args.max_epoch - warmup_epoch)
            current_batch = Batch.from_example_list(cur_dataset, device, train=True, method='gtol', smoothing=args.smoothing)

            if count == args.grad_accumulate or j == last_iter:
                (loss, hyps), vr_loss, gp_loss = model(current_batch, sample_size=sample_size, gtol_size=args.gtol_size, n_best=args.n_best, order_method=args.order_method, cum_method=args.cumulate_method, method='gtol')
                epoch_loss += loss.item()
                epoch_vr_loss += vr_loss.item()
                epoch_gp_loss += gp_loss.item()
                loss = loss + vr_loss + gp_loss
                (world_size * loss).backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                count = 0
            else:
                with model.no_sync():
                    (loss, hyps), vr_loss, gp_loss = model(current_batch, sample_size=sample_size, gtol_size=args.gtol_size, n_best=args.n_best, order_method=args.order_method, cum_method=args.cumulate_method, method='gtol')
                    epoch_loss += loss.item()
                    epoch_vr_loss += vr_loss.item()
                    epoch_gp_loss += gp_loss.item()
                    loss = loss + vr_loss + gp_loss
                    (world_size * loss).backward()
            # prod_results = sql_trans.grammar.order_controller.order_statistics(hyps, prod_results)
            # all_results = sql_trans.grammar.order_controller.order_history(hyps, current_batch.ids, all_results, idx=j)

        torch.cuda.empty_cache()
        gc.collect()
        if is_master():
            logger.info('Training: \tEpoch: %d\tTime: %.4f\tTraining loss: %.4f/%.4f/%.4f' % (i, time.time() - start_time, epoch_loss, epoch_vr_loss, epoch_gp_loss))
        # all_prod2fields.append(prod_results)
        if i < args.eval_after_epoch: # avoid unnecessary evaluation
            continue

        if is_master(): # only evaluate on the master GPU
            start_time = time.time()
            dev_acc = decode('dev', os.path.join(exp_path, 'dev.iter' + str(i)), acc_type='sql', order_method='all')
            logger.info('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.4f' % (i, time.time() - start_time, dev_acc))
            if dev_acc > best_result['dev_acc']:
                best_result['dev_acc'], best_result['iter'] = dev_acc, i
                torch.save({
                    'epoch': i, 'model': model.module.state_dict(),
                    'optim': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, open(os.path.join(exp_path, 'model.bin'), 'wb'))
                logger.info('NEW BEST MODEL: \tEpoch: %d\tDev acc: %.4f' % (i, dev_acc))
    if is_master():
        logger.info('FINAL BEST RESULT: \tEpoch: %d\tDev acc: %.4f' % (best_result['iter'], best_result['dev_acc']))
    # pickle.dump(all_results, open(os.path.join(exp_path, 'history.bin.rank%s' % (rank)), 'wb'))
    # pickle.dump(all_prod2fields, open(os.path.join(exp_path, 'order.bin.rank%s' % (rank)), 'wb'))
    dist.destroy_process_group()
else:
    if is_master():
        logger.info('Start evaluating ......')
        start_time = time.time()
        ex_acc, value_fscore, schema_acc = decode('train', output_path=os.path.join(args.read_model_path, 'train.eval' + str(args.beam_size)), acc_type='eg-sql', value_fscore=True, schema_acc=True)
        logger.info("Evaluation costs %.2fs ; Train dataset execution execution-guided acc is %.4f ." % (time.time() - start_time, ex_acc))
        logger.info("Evaluation costs %.2fs ; Train dataset value recognition fscore/graph pruning acc is %.4f/%.4f ." % (time.time() - start_time, value_fscore, schema_acc))
        start_time = time.time()
        ex_acc, value_fscore, schema_acc = decode('dev', output_path=os.path.join(args.read_model_path, 'dev.eval' + str(args.beam_size)), acc_type='eg-sql', value_fscore=True, schema_acc=True)
        logger.info("Evaluation costs %.2fs ; Dev dataset execution execution-guided acc is %.4f ." % (time.time() - start_time, ex_acc))
        logger.info("Evaluation costs %.2fs ; Dev dataset value recognition fscore/graph pruning acc is %.4f/%.4f ." % (time.time() - start_time, value_fscore, schema_acc))
        # start_time = time.time()
        # ex_acc_beam = decode('dev', output_path=os.path.join(args.read_model_path, 'dev.eval.beam' + str(args.beam_size)), acc_type='beam')
        # logger.info("Evaluation costs %.2fs ; Dev dataset execution beam acc (provided any one in the beam is correct) is %.4f ." % (time.time() - start_time, ex_acc_beam))
