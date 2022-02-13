#coding=utf8
import sys, os, time, json, gc
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from argparse import Namespace
from utils.args import init_args
from utils.initialization import initialization_wrapper
from utils.example import Example
from utils.batch import Batch
from utils.optimization import set_optimizer
from model.model_utils import Registrable
from model.model_constructor import *
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from scripts.eval_model import decode, gather_ts_order_from_dataset

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
exp_path, logger, device, local_rank, rank, world_size = initialization_wrapper(args)
is_master = (rank == 0)

# configure class Example and load dataset
start_time = time.time()
if args.read_model_path:
    params = json.load(open(os.path.join(args.read_model_path, 'params.json')), object_hook=lambda d: Namespace(**d))
    params.lazy_load = True
else: params = args
# set up the grammar, order controller, transition system, evaluator, vocabulary, etc.
Example.configuration(params.dataset, plm=params.plm, encode_method=params.encode_method, ts_order_path=params.read_ts_order_path)
train_dataset, dev_dataset = Example.load_dataset('train'), Example.load_dataset('dev')
logger.info(f"Load dataset and database finished, cost {time.time() - start_time:.4f}s ...")
logger.info(f"Dataset size: train -> {len(train_dataset):d} ; dev -> {len(dev_dataset):d}")
sql_trans, controller, evaluator = Example.trans, Example.order_controller, Example.evaluator
args.word_vocab, args.relation_num = len(Example.word_vocab), len(Example.relation_vocab)

# model initialization
model = Registrable.by_name('text2sql')(params, sql_trans).to(device)
if args.read_model_path:
    check_point = torch.load(open(os.path.join(args.read_model_path, 'model.bin'), 'rb'), map_location=device)
    model.load_state_dict(check_point['model'])
    logger.info(f"Load saved model from path: {args.read_model_path:s}")
else:
    json.dump(vars(params), open(os.path.join(exp_path, 'params.json'), 'w'), indent=4)
    if params.plm is None:
        ratio = Example.word2vec.load_embeddings(model.encoder.input_layer.word_embed, Example.word_vocab, device=device)
        logger.info(f"Init word embedding layer with a coverage {ratio:.2f}")
if args.ddp: # add DDP wrapper for model
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
base_model = model.module if args.ddp else model

if not args.testing:
    assert args.batch_size % (world_size * args.grad_accumulate) == 0
    batch_size = args.batch_size // (world_size * args.grad_accumulate)
    # set training dataloader
    if args.ddp:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, collate_fn=Example.collate_fn)
        size_per_world = (len(train_dataset) + world_size - 1) // world_size
        last_iter, total_size = (size_per_world - 1) // batch_size, size_per_world * world_size
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=Example.collate_fn)
        last_iter, total_size = (len(train_dataset) - 1) // batch_size, len(train_dataset)
    # set optimizer and scheduler
    num_training_steps = ((total_size + args.batch_size - 1) // args.batch_size) * args.max_epoch
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    warmup_epoch = int(args.max_epoch * args.warmup_ratio)
    optimizer, scheduler = set_optimizer(base_model, args, num_warmup_steps, num_training_steps)

    start_epoch, best_result = 0, { 'dev_acc': 0. }
    if args.read_model_path and args.load_optimizer:
        optimizer.load_state_dict(check_point['optim'])
        scheduler.load_state_dict(check_point['scheduler'])
        start_epoch = check_point['epoch'] + 1
    logger.info(f'Total training steps: {num_training_steps:d};\t Warmup steps: {num_warmup_steps:d}')
    logger.info('Start training ......')

    for i in range(start_epoch, args.max_epoch):
        start_time = time.time()
        if args.ddp:
            train_loader.sampler.set_epoch(i)
        epoch_loss, count = {'ast_loss': 0., 'vr_loss': 0., 'gp_loss': 0.}, 0
        model.train()
        for j, cur_dataset in enumerate(train_loader):
            count += 1
            sample_size = 0 if i < warmup_epoch else 1 + args.beam_size * (i - warmup_epoch) // (args.max_epoch - warmup_epoch)
            current_batch = Batch.from_example_list(cur_dataset, device, train=True, ts_order=args.ts_order, uts_order=args.uts_order, smoothing=args.smoothing)

            if count == args.grad_accumulate or j == last_iter:
                outputs = model(current_batch, sample_size=sample_size, gtl_size=args.gtl_size, n_best=args.n_best, ts_order=args.ts_order, uts_order=args.uts_order)
                loss = outputs['ast_loss'] + outputs['vr_loss'] + outputs['gp_loss']
                (world_size * loss).backward() # reduction=sum
                base_model.pad_embedding_grad_zero()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                count = 0
            else:
                if args.ddp:
                    with model.no_sync(): # save communication time
                        outputs = model(current_batch, sample_size=sample_size, gtl_size=args.gtl_size, n_best=args.n_best, ts_order=args.ts_order, uts_order=args.uts_order)
                        loss = outputs['ast_loss'] + outputs['vr_loss'] + outputs['gp_loss']
                        (world_size * loss).backward() # reduction=sum
                else:
                    outputs = model(current_batch, sample_size=sample_size, gtl_size=args.gtl_size, n_best=args.n_best, ts_order=args.ts_order, uts_order=args.uts_order)
                    loss = outputs['ast_loss'] + outputs['vr_loss'] + outputs['gp_loss']
                    (world_size * loss).backward() # reduction=sum

            epoch_loss['ast_loss'] += outputs['ast_loss'].item()
            epoch_loss['vr_loss'] += outputs['vr_loss'].item()
            epoch_loss['gp_loss'] += outputs['gp_loss'].item()

        torch.cuda.empty_cache()
        gc.collect()
        logger.info('Training Epoch: %d\tTime: %.2fs\tTraining Loss: %.4f/%.4f/%.4f' %
            (i, time.time() - start_time, epoch_loss['ast_loss'], epoch_loss['vr_loss'], epoch_loss['gp_loss']))

        if i < args.eval_after_epoch: # avoid unnecessary evaluation
            continue

        if is_master: # only evaluate on the master GPU
            start_time = time.time()
            em, ex = decode(base_model, evaluator, dev_dataset, os.path.join(exp_path, 'dev.iter' + str(i)), batch_size=args.test_batch_size, beam_size=args.beam_size, ts_order=('controller' if args.ts_order == 'controller' else 'enum'), acc_type='eg-sql', etype='all', device=device)['sql']
            dev_acc = (em + ex) / 2.0 if Example.dataset == 'spider' else em if Example.dataset == 'cspider' else ex
            logger.info(f"Evaluation: \tEpoch: {i:d}\tTime: {time.time() - start_time:.2f}s\tDev sql exact set match/execution acc: {em:.4f}/{ex:.4f}")
            if dev_acc >= best_result['dev_acc']:
                best_result['dev_acc'], best_result['iter'] = dev_acc, i
                torch.save({
                    'epoch': i, 'model': base_model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, open(os.path.join(exp_path, 'model.bin'), 'wb'))
                logger.info(f"NEW BEST MODEL: \tEpoch: {i:d}\tDev sql exact set match/execution acc: {em:.4f}/{ex:.4f}")

    if is_master:
        check_point = torch.load(open(os.path.join(exp_path, 'model.bin'), 'rb'), map_location=device)
        base_model.load_state_dict(check_point['model'])
        logger.info(f"\nReload saved model in epoch {check_point['epoch']:d} from path: {exp_path:s}")

if is_master:
    Example.use_database_testsuite()
    logger.info("Start evaluating with enum method using testsuite database ......")
    start_time = time.time()
    results = decode(base_model, evaluator, dev_dataset, os.path.join(exp_path, 'dev.eval.enum'), batch_size=args.test_batch_size, beam_size=args.beam_size, ts_order='enum', acc_type='eg-sql', etype='all', value_metric='all', schema_metric='all', device=device)
    logger.info(f"EVALUATION costs {time.time() - start_time:.2f}s")
    logger.info(f"EVALUATION: \tDev sql exact set match/execution acc: {results['sql']}")
    logger.info(f"EVALUATION: \tDev value acc/acc-recall/fscore: {results['value']}")
    logger.info(f"EVALUATION: \tDev schema acc/acc-recall/fscore: {results['schema']}\n")

    if args.ts_order != 'controller': # accumulate and save ts canonical order from training dataset
        start_time = time.time()
        gather_ts_order_from_dataset(base_model, controller, train_dataset, os.path.join(exp_path, 'order.bin'), batch_size=args.test_batch_size, beam_size=args.beam_size, device=device)
        logger.info(f"Typed set order accumulation costs {time.time() - start_time:.2f}s")
    elif not os.path.exists(os.path.join(exp_path, 'order.bin')):
        controller.save_canonical_ts_order(save_path=os.path.join(exp_path, 'order.bin'))

    logger.info("Start evaluating with controller method using testsuite database ......")
    start_time = time.time()
    results = decode(base_model, evaluator, dev_dataset, os.path.join(exp_path, 'dev.eval.controller'), batch_size=args.test_batch_size, beam_size=args.beam_size, ts_order='controller', acc_type='eg-sql', etype='all', device=device)
    logger.info(f"EVALUATION costs {time.time() - start_time:.2f}s")
    logger.info(f"EVALUATION: \tDev sql exact set match/execution acc: {results['sql']}")

if args.ddp:
    dist.destroy_process_group()
