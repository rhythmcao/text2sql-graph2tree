#coding=utf8
""" Utility functions include:
    1. set output logging path
    2. set random seed for all libs
    3. select torch.device
"""

import re, sys, os, logging
import random, torch, dgl
import numpy as np
import torch.distributed as dist

def set_logger(exp_path, testing=False, rank=-1):
    logFormatter = logging.Formatter('%(asctime)s - %(message)s') #('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('mylogger')
    level = logging.DEBUG if rank in [-1, 0] else logging.WARN
    logger.setLevel(level)
    if testing:
        fileHandler = logging.FileHandler('%s/log_test.txt' % (exp_path), mode='w')
    else:
        fileHandler = logging.FileHandler('%s/log_train.txt' % (exp_path), mode='w')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    return logger

def set_random_seed(random_seed=999):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    dgl.random.seed(random_seed)

def set_torch_device(deviceId):
    if deviceId < 0:
        device = torch.device("cpu")
    else:
        assert torch.cuda.device_count() >= deviceId + 1
        device = torch.device("cuda:%d" % (deviceId))
        # os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # used when debug
        ## These two sentences are used to ensure reproducibility with cudnnbacken
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
    return device

def set_torch_device_ddp(local_rank):
    # print(os.environ['CUDA_VISIBLE_DEVICES'])
    # print(torch.cuda.device_count())
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda:%s" % (local_rank))
    torch.backends.cudnn.enabled = False
    return device

def get_master_node_addr():
    try:
        nodelist = os.environ['SLURM_STEP_NODELIST']
    except:
        nodelist = os.environ['SLURM_JOB_NODELIST']
    nodelist = nodelist.strip().split(',')[0]
    text = re.split('[-\[\]]',nodelist)
    if ('' in text):
        text.remove('')
    return text[0] + '-' + text[1] + '-' + text[2]

def distributed_init(host_addr, rank, local_rank, world_size, port=23456):
    host_addr_full = 'tcp://' + host_addr + ':' + str(port)
    try:
        torch.distributed.init_process_group("nccl", init_method=host_addr_full,
                                              rank=rank, world_size=world_size)
    except:
        print(f"host addr {host_addr_full}")
        print(f"process id {int(os.environ['SLURM_PROCID'])}")
        exit("Distributed training initialization failed")
    assert torch.distributed.is_initialized()
    return set_torch_device_ddp(local_rank)
