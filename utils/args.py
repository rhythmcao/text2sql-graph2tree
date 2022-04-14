#coding=utf-8
import argparse
import sys

def init_args(params=sys.argv[1:]):
    arg_parser = argparse.ArgumentParser()
    arg_parser = add_argument_base(arg_parser)
    arg_parser = add_argument_encoder(arg_parser)
    arg_parser = add_argument_decoder(arg_parser)
    arg_parser = add_argument_gtl(arg_parser)
    opt = arg_parser.parse_args(params)
    if opt.encode_method in ['irnet', 'rgatsql'] and opt.local_and_nonlocal == 'msde':
        opt.local_and_nonlocal = 'global'
    if opt.encode_method == 'lgesql' and opt.local_and_nonlocal == 'global':
        opt.local_and_nonlocal = 'msde'
    return opt

def add_argument_base(arg_parser):
    #### General configuration ####
    arg_parser.add_argument('--task', default='text2sql', help='task name')
    arg_parser.add_argument('--dataset', type=str, default='spider', choices=['spider', 'dusql', 'wikisql', 'nl2sql', 'cspider', 'cspider_raw'])
    arg_parser.add_argument('--seed', default=999, type=int, help='Random seed')
    arg_parser.add_argument('--device', type=int, default=0, help='Use which device: -1 -> cpu ; the index of gpu o.w.')
    arg_parser.add_argument('--ddp', action='store_true', help='use distributed data parallel training')
    arg_parser.add_argument('--testing', action='store_true', help='training or evaluation mode')
    arg_parser.add_argument('--read_model_path', type=str, help='read pretrained model path')
    #### Training Hyperparams ####
    arg_parser.add_argument('--batch_size', default=20, type=int, help='Batch size')
    arg_parser.add_argument('--test_batch_size', default=64, type=int, help='Test batch size')
    arg_parser.add_argument('--grad_accumulate', default=1, type=int, help='accumulate grad and update once every x steps')
    arg_parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    arg_parser.add_argument('--layerwise_decay', type=float, default=1.0, help='layerwise decay rate for lr, used for PLM')
    arg_parser.add_argument('--l2', type=float, default=1e-4, help='weight decay coefficient')
    arg_parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup steps proportion')
    arg_parser.add_argument('--lr_schedule', default='linear', choices=['constant', 'linear', 'ratsql', 'cosine'], help='lr scheduler')
    arg_parser.add_argument('--eval_after_epoch', default=40, type=int, help='Start to evaluate after x epoch')
    arg_parser.add_argument('--load_optimizer', action='store_true', default=False, help='Whether to load optimizer state')
    arg_parser.add_argument('--max_epoch', type=int, default=100, help='terminate after maximum epochs')
    arg_parser.add_argument('--max_norm', default=5., type=float, help='clip gradients')
    arg_parser.add_argument('--translator', default='none', choices=['mbart50_m2m', 'mbart50_m2en', 'm2m_100_418m', 'm2m_100_1.2b', 'none'], help='translator for cspider series')
    return arg_parser

def add_argument_encoder(arg_parser):
    # Encoder Hyperparams
    arg_parser.add_argument('--encode_method', choices=['irnet', 'rgatsql', 'lgesql'], default='lgesql', help='which text2sql encoder to use')
    arg_parser.add_argument('--local_and_nonlocal', choices=['mmc', 'msde', 'local', 'global'], default='mmc', 
        help='how to integrate local and non-local relations: mmc -> multi-head multi-view concatenation ; msde -> mixed static and dynamic embeddings')
    arg_parser.add_argument('--plm', type=str, help='pretrained model name in Huggingface')
    arg_parser.add_argument('--subword_aggregation', choices=['mean-pooling', 'max-pooling', 'attentive-pooling'], default='attentive-pooling', help='aggregate subword feats from PLM')
    arg_parser.add_argument('--schema_aggregation', choices=['mean-pooling', 'max-pooling', 'attentive-pooling', 'head+tail'], default='head+tail', help='aggregate schema words feats')
    arg_parser.add_argument('--value_aggregation', choices=['mean-pooling', 'max-pooling', 'attentive-pooling'], default='attentive-pooling', help='aggregate value word feats')
    arg_parser.add_argument('--dropout', type=float, default=0.2, help='feature dropout rate')
    arg_parser.add_argument('--attn_drop', type=float, default=0., help='dropout rate of attention weights')
    arg_parser.add_argument('--embed_size', default=300, type=int, help='size of word embeddings, only used in glove.42B.300d')
    arg_parser.add_argument('--gnn_num_layers', default=8, type=int, help='num of GNN layers in encoder')
    arg_parser.add_argument('--gnn_hidden_size', default=256, type=int, help='size of GNN layers hidden states')
    arg_parser.add_argument('--num_heads', default=8, type=int, help='num of heads in multihead attn')
    arg_parser.add_argument('--relation_share_layers', action='store_true')
    arg_parser.add_argument('--relation_share_heads', action='store_true')
    arg_parser.add_argument('--score_function', choices=['affine', 'bilinear', 'biaffine'], default='affine', help='graph pruning score function')
    arg_parser.add_argument('--smoothing', type=float, default=0.15, help='label smoothing factor for graph pruning and value recognition')
    return arg_parser

def add_argument_decoder(arg_parser):
    # Decoder Hyperparams
    arg_parser.add_argument('--lstm', choices=['lstm', 'onlstm'], default='onlstm', help='Type of LSTM used, ONLSTM or traditional LSTM')
    arg_parser.add_argument('--chunk_size', default=8, type=int, help='parameter of ONLSTM')
    arg_parser.add_argument('--att_vec_size', default=512, type=int, help='size of attentional vector')
    arg_parser.add_argument('--drop_connect', type=float, default=0.2, help='recurrent connection dropout rate in decoder lstm')
    arg_parser.add_argument('--lstm_num_layers', type=int, default=1, help='num_layers of decoder')
    arg_parser.add_argument('--lstm_hidden_size', default=512, type=int, help='Size of LSTM hidden states')
    arg_parser.add_argument('--action_embed_size', default=128, type=int, help='Size of ApplyRule/GenToken action embeddings')
    arg_parser.add_argument('--field_embed_size', default=64, type=int, help='Embedding size of ASDL fields')
    arg_parser.add_argument('--type_embed_size', default=64, type=int, help='Embeddings ASDL types')
    arg_parser.add_argument('--context_feeding', action='store_true', default=False,
                            help='Whether use embedding of context vectors')
    arg_parser.add_argument('--struct_feeding', action='store_true', default=False,
                            help='whether add field embedding during attention calculation and classification')
    arg_parser.add_argument('--beam_size', default=5, type=int, help='Beam size for beam search')
    arg_parser.add_argument('--decode_max_step', default=120, type=int, help='Maximum number of time steps used in decoding')
    return arg_parser

def add_argument_gtl(arg_parser):
    arg_parser.add_argument('--gtl_size', default=4, type=int, help='number of reserved size during GTL')
    arg_parser.add_argument('--n_best', default=5, type=int, help='returned number of examples to calculate loss')
    arg_parser.add_argument('--ts_order', type=str, default='random', choices=['controller', 'random', 'enum'], help='order method for typed set in GTL training')
    arg_parser.add_argument('--uts_order', type=str, default='enum', choices=['controller', 'random', 'enum'], help='order method for untyped set in GTL training')
    arg_parser.add_argument('--read_ts_order_path', type=str, help='read saved best order for typed set, actually the order for each grammar rule')
    arg_parser.add_argument('--read_canonical_action_path', type=str, help='read saved canonical action sequence')
    return arg_parser
