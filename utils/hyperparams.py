#coding=utf8
import os

EXP_PATH = 'exp'

def hyperparam_path_from_pretrained(args, create=True):
    if args.read_model_path and args.testing:
        return args.read_model_path
    assert args.read_pretrained_model_path, 'read_pretrained_model_path must be set'
    task = 'task_%s__dataset_%s' % (args.task, args.dataset)
    exp_path = '' if not args.supervised else 'sup__'
    exp_path += 'rm_%s_rs_%s_lm_%s' % (args.rl_method, args.rl_size, args.lam)
    exp_path += '__bs_%s' % (args.batch_size)
    exp_path += '__lr_%s_ld_%s' % (args.lr, args.layerwise_decay)
    exp_path += '__l2_%s' % (args.l2)
    exp_path += '__ls_%s' % (args.smoothing)
    exp_path += '__wp_%s' % (args.warmup_ratio)
    exp_path += '__sd_%s' % (args.lr_schedule)
    exp_path += '__me_%s' % (args.max_epoch)
    exp_path += '__mn_%s' % (args.max_norm)
    exp_path += '__bm_%s' % (args.beam_size)
    exp_path += '__seed_%s' % (args.seed)
    exp_path = os.path.join(EXP_PATH, task, exp_path)
    if create and not os.path.exists(exp_path):
        os.makedirs(exp_path)
    return exp_path


def hyperparam_path(args, create=True):
    if args.read_model_path and args.testing:
        return args.read_model_path
    exp_path = hyperparam_path_text2sql(args)
    if create and not os.path.exists(exp_path):
        os.makedirs(exp_path)
    return exp_path

def hyperparam_path_text2sql(args):
    task = 'task_%s__model_%s__dataset_%s' % (args.task, args.model, args.dataset)
    # encoder params
    exp_path = 'view_%s__' % (args.local_and_nonlocal)
    exp_path += 'emb_%s' % (args.embed_size) if args.plm is None else 'plm_%s' % (args.plm)
    # exp_path += '__gnn_%s_x_%s' % (args.gnn_hidden_size, args.gnn_num_layers)
    # exp_path += '__share' if args.relation_share_layers else ''
    # exp_path += '__head_%s' % (args.num_heads)
    # exp_path += '__share' if args.relation_share_heads else ''
    exp_path += '__gs_%s__nb_%s__om_%s__cm_%s' % (args.gtol_size, args.n_best, args.order_method, args.cumulate_method)
    exp_path += '__sf' if args.struct_feeding else ''
    exp_path += '__noshare' if args.no_share_clsfy else ''
    # exp_path += '__dp_%s' % (args.dropout)
    # exp_path += '__dpa_%s' % (args.attn_drop)
    # exp_path += '__dpc_%s' % (args.drop_connect)
    # decoder params
    # exp_path += '__cell_%s_%s_x_%s' % (args.lstm, args.lstm_hidden_size, args.lstm_num_layers)
    # exp_path += '_chunk_%s' % (args.chunk_size) if args.lstm == 'onlstm' else ''
    # exp_path += '__attvec_%s' % (args.att_vec_size)
    # exp_path += '_no' if args.no_context_feeding else ''
    # exp_path += '__ae_%s' % (args.action_embed_size)
    # exp_path += '__fe_%s' % (args.field_embed_size)
    # exp_path += '__te_%s' % (args.type_embed_size)
    # training params
    exp_path += '__bs_%s' % (args.batch_size)
    exp_path += '__lr_%s' % (args.lr) if args.plm is None else '__lr_%s_ld_%s' % (args.lr, args.layerwise_decay)
    exp_path += '__l2_%s' % (args.l2)
    # exp_path += '__ls_%s' % (args.smoothing)
    # exp_path += '__wp_%s' % (args.warmup_ratio)
    # exp_path += '__sd_%s' % (args.lr_schedule)
    exp_path += '__me_%s' % (args.max_epoch)
    exp_path += '__mn_%s' % (args.max_norm)
    exp_path += '__bm_%s' % (args.beam_size)
    exp_path += '__seed_%s' % (args.seed)
    exp_path += '__os_%s' % (args.order_seed)
    exp_path = os.path.join(EXP_PATH, task, exp_path)
    return exp_path
