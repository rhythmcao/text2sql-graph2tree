#coding=utf8
import os

EXP_PATH = 'exp'


def hyperparam_path(args, create=True):
    if args.read_model_path and args.testing:
        return args.read_model_path
    exp_path = hyperparam_path_text2sql(args)
    if create and not os.path.exists(exp_path):
        os.makedirs(exp_path)
    return exp_path


def hyperparam_path_text2sql(args):
    task = 'task_%s__encoder_%s__dataset_%s' % (args.task, args.encode_method, args.dataset)

    # encoder params
    exp_path = '' if args.translator == 'none' else args.translator + '__'
    if args.ts_order == 'enum' or args.uts_order == 'enum':
        exp_path += 'gs_%s__nb_%s__ts_%s__uts_%s' % (args.gtl_size, args.n_best, args.ts_order, args.uts_order)
    else:
        exp_path += 'ts_%s__uts_%s' % (args.ts_order, args.uts_order)
    exp_path += '__emb_%s' % (args.embed_size) if args.plm is None else '__plm_%s' % (args.plm)
    exp_path += '__view_%s' % (args.local_and_nonlocal)
    # exp_path += '__gnn_%s_x_%s' % (args.gnn_hidden_size, args.gnn_num_layers)
    # exp_path += '__sl' if args.relation_share_layers else ''
    # exp_path += '__hd_%s' % (args.num_heads)
    # exp_path += '__sh' if args.relation_share_heads else ''
    # exp_path += '__dp_%s' % (args.dropout)
    # exp_path += '__dpa_%s' % (args.attn_drop)
    # exp_path += '__dpc_%s' % (args.drop_connect)

    # decoder params
    # exp_path += '__cell_%s_%s_x_%s' % (args.lstm, args.lstm_hidden_size, args.lstm_num_layers)
    # exp_path += '_ck_%s' % (args.chunk_size) if args.lstm == 'onlstm' else ''
    # exp_path += '__av_%s' % (args.att_vec_size)
    # exp_path += '__ae_%s' % (args.action_embed_size)
    # exp_path += '__fe_%s' % (args.field_embed_size)
    # exp_path += '__te_%s' % (args.type_embed_size)
    # exp_path += '__cf' if args.context_feeding else ''
    exp_path += '__sf' if args.struct_feeding else ''

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
    exp_path = os.path.join(EXP_PATH, task, exp_path)
    return exp_path
