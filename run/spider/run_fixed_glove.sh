task=all_best_order
seed=$1
order_seed=1024
device=0
testing='' #'--testing'
read_model_path=''
read_order_path='--read_order_path exp/task_random__model_rgatsql__dataset_spider/view_global__emb_300__gnn_256_x_8__share__head_8__share__gs_5__nb_5__om_controller__cm_sum__bs_20__lr_0.0005__l2_0.0001__ls_0.15__wp_0.1__sd_linear__me_100__mn_5.0__bm_5__seed_111__os_1024/'

model=rgatsql
local_and_nonlocal=global # mmc, local, global
embed_size=300
schema_aggregation=head+tail
gnn_hidden_size=256
gnn_num_layers=8
relation_share_layers='--relation_share_layers' #'--relation_share_layers'
relation_share_heads='--relation_share_heads' #'--relation_share_heads'
score_function=affine
value_aggregation=attentive-pooling
num_heads=8
dropout=0.2
attn_drop=0.0
drop_connect=0.2

lstm=onlstm
chunk_size=8
att_vec_size=512
lstm_hidden_size=512
lstm_num_layers=1
action_embed_size=128
field_embed_size=64
type_embed_size=64
context_feeding=''
struct_feeding='--struct_feeding'
no_share_clsfy=''

batch_size=20
grad_accumulate=1
lr=5e-4
l2=1e-4
smoothing=0.15
warmup_ratio=0.1
lr_schedule=linear
eval_after_epoch=60
max_epoch=100
max_norm=5
beam_size=5

python -u scripts/spider/train_and_eval_fixed.py --task $task --dataset 'spider' --db_dir 'data/spider/database' --seed $seed --order_seed $order_seed --device $device $testing $read_model_path $read_order_path \
    --model $model --local_and_nonlocal $local_and_nonlocal --gnn_hidden_size $gnn_hidden_size --dropout $dropout --attn_drop $attn_drop --att_vec_size $att_vec_size \
    --value_aggregation $value_aggregation --score_function $score_function $relation_share_layers $relation_share_heads \
    --schema_aggregation $schema_aggregation --embed_size $embed_size --gnn_num_layers $gnn_num_layers --num_heads $num_heads \
    --lstm $lstm --chunk_size $chunk_size --drop_connect $drop_connect --lstm_hidden_size $lstm_hidden_size --lstm_num_layers $lstm_num_layers \
    --struct_feeding $struct_feeding --no_share_clsfy $no_share_clsfy \
    --action_embed_size $action_embed_size --field_embed_size $field_embed_size --type_embed_size $type_embed_size $context_feeding \
    --batch_size $batch_size --grad_accumulate $grad_accumulate --lr $lr --l2 $l2 --warmup_ratio $warmup_ratio --lr_schedule $lr_schedule --eval_after_epoch $eval_after_epoch \
    --smoothing $smoothing --max_epoch $max_epoch --max_norm $max_norm --beam_size $beam_size
