task=gtl
seed=999
device=0
ddp='--ddp' # --ddp
testing='' # --testing
read_model_path=''
read_ts_order_path=''
read_canonical_action_path=''

plm=$1 # xlm-roberta-large, xlm-roberta-large-finetuned-conll03-english
encode_method=lgesql # irnet, rgatsql, lgesql
local_and_nonlocal=msde # mmc, msde, local, global
gnn_hidden_size=512
gnn_num_layers=8
num_heads=8
subword_aggregation=attentive-pooling
schema_aggregation=head+tail
value_aggregation=attentive-pooling
score_function=affine
relation_share_layers='--relation_share_layers'
relation_share_heads='--relation_share_heads'
dropout=0.2
attn_drop=0.0
drop_connect=0.2

lstm=onlstm
chunk_size=8
lstm_hidden_size=512
lstm_num_layers=1
att_vec_size=512
context_feeding=''
action_embed_size=128
field_embed_size=64
type_embed_size=64
struct_feeding='--struct_feeding'

batch_size=20
test_batch_size=50
grad_accumulate=2
lr=1e-4
layerwise_decay=0.8
l2=0.1
warmup_ratio=0.1
lr_schedule=linear
smoothing=0.15
eval_after_epoch=120
max_epoch=200
max_norm=5
beam_size=5

gtl_size=4
n_best=1
ts_order=random
uts_order=enum

python -u scripts/cspider_raw/train_and_eval.py --task $task --dataset 'cspider_raw' --seed $seed --device $device $ddp $testing $read_model_path $read_ts_order_path $read_canonical_action_path \
    --plm $plm --encode_method $encode_method --local_and_nonlocal $local_and_nonlocal --gnn_hidden_size $gnn_hidden_size --gnn_num_layers $gnn_num_layers --num_heads $num_heads \
    --schema_aggregation $schema_aggregation --subword_aggregation $subword_aggregation --value_aggregation $value_aggregation --score_function $score_function \
    $relation_share_layers $relation_share_heads --dropout $dropout --attn_drop $attn_drop --drop_connect $drop_connect \
    --lstm $lstm --chunk_size $chunk_size --lstm_hidden_size $lstm_hidden_size --lstm_num_layers $lstm_num_layers --att_vec_size $att_vec_size $context_feeding \
    --action_embed_size $action_embed_size --field_embed_size $field_embed_size --type_embed_size $type_embed_size $struct_feeding \
    --batch_size $batch_size --test_batch_size $test_batch_size --grad_accumulate $grad_accumulate --lr $lr --layerwise_decay $layerwise_decay --l2 $l2 --warmup_ratio $warmup_ratio --lr_schedule $lr_schedule \
    --smoothing $smoothing --eval_after_epoch $eval_after_epoch --max_epoch $max_epoch --max_norm $max_norm --beam_size $beam_size \
    --gtl_size $gtl_size --n_best $n_best --ts_order $ts_order --uts_order $uts_order
