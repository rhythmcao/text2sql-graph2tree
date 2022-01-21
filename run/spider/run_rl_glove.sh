task=bandit_rl
seed=999
device=0
testing='' #'--testing'
read_model_path=''

read_pretrained_model_path='exp/task_fixed__model_rgatsql__dataset_spider/view_global__emb_300__gnn_256_x_8__share__head_8__share__gs_5__nb_5__om_controller__cm_sum__bs_20__lr_0.0005__l2_0.0001__ls_0.15__wp_0.1__sd_linear__me_100__mn_5.0__bm_5__seed_999__os_999/'
rl_size=4
rl_method=bandit
supervised='--supervised'
lam=$1

batch_size=20
grad_accumulate=2
lr=1e-4
l2=1e-4
smoothing=0.15
warmup_ratio=0.1
lr_schedule=linear # constant
max_epoch=$2
max_norm=5
beam_size=5

python -u scripts/spider/train_and_eval_with_rl.py --task $task --dataset 'spider' --db_dir 'data/spider/database' --seed $seed --device $device $testing $read_model_path \
    --read_pretrained_model_path $read_pretrained_model_path $supervised --rl_method $rl_method --rl_size $rl_size --lam $lam \
    --batch_size $batch_size --grad_accumulate $grad_accumulate --lr $lr --l2 $l2 --warmup_ratio $warmup_ratio --lr_schedule $lr_schedule \
    --smoothing $smoothing --max_epoch $max_epoch --max_norm $max_norm --beam_size $beam_size
