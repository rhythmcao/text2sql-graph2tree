#!/bin/bash
task=evaluation
read_model_path=exp/task_gtol_mixed_nosample__model_rgatsql__dataset_dusql/view_global__plm_chinese-macbert-base__gs_4__nb_1__om_random__cm_sum__sf__bs_20__lr_0.0002_ld_0.8__l2_0.1__me_160__mn_5.0__bm_5__seed_${1}__os_1024

batch_size=50
beam_size=5
device=0

python -u scripts/dusql/extract_order.py --task $task --dataset 'dusql' --db_dir 'data/dusql/db_content.json' --read_model_path $read_model_path \
    --batch_size $batch_size --beam_size $beam_size --device $device
