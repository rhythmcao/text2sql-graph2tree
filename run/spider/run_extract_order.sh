#!/bin/bash
task=evaluation
read_model_path=exp/task_gtol_mixed__model_rgatsql__dataset_spider/view_global__plm_electra-large-discriminator__gs_4__nb_1__om_random__cm_sum__sf__bs_20__lr_0.0001_ld_0.8__l2_0.1__me_200__mn_5.0__bm_5__seed_1024__os_1024
#exp/task_gtol_mixed__model_rgatsql__dataset_spider/view_global__plm_electra-large-discriminator__gs_4__nb_1__om_random__cm_sum__sf__bs_20__lr_0.0001_ld_0.8__l2_0.1__me_200__mn_5.0__bm_5__seed_999__os_1024
#exp/task_gtol_controller__model_rgatsql__dataset_spider/view_global__emb_300__gs_4__nb_1__om_controller__cm_sum__sf__bs_20__lr_0.0005__l2_0.0001__me_100__mn_5.0__bm_5__seed_555__os_1024/

batch_size=50
beam_size=5
device=0

python -u scripts/spider/extract_order.py --task $task --dataset 'spider' --db_dir 'data/spider/database' --read_model_path $read_model_path \
    --batch_size $batch_size --beam_size $beam_size --device $device
