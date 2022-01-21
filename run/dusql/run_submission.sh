saved_model=exp/task_gtol_large_mixed__model_rgatsql__dataset_dusql/view_global__plm_chinese-macbert-large__gs_4__nb_1__om_random__cm_sum__sf__bs_20__lr_0.0001_ld_0.8__l2_0.1__me_200__mn_5.0__bm_5__seed_999__os_1024/
output_path=exp/task_gtol_large_mixed__model_rgatsql__dataset_dusql/view_global__plm_chinese-macbert-large__gs_4__nb_1__om_random__cm_sum__sf__bs_20__lr_0.0001_ld_0.8__l2_0.1__me_200__mn_5.0__bm_5__seed_999__os_1024/dusql.sql
batch_size=50
beam_size=5

python -u scripts/dusql/eval_test.py --db_dir 'data/dusql/db_content.json' --table_path 'data/dusql/tables.bin' --raw_table_path 'data/dusql/tables.json' \
    --dataset_path 'data/dusql/test.bin' --saved_model $saved_model --output_path $output_path --batch_size $batch_size --beam_size $beam_size --use_gpu
