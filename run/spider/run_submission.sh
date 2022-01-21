saved_model=exp/task_gtol_mixed__model_rgatsql__dataset_spider/view_global__plm_electra-large-discriminator__gs_4__nb_1__om_random__cm_sum__sf__bs_20__lr_0.0001_ld_0.8__l2_0.1__me_200__mn_5.0__bm_5__seed_1024__os_1024
output_path=exp/task_gtol_mixed__model_rgatsql__dataset_spider/view_global__plm_electra-large-discriminator__gs_4__nb_1__om_random__cm_sum__sf__bs_20__lr_0.0001_ld_0.8__l2_0.1__me_200__mn_5.0__bm_5__seed_1024__os_1024/predicted_sql.txt
batch_size=50
beam_size=5
order_method=all

python3 -u scripts/spider/eval_from_scratch.py --db_dir 'data/spider/database' --table_path 'data/spider/tables.json' --dataset_path 'data/spider/test.json' --saved_model $saved_model --output_path $output_path --batch_size $batch_size --beam_size $beam_size --order_method $order_method
python3 eval/spider/evaluation.py --gold 'data/spider/test_gold.sql' --pred $output_path --db 'data/spider/database' --table 'data/spider/tables.json' --etype 'all' > $saved_model/evaluation_all.log
