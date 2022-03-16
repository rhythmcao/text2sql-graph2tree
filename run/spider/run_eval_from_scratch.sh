#read_model_path=saved_models/spider-electra-large-79.0/
#output_path=saved_models/spider-electra-large-79.0/dev_pred.sql

read_model_path=saved_models/spider-glove-69.63
output_path=saved_models/spider-glove-69.63/dev_pred.sql

batch_size=50
beam_size=5
ts_order=enum
deviceId=0

python3 -u scripts/spider/eval_from_scratch.py --db_dir 'data/spider/database-testsuite' --table_path 'data/spider/tables.original.json' --dataset_path 'data/spider/dev.original.json' --read_model_path $read_model_path --output_path $output_path --batch_size $batch_size --beam_size $beam_size --ts_order $ts_order --deviceId $deviceId
python3 eval/spider/evaluation.py --gold 'data/spider/dev_gold.sql' --pred $output_path --db 'data/spider/database-testsuite' --table 'data/spider/tables.original.json' --etype 'all' > $read_model_path/dev_eval.log
