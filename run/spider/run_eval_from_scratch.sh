read_model_path=saved_models/spider-electra-large-79.0/
output_path=saved_models/spider-electra-large-79.0/dev_pred.sql
ts_order=controller

#read_model_path=saved_models/spider-glove-69.63
#output_path=saved_models/spider-glove-69.63/dev_pred.sql
#ts_order=enum

dataset_path=data/spider/dev.original.json
table_path=data/spider/tables.original.json
db_dir=data/spider/database-testsuite
batch_size=50
beam_size=5
ts_order=controller
device=0

python3 -u scripts/spider/eval_from_scratch.py --db_dir $db_dir --table_path $table_path --dataset_path $dataset_path --read_model_path $read_model_path --output_path $output_path --batch_size $batch_size --beam_size $beam_size --ts_order $ts_order --device $device
python3 eval/spider/evaluation.py --gold 'data/spider/dev_gold.sql' --pred $output_path --db $db_dir --table table_path --etype 'all' > $read_model_path/dev_eval.log
