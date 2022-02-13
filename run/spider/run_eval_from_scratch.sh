read_model_path=
output_path=
batch_size=50
beam_size=5
ts_order=controller
device=0

python3 -u scripts/spider/eval_from_scratch.py --db_dir 'data/spider/database-testsuite' --table_path 'data/spider/tables.json' --dataset_path 'data/spider/test.json' --read_model_path $read_model_path --output_path $output_path --batch_size $batch_size --beam_size $beam_size --ts_order $ts_order --deviceId $device
python3 eval/spider/evaluation.py --gold 'data/spider/test_gold.sql' --pred $output_path --db 'data/spider/database-testsuite' --table 'data/spider/tables.json' --etype 'all' > $read_model_path/evaluation.log
