read_model_path=
output_path=
batch_size=50
beam_size=5
ts_order=controller
device=0

python3 -u scripts/cspider_raw/eval_from_scratch.py --db_dir 'data/cspider_raw/database' --table_path 'data/cspider_raw/tables.original.json' --dataset_path 'data/cspider_raw/dev.original.json' --read_model_path $read_model_path --output_path $output_path --batch_size $batch_size --beam_size $beam_size --ts_order $ts_order --deviceId $device
python3 eval/cspider_raw/evaluation.py --gold 'data/cspider_raw/dev_gold.sql' --pred $output_path --db 'data/cspider_raw/database' --table 'data/cspider_raw/tables.original.json' --etype 'match' > $read_model_path/evaluation.log
