db_dir=data/cspider_raw/database
table_path=data/cspider_raw/tables.original.json
dataset_path=data/cspider_raw/dev.original.json
gold=data/cspider_raw/dev_gold.sql

read_model_path=
output_path=
batch_size=50
beam_size=5
ts_order=controller
device=0

python3 -u scripts/cspider_raw/eval_from_scratch.py --db_dir $db_dir --table_path $table_path --dataset_path --read_model_path $read_model_path --output_path $output_path --batch_size $batch_size --beam_size $beam_size --ts_order $ts_order --device $device
python3 eval/cspider_raw/evaluation.py --gold $gold --pred $output_path --db $db_dir --table $table_path --etype 'match' > $read_model_path/evaluation.log