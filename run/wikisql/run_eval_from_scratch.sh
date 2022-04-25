read_model_path=
output_path=$read_model_path/wikisql.sql
batch_size=50
beam_size=5
ts_order=enum
device=0

python3 -u scripts/wikisql/eval_from_scratch.py --read_model_path $read_model_path --output_path $output_path --batch_size $batch_size --beam_size $beam_size --ts_order $ts_order --device $device
python3 -u eval/wikisql/evaluate.py data/wikisql/test_gold.sql data/wikisql/test.db $output_path > $read_model_path/test.eval