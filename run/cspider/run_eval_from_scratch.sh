read_model_path=
batch_size=50
beam_size=5
ts_order=controller
deviceId=0

python3 -u scripts/cspider/eval_from_scratch.py --read_model_path $read_model_path --batch_size $batch_size --beam_size $beam_size --ts_order $ts_order --deviceId $device