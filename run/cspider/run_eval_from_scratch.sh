read_model_path=exp/task_gtl__encoder_lgesql__dataset_cspider/gs_4__nb_1__ts_random__uts_enum__plm_infoxlm-large__view_mmc__sf__bs_20__lr_0.0001_ld_0.8__l2_0.1__me_200__mn_5.0__bm_5__seed_999
batch_size=50
beam_size=5
ts_order=controller
deviceId=0

python3 -u scripts/cspider/eval_from_scratch.py --read_model_path $read_model_path --batch_size $batch_size --beam_size $beam_size --ts_order $ts_order --deviceId $deviceId
