task=evaluation
dataset=dusql
read_model_path=exp/task_gtl__encoder_lgesql__dataset_dusql/gs_4__nb_1__ts_random__uts_enum__plm_chinese-macbert-large__view_mmc__sf__bs_24__lr_0.0001_ld_0.8__l2_0.1__me_200__mn_5.0__bm_5__seed_1024
test_batch_size=50
beam_size=5
device=0

python -u scripts/$dataset/train_and_eval.py --task $task --dataset $dataset --device $device --testing --read_model_path $read_model_path --test_batch_size $test_batch_size --beam_size $beam_size
