task=evaluation
dataset=nl2sql
read_model_path=exp/task_gtl__encoder_lgesql__dataset_nl2sql/gs_4__nb_1__ts_random__uts_enum__plm_chinese-macbert-large__view_msde__sf__bs_20__lr_0.0001_ld_0.8__l2_0.1__me_100__mn_5.0__bm_5__seed_999
test_batch_size=50
beam_size=5
device=0

python -u scripts/$dataset/train_and_eval.py --task $task --dataset $dataset --device $device --testing --read_model_path $read_model_path --test_batch_size $test_batch_size --beam_size $beam_size
