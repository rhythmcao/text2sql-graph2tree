#!/bin/bash
encode_method=lgesql #irnet, rgatsql, lgesql

#echo "Fix some annotation errors in the train and dev dataset ..."
#python3 -u preprocess/dusql/fix_error.py

echo "Start to preprocess the original train dataset ..."
python3 -u preprocess/process_input.py --dataset 'dusql' --raw_table --data_split train --encode_method $encode_method --skip_large #--verbose > train.log
# python3 -u preprocess/process_input.py --dataset 'dusql' --data_split train --encode_method $encode_method --skip_large #--verbose > train.log
python3 -u preprocess/process_output.py --dataset 'dusql' --data_split train --encode_method $encode_method #--verbose >> train.log

echo "Start to preprocess the original dev dataset ..."
python3 -u preprocess/process_input.py --dataset 'dusql' --data_split dev --encode_method $encode_method #--verbose > dev.log
python3 -u preprocess/process_output.py --dataset 'dusql' --data_split dev --encode_method $encode_method #--verbose >> dev.log

echo "Start to preprocess the original test dataset ..."
python3 -u preprocess/process_input.py --dataset 'dusql' --data_split 'test' --encode_method $encode_method #--verbose > test.log
