#!/bin/bash
encode_method=lgesql #irnet, rgatsql, lgesql
translator=$1 #'mbart50_m2m', 'mbart50_m2en', 'm2m_100_418m', 'm2m_100_1.2b'

echo "Start to preprocess the original train dataset ..."
python3 -u preprocess/process_input.py --dataset 'cspider' --raw_table --data_split train --encode_method $encode_method --skip_large --translator $translator #--verbose > train.log
#python3 -u preprocess/process_input.py --dataset 'cspider' --data_split train --encode_method $encode_method --skip_large --translator $translator #--verbose > train.log
python3 -u preprocess/process_output.py --dataset 'cspider' --data_split train --encode_method $encode_method --translator $translator #--verbose >> train.log

echo "Start to preprocess the original dev dataset ..."
python3 -u preprocess/process_input.py --dataset 'cspider' --data_split dev --encode_method $encode_method --translator $translator #--verbose > dev.log
python3 -u preprocess/process_output.py --dataset 'cspider' --data_split dev --encode_method $encode_method --translator $translator #--verbose >> dev.log

echo "Start to preprocess the original test dataset ..."
python3 -u preprocess/process_input.py --dataset 'cspider' --data_split 'test' --encode_method $encode_method --translator $translator #--verbose > dev.log
