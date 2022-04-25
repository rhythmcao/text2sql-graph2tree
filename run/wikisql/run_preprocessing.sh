#!/bin/bash
encode_method=lgesql # irnet, rgatsql, lgesql
mwf=30
vocab_glove='pretrained_models/glove.42b.300d/vocab_glove.txt'
vocab='pretrained_models/glove.42b.300d/vocab_wikisql.txt'

echo "Transform the original WikiSQL data format into Spider data format..."
python3 preprocess/wikisql/transform_format.py

echo "Start to preprocess the original train dataset ..."
python3 -u preprocess/process_input.py --dataset 'wikisql' --raw_table --data_split train --encode_method $encode_method --skip_large #--verbose > train.log
# python3 -u preprocess/process_input.py --dataset 'wikisql' --data_split train --encode_method $encode_method --skip_large #--verbose > train.log
python3 -u preprocess/process_output.py --dataset 'wikisql' --data_split train --encode_method $encode_method #--verbose >> train.log

echo "Start to preprocess the original dev dataset ..."
python3 -u preprocess/process_input.py --dataset 'wikisql' --data_split dev --encode_method $encode_method #--verbose > dev.log
python3 -u preprocess/process_output.py --dataset 'wikisql' --data_split dev --encode_method $encode_method #--verbose >> dev.log

echo "Start to preprocess the original test dataset ..."
python3 -u preprocess/process_input.py --dataset 'wikisql' --data_split 'test' --encode_method $encode_method #--verbose > test.log

echo "Start to build word vocab for the dataset ..."
python3 -u preprocess/build_glove_vocab.py --dataset 'wikisql' --encode_method $encode_method --reference_file $vocab_glove --output_path $vocab --mwf $mwf