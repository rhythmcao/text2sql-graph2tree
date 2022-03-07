#coding=utf8
import os, json, pickle, argparse, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import DATASETS

def get_input_processor(**kargs):
    dataset = kargs.pop('dataset', 'spider')
    if dataset == 'spider':
        from preprocess.spider.input_utils import InputProcessor
    elif dataset == 'dusql':
        from preprocess.dusql.input_utils import InputProcessor
    elif dataset == 'cspider':
        from preprocess.cspider.input_utils import InputProcessor
    elif dataset == 'cspider_raw':
        from preprocess.cspider_raw.input_utils import InputProcessor
    elif dataset == 'wikisql':
        pass
    elif dataset == 'nl2sql':
        pass
    else:
        raise ValueError('Not recognized dataset name %s' % (dataset))
    encode_method = kargs.pop('encode_method', 'lgesql')
    db_dir = kargs.pop('db_dir', DATASETS[dataset]['database'])
    db_content = kargs.pop('db_content', DATASETS[dataset]['db_content'])
    bridge = kargs.pop('bridge', DATASETS[dataset]['bridge'])
    input_processor = InputProcessor(encode_method=encode_method, db_dir=db_dir, db_content=db_content, bridge=bridge)
    return input_processor

def process_tables(processor, tables_list, output_path=None, verbose=False):
    tables = {}
    for each in tables_list:
        print('*************** Processing database %s **************' % (each['db_id']))
        tables[each['db_id']] = processor.preprocess_database(each, verbose=verbose)
    print('In total, process %d databases .' % (len(tables)))
    if output_path is not None:
        pickle.dump(tables, open(output_path, 'wb'))
    return tables

def process_dataset_input(processor, dataset, tables, output_path=None, skip_large=False, verbose=False):
    processed_dataset = []
    for idx, entry in enumerate(dataset):
        if skip_large and len(tables[entry['db_id']]['column_names']) > 100: continue
        if (idx + 1) % 500 == 0:
            print('*************** Processing inputs of the %d-th sample **************' % (idx + 1))
        # preprocess raw question tokens, schema linking and graph construction
        entry = processor.pipeline(entry, tables[entry['db_id']], verbose=verbose)
        processed_dataset.append(entry)
    print('Table: partial match %d ; exact match %d' % (processor.table_pmatch, processor.table_ematch))
    print('Column: partial match %d ; exact match %d ; value match %d' % (processor.column_pmatch, processor.column_ematch, processor.column_vmatch))
    print('In total, process %d samples , skip %d extremely large databases.' % (len(processed_dataset), len(dataset) - len(processed_dataset)))
    if output_path is not None:
        # serialize preprocessed dataset
        pickle.dump(processed_dataset, open(output_path, 'wb'))
    return processed_dataset

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset', type=str, required=True, choices=['spider', 'dusql', 'wikisql', 'nl2sql', 'cspider', 'cspider_raw'])
    arg_parser.add_argument('--data_split', type=str, required=True, choices=['train', 'dev', 'test'], help='dataset path')
    arg_parser.add_argument('--raw_table', action='store_true', help='use raw tables, need preprocessing')
    arg_parser.add_argument('--encode_method', choices=['irnet', 'rgatsql', 'lgesql'], default='lgesql', help='gnn model name')
    arg_parser.add_argument('--skip_large', action='store_true', help='whether skip large databases')
    arg_parser.add_argument('--verbose', action='store_true', help='whether print processing information')
    args = arg_parser.parse_args()

    processor = get_input_processor(**vars(args))
    data_dir = DATASETS[args.dataset]['data']
    table_path = os.path.join(data_dir, 'tables.bin')
    # loading database and dataset
    if args.raw_table:
        # need to preprocess database items
        start_time = time.time()
        print('Firstly, preprocess the original databases ...')
        raw_table_path = os.path.join(data_dir, 'tables.json')
        tables_list = json.load(open(raw_table_path, 'r'))
        tables = process_tables(processor, tables_list, table_path, args.verbose)
        print('Databases preprocessing costs %.4fs .' % (time.time() - start_time))
    else: tables = pickle.load(open(table_path, 'rb'))

    start_time = time.time()
    dataset_path = os.path.join(data_dir, args.data_split + '.json')
    dataset = json.load(open(dataset_path, 'r'))
    output_path = os.path.join(data_dir, '.'.join([args.data_split, args.encode_method, 'bin']))
    dataset = process_dataset_input(processor, dataset, tables, output_path, args.skip_large, verbose=args.verbose)
    if args.dataset == 'cspider_raw':
        processor.translator.save_translation_memory()
    print('Dataset preprocessing costs %.4fs .' % (time.time() - start_time))
