#coding=utf8
import os, pickle, argparse, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import DATASETS

def get_output_processor(dataset, table_path=None, db_dir=None):
    if dataset == 'spider':
        from preprocess.spider.output_utils import OutputProcessor
    elif dataset == 'dusql':
        from preprocess.dusql.output_utils import OutputProcessor
    elif dataset == 'cspider':
        from preprocess.cspider.output_utils import OutputProcessor
    elif dataset == 'cspider_raw':
        from preprocess.cspider_raw.output_utils import OutputProcessor
    elif dataset == 'wikisql':
        raise NotImplementedError
    elif dataset == 'nl2sql':
        raise NotImplementedError
    else:
        raise ValueError('Not recognized dataset name %s' % (dataset))
    table_path = os.path.join(DATASETS[dataset]['data'], 'tables.bin') if table_path is None else table_path
    db_dir = DATASETS[dataset]['database'] if db_dir is None else db_dir
    output_processor = OutputProcessor(table_path, db_dir)
    return output_processor

def process_dataset_output(processor, dataset, tables, output_path=None, skip_large=False, verbose=False):
    processed_dataset = []
    for idx, entry in enumerate(dataset):
        if skip_large and len(tables[entry['db_id']]['column_names']) > 100: continue
        if (idx + 1) % 500 == 0:
            print('*************** Processing outputs of the %d-th sample **************' % (idx + 1))
        # extract schema subgraph, graph pruning labels, values from (question, sql) pairs, and output action sequence
        entry = processor.pipeline(entry, tables[entry['db_id']], verbose=verbose)
        processed_dataset.append(entry)
    print('In total, process %d samples, skip %d samples .' % (len(processed_dataset), len(dataset) - len(processed_dataset)))
    if output_path is not None:
        # serialize preprocessed dataset
        pickle.dump(processed_dataset, open(output_path, 'wb'))
    return processed_dataset

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset', type=str, required=True, choices=['spider', 'dusql', 'wikisql', 'nl2sql', 'cspider', 'cspider_raw'])
    arg_parser.add_argument('--data_split', type=str, required=True, choices=['train', 'dev', 'test'], help='dataset path')
    arg_parser.add_argument('--encode_method', type=str, required=True, choices=['irnet', 'ratsql', 'lgesql'])
    arg_parser.add_argument('--verbose', action='store_true', help='whether print processing information')
    args = arg_parser.parse_args()

    # loading database and dataset
    start_time = time.time()
    data_dir, db_dir = DATASETS[args.dataset]['data'], DATASETS[args.dataset]['database']
    tables = pickle.load(open(os.path.join(data_dir, 'tables.bin'), 'rb'))
    processor = get_output_processor(args.dataset, table_path=tables, db_dir=db_dir)
    dataset_path = os.path.join(data_dir, '.'.join([args.data_split, args.encode_method, 'bin']))
    dataset = pickle.load(open(dataset_path, 'rb'))
    dataset = process_dataset_output(processor, dataset, tables, dataset_path, verbose=args.verbose)
    print('Dataset preprocessing costs %.4fs .' % (time.time() - start_time))