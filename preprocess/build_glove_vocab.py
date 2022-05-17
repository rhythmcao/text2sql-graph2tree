#coding=utf8
import argparse, os, sys, pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter
from utils.constants import DATASETS

def construct_vocab_from_dataset(dataset, mwf=4, reference_file=None, output_path=None, sep='\t'):
    data_dir, bridge = DATASETS[dataset]['data'], DATASETS[dataset]['db_content'] and DATASETS[dataset]['bridge']
    dataset_path = os.path.join(data_dir, 'train.' + args.encode_method + '.bin')
    table_path = os.path.join(data_dir, 'tables.bin')
    tables = pickle.load(open(table_path, 'rb'))
    dataset = pickle.load(open(dataset_path, 'rb'))
    words = []
    for ex in dataset:
        words.extend(ex['processed_question_toks'])
        db = tables[ex['db_id']]
        words.extend(['table'] * len(db['table_names']))
        words.extend(db['column_types'])
        for c in db['processed_column_toks']:
            words.extend(c)
        for t in db['processed_table_toks']:
            words.extend(t)
        if bridge:
            for c in ex['processed_cells']:
                if c: words.extend(c)
    cnt = Counter(words)
    vocab = sorted(list(cnt.items()), key=lambda x: - x[1])
    glove_vocab = set()
    with open(reference_file, 'r', encoding='utf-8') as inf:
        for line in inf:
            line = line.strip()
            if line == '': continue
            glove_vocab.add(line)
    oov_words, oov_but_freq_words = set(), []
    for w, c in vocab:
        if w not in glove_vocab:
            oov_words.add(w)
            if c >= mwf:
                oov_but_freq_words.append((w, c))
    print('Out of glove vocabulary size: %d\nAmong them, %d words occur equal or more than %d times in training dataset.' % (len(oov_words), len(oov_but_freq_words), mwf))
    with open(output_path, 'w') as of:
        # first serialize oov but frequent words, allowing fine-tune them during training
        for w, c in oov_but_freq_words:
            of.write(w + sep + str(c) + '\n')
        # next serialize words in both train vocab and glove vocab according to decreasing frequency
        for w, c in vocab:
            if w not in oov_words:
                of.write(w + sep + str(c) + '\n')
    return len(vocab)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['spider', 'dusql', 'wikisql', 'nl2sql', 'cspider', 'cspider_raw'])
    parser.add_argument('--reference_file', type=str, default='pretrained_models/glove-42b-300d/vocab_glove.txt',
        help='eliminate word not in glove vocabulary, unless it occurs frequently >= mwf')
    parser.add_argument('--encode_method', choices=['irnet', 'rgatsql', 'lgesql'], default='lgesql')
    parser.add_argument('--output_path', type=str, required=True, help='output word vocabulary path')
    parser.add_argument('--mwf', default=4, type=int,
        help='minimum word frequency used to pick up frequent words in training dataset, but not in glove vocabulary')
    args = parser.parse_args()

    construct_vocab_from_dataset(args.dataset, mwf=args.mwf, reference_file=args.reference_file, output_path=args.output_path)