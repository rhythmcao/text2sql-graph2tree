#coding=utf8
import os, json, sys
from easynmt import EasyNMT
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.constants import DATASETS

class SentenceTranslator():

    def __init__(self, model_name='mbart50_m2en', cache_folder=None) -> None:
        super(SentenceTranslator, self).__init__()
        self.model_name = model_name
        self.cache_folder = cache_folder if cache_folder is not None else DATASETS['cspider_raw']['cache_folder']
        self.translator = EasyNMT(self.model_name, cache_folder=self.cache_folder, load_translator=True)

    def translate(self, dataset: list, target_lang: str = 'en', batch_size: int = 16):
        questions = [ex['question'] for ex in dataset]
        translated_questions = []
        for idx in range(0, len(questions), batch_size):
            translated_questions.extend(self.translator.translate(questions[idx: idx + batch_size], target_lang=target_lang, batch_size=batch_size))
        for idx, ex in enumerate(dataset):
            ex['question'] = translated_questions[idx]
        return dataset
    
    def change_translator(self, model_name):
        if self.model_name != model_name:
            self.model_name = model_name
            self.translator = EasyNMT(self.model_name, cache_folder=self.cache_folder, load_translator=True)
        return

if __name__ == '__main__':

    models = ['mbart50_m2m', 'mbart50_m2en', 'm2m_100_418m', 'm2m_100_1.2b']
    data_dir = DATASETS['cspider_raw']['data']
    translator = SentenceTranslator(models[0])

    # translate the raw chinese sentences into english
    for model in models:
        translator.change_translator(model)

        train = json.load(open(os.path.join(data_dir, 'train.json'), 'r'))
        train = translator.translate(train)
        json.dump(train, open(os.path.join(data_dir, 'train_' + model + '.json'), 'w'))

        dev = json.load(open(os.path.join(data_dir, 'dev.json'), 'r'))
        dev = translator.translate(dev)
        json.dump(dev, open(os.path.join(data_dir, 'dev_' + model + '.json'), 'w'))