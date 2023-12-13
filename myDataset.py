import json

import jieba
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset


token = BertTokenizer.from_pretrained('pretrain/chinese_wwm_ext_pytorch')

class MyDataset(Dataset):
    def __init__(self, path):
        self.data = self.__load_data__(path)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __load_data__(self, path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = json.load(f)
        for line in lines:
            label, topic, text = int(line['label']), line['topic'], line['text']
            data.append({'label': label, 'topic': topic, 'text': text})
        return data

    def process(self):
        pass

class MyCollate():
    def __init__(self,config):
        self.max_length = config.max_length
    def collate_fn(self,data):
        max_length = self.max_length
        labels = [i['label'] for i in data]
        topic = [i['topic'] for i in data]
        text = [i['text'] for i in data]
        text_word = [' '.join(jieba.lcut(i)) for i in text]

        text_data = token.batch_encode_plus(batch_text_or_text_pairs=text,
                                            truncation=True,
                                            padding='max_length',
                                            max_length=max_length,
                                            return_tensors='pt',
                                            return_length=True)
        text_input_ids = text_data['input_ids']
        text_attention_mask = text_data['attention_mask']
        text_token_type_ids = text_data['token_type_ids']

        topic_data = token.batch_encode_plus(batch_text_or_text_pairs=topic,
                                             truncation=True,
                                             padding='max_length',
                                             max_length=max_length,
                                             return_tensors='pt',
                                             return_length=True)
        topic_input_ids = topic_data['input_ids']
        topic_attention_mask = topic_data['attention_mask']
        topic_token_type_ids = topic_data['token_type_ids']
        labels = torch.LongTensor(labels)

        word_data = token.batch_encode_plus(batch_text_or_text_pairs=text_word,
                                            truncation=True,
                                            padding='max_length',
                                            max_length=max_length,
                                            return_tensors='pt',
                                            return_length=True)
        word_input_ids = word_data['input_ids']
        word_attention_mask = word_data['attention_mask']
        word_token_type_ids = word_data['token_type_ids']

        return (labels, [text_input_ids, text_attention_mask, text_token_type_ids],
                [topic_token_type_ids, topic_attention_mask, topic_input_ids],
                [word_token_type_ids, word_attention_mask, word_input_ids]
                )


