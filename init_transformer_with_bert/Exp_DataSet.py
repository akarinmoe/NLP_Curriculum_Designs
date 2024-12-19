# Exp_DataSet.py

import os
import json
import torch
from torch.utils.data import Dataset, TensorDataset
from transformers import BertTokenizer

class Dictionary(object):
    def __init__(self, path):
        self.label2idx = {}
        self.idx2label = []

        with open(os.path.join(path, 'labels.json'), 'r', encoding='utf-8') as f:
            for line in f:
                one_data = json.loads(line)
                label, label_desc = one_data['label'], one_data['label_desc']
                self.idx2label.append([label, label_desc])
                self.label2idx[label] = len(self.idx2label) -1

class Corpus(object):
    def __init__(self, path, max_token_per_sent=50):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.max_token_per_sent = max_token_per_sent
        self.dictionary = Dictionary(path)

        self.train = self.tokenize(os.path.join(path, 'train.json'), is_test=False)
        self.valid = self.tokenize(os.path.join(path, 'dev.json'), is_test=False)
        self.test = self.tokenize(os.path.join(path, 'test.json'), is_test=True)

        # 打印词表大小，确保与 BERT 的词汇表一致
        print(f"Loaded Corpus with vocab_size={self.tokenizer.vocab_size}")
    
    def tokenize(self, path, is_test=False):
        input_ids = []
        labels = []
        ids = []
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                one_data = json.loads(line)
                text = one_data['sentence']

                # 使用 BERT 的分词器进行编码
                encoding = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_token_per_sent,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                input_id = encoding['input_ids'].squeeze(0)  # [max_token_per_sent]
                input_ids.append(input_id)

                if is_test:
                    id_ = one_data['id']
                    ids.append(id_)
                else:
                    label = one_data['label']
                    if label not in self.dictionary.label2idx:
                        raise ValueError(f"Label '{label}' not found in labels.json")
                    label_idx = self.dictionary.label2idx[label]
                    labels.append(label_idx)

        input_ids = torch.stack(input_ids)  # [num_samples, max_token_per_sent]
        if is_test:
            # 返回 (input_ids, ids)
            ids_tensor = torch.tensor(ids)  # [num_samples]
            return TensorDataset(input_ids, ids_tensor)
        else:
            # 返回 (input_ids, labels)
            labels_tensor = torch.tensor(labels).long()  # [num_samples]
            return TensorDataset(input_ids, labels_tensor)
