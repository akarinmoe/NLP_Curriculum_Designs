import os
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset
import jieba

class Dictionary(object):
    def __init__(self, path):
        self.word2tkn = {"[PAD]": 0}
        self.tkn2word = ["[PAD]"]
        self.label2idx = {}
        self.idx2label = []

        with open(os.path.join(path, 'labels.json'), 'r', encoding='utf-8') as f:
            for line in f:
                one_data = json.loads(line)
                label, label_desc = one_data['label'], one_data['label_desc']
                self.idx2label.append([label, label_desc])
                self.label2idx[label] = len(self.idx2label) - 1

    def add_word(self, word):
        if word not in self.word2tkn:
            self.tkn2word.append(word)
            self.word2tkn[word] = len(self.tkn2word) - 1
        return self.word2tkn[word]


class Corpus(object):
    def __init__(self, path, max_token_per_sent):
        self.dictionary = Dictionary(path)
        self.max_token_per_sent = max_token_per_sent

        self.train = self.tokenize(os.path.join(path, 'train.json'))
        self.valid = self.tokenize(os.path.join(path, 'dev.json'))
        self.test = self.tokenize(os.path.join(path, 'test.json'), True)

        # 打印词表前50个词，帮助检查分词是否与预训练词典一致
        print("Sample of dictionary words:", self.dictionary.tkn2word[:50])

        self.embedding_weight = self.build_pretrained_embedding(path)

        print("Embedding weight shape:", self.embedding_weight.shape)
        print(self.embedding_weight)

    def build_pretrained_embedding(self, path):
        pretrained_file = os.path.join(path, 'sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5')
        pretrained_dict = {}
        embedding_dim = 300

        with open(pretrained_file, 'r', encoding='utf-8') as f:
            # 尝试读取首行看看是否是 "vocab_size embedding_dim"
            first_line = f.readline().strip().split()
            if len(first_line) == 2 and first_line[0].isdigit() and first_line[1].isdigit():
                vocab_count = int(first_line[0])
                embedding_dim = int(first_line[1])
            else:
                f.seek(0)

            for line in f:
                split_line = line.strip().split()
                # 行格式: word val1 val2 ... valN
                if len(split_line) == embedding_dim + 1:
                    w = split_line[0]
                    vec = list(map(float, split_line[1:]))
                    pretrained_dict[w] = vec

        # 调试输出：检查预训练词表大小和部分词条
        print(f"Loaded pretrained_dict with size: {len(pretrained_dict)}")
        if len(pretrained_dict) > 0:
            print("Sample of pretrained_dict keys:", list(pretrained_dict.keys())[:50])

        unk_vector = np.random.normal(scale=0.01, size=(embedding_dim,))
        pad_vector = np.zeros((embedding_dim,))

        vocab_size = len(self.dictionary.tkn2word)
        embedding_weight = np.zeros((vocab_size, embedding_dim))
        embedding_weight[0] = pad_vector

        match_count = 0
        for idx, word in enumerate(self.dictionary.tkn2word[1:], start=1):
            if word in pretrained_dict:
                embedding_weight[idx] = pretrained_dict[word]
                match_count += 1
            else:
                embedding_weight[idx] = unk_vector

        print(f"Matched {match_count}/{vocab_size} words in pretrained embeddings.")

        return torch.tensor(embedding_weight, dtype=torch.float)

    def pad(self, origin_token_seq):
        if len(origin_token_seq) > self.max_token_per_sent:
            return origin_token_seq[:self.max_token_per_sent]
        else:
            return origin_token_seq + [0 for _ in range(self.max_token_per_sent - len(origin_token_seq))]

    def tokenize(self, path, test_mode=False):
        idss = []
        labels = []
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                one_data = json.loads(line)
                sent = one_data['sentence']
                
                if isinstance(sent, str):
                    sent = list(jieba.cut(sent.strip()))

                for word in sent:
                    self.dictionary.add_word(word)

                ids = [self.dictionary.word2tkn[word] for word in sent]
                idss.append(self.pad(ids))

                if test_mode:
                    label = one_data['id']
                    labels.append(label)
                else:
                    label = one_data['label']
                    labels.append(self.dictionary.label2idx[label])

        idss = torch.tensor(np.array(idss))
        labels = torch.tensor(np.array(labels)).long()

        return TensorDataset(idss, labels)
