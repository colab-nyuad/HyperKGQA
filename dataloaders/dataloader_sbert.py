import torch
import random
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import os
import unicodedata
import re
import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from transformers import *


class Dataset_SBERT(Dataset):
    def __init__(self, data, word2idx, vec_len, entity2idx, rel2idx):
        self.data = data
        self.word2idx = word2idx
        self.entity2idx = entity2idx
        self.rel2idx = rel2idx
        self.max_length = 64
        self.vec_len = vec_len
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

    def get_shape(self):
        return self.kg_size

    def __len__(self):
        return len(self.data)

    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        batch_size = len(indices)
        one_hot = torch.FloatTensor(self.vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def relationsToEncode(self, indicies):
        encoded = torch.FloatTensor(len(self.rel2idx))
        encoded.zero_()
        for i in indicies:
            encoded[i] += 1
        return encoded

    def __getitem__(self, index):
        data_point = self.data[index]
        question_text = data_point[1]
        question_tokenized, attention_mask = self.tokenize_question(question_text)
        head_id = self.entity2idx[data_point[0].strip()]
        tail_ids = [self.entity2idx[t.strip()] for t in data_point[2] if t.strip() in self.entity2idx]
        tail_onehot = self.toOneHot(tail_ids)
        path_ids = [self.rel2idx[rel_name.strip()] for rel_name in data_point[3]]
        path_encoded = self.relationsToEncode(path_ids)

        return question_tokenized, attention_mask, head_id, tail_onehot

    def tokenize_question(self, question):
        encoded_que = self.tokenizer.encode_plus(question, padding='max_length', max_length=self.max_length, return_tensors='pt')
        question_tokenized, attention_mask = encoded_que['input_ids'][0], encoded_que['attention_mask'][0]
        return question_tokenized, attention_mask

    def data_generator(self, data):
        for i in range(len(data)):
            data_sample = data[i]
            head = self.entity2idx[data_sample[0].strip()]
            question = data_sample[1]
            question_tokenized, attention_mask = self.tokenize_question(question)
            if type(data_sample[2]) is str:
                ans = self.entity2idx[data_sample[2]]
            else:
                 ans = [self.entity2idx[entity.strip()] for entity in list(data_sample[2]) if entity.strip() in self.entity2idx]

            yield torch.tensor(head, dtype=torch.long), question_tokenized, ans, attention_mask

class DataLoader_SBERT(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoader_SBERT, self).__init__(*args, **kwargs)

