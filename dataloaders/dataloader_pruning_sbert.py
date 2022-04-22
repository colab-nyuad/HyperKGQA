import torch
import random
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import os
import unicodedata
import re
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from transformers import *


class DatasetPruning_SBERT(Dataset):
    def __init__(self, data, rel2idx, word2idx):
        self.data = data
        self.rel2idx = rel2idx
        self.tokenizer_class = RobertaTokenizer
        self.max_length = 128
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

    def __len__(self):
        return len(self.data)

    def pad_sequence(self, arr, max_len=128):
        num_to_add = max_len - len(arr)
        for _ in range(num_to_add):
            arr.append('<pad>')
        return arr

    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        batch_size = len(indices)
        vec_len = len(self.rel2idx)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def tokenize_question(self, question):
        encoded_que = self.tokenizer.encode_plus(question, padding='max_length', max_length=self.max_length, return_tensors='pt')
        question_tokenized, attention_mask = encoded_que['input_ids'][0], encoded_que['attention_mask'][0]
        return question_tokenized, attention_mask

    def __getitem__(self, index):
        data_point = self.data[index]
        question_text = data_point[0]
        question_tokenized, attention_mask = self.tokenize_question(question_text)
        rel_onehot = self.toOneHot(data_point[1])
        return question_tokenized, attention_mask, rel_onehot


    def data_generator(self, data):
        for i in range(len(data)):
            data_sample = data[i]
            question = data_sample[0]
            question_tokenized, attention_mask = self.tokenize_question(question)
            rel_idx = data_sample[1]
            yield question_tokenized, attention_mask, rel_idx

class DataLoaderPruning_SBERT(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoaderPruning_SBERT, self).__init__(*args, **kwargs)


