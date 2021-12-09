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

class DatasetPruning_RoBERTa(Dataset):
    def __init__(self, data, rel2idx, idx2rel, word2idx):
        self.data = data
        self.rel2idx = rel2idx
        self.idx2rel = idx2rel
        self.tokenizer_class = RobertaTokenizer
        self.pretrained_weights = 'roberta-base'
        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights, cache_dir='.')

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
        question = "<s> " + question + " </s>"
        question_tokenized = self.tokenizer.tokenize(question)
        question_tokenized = self.pad_sequence(question_tokenized, 64)
        question_tokenized = torch.tensor(self.tokenizer.encode(question_tokenized, add_special_tokens=False))
        attention_mask = []
        for q in question_tokenized:
            # 1 means padding token
            if q == 1:
                attention_mask.append(0)
            else:
                attention_mask.append(1)
        return question_tokenized, torch.tensor(attention_mask, dtype=torch.long)
    
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

class DataLoaderPruning_RoBERTa(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoaderPruning_RoBERTa, self).__init__(*args, **kwargs)
