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
import string

class DatasetPruning_LSTM(Dataset):
    def __init__(self, data, rel2idx, idx2rel, word2idx):
        self.data = data
        self.rel2idx = rel2idx
        self.idx2rel = idx2rel
        self.word2idx = word2idx

    def __len__(self):
        return len(self.data)

    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        batch_size = len(indices)
        vec_len = len(self.rel2idx)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def __getitem__(self, index):
        data_point = self.data[index]
        encoded_question = self.encode_question(data_point[0])
        tail_onehot = self.toOneHot(data_point[1])
        return encoded_question, tail_onehot

    def encode_question(self, question):
        question = question.replace(',','')
        question = re.split('\s+', question.strip())
        encoded_question = [self.word2idx[word.strip()] for word in question]
        return encoded_question

    def tokenize_question(self, question):
        encoded_question = self.encode_question(question)
        return torch.tensor(encoded_question, dtype=torch.long), torch.tensor(len(encoded_question), dtype=torch.long)

    def data_generator(self, data):
        for i in range(len(data)):
            data_sample = data[i]
            encoded_question = self.encode_question(data_sample[0])
            yield torch.tensor(encoded_question, dtype=torch.long), torch.tensor(len(encoded_question), dtype=torch.long), data_sample[1]

def _collate_fn(batch):
        sorted_seq = sorted(batch, key=lambda sample: len(sample[0]), reverse=True)
        sorted_seq_lengths = [len(i[0]) for i in sorted_seq]
        longest_sample = sorted_seq_lengths[0]
        minibatch_size = len(batch)
        input_lengths = []
        p_tail = []
        inputs = torch.zeros(minibatch_size, longest_sample, dtype=torch.long)

        for x in range(minibatch_size):
            sample = sorted_seq[x][0]
            tail_onehot = sorted_seq[x][1]
            p_tail.append(tail_onehot)
            seq_len = len(sample)
            input_lengths.append(seq_len)
            sample = torch.tensor(sample, dtype=torch.long)
            sample = sample.view(sample.shape[0])
            inputs[x].narrow(0,0,seq_len).copy_(sample)

        return inputs, torch.tensor(input_lengths, dtype=torch.long), torch.stack(p_tail)


class DataLoaderPruning_LSTM(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoaderPruning_LSTM, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
