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

class Dataset_LSTM(Dataset):
    def __init__(self, data, word2idx, entity2idx, rel2idx):
        self.data = data
        self.entity2idx = entity2idx
        self.rel2idx = rel2idx
        self.word2idx = word2idx        

    def get_shape(self):
        return self.kg_size

    def __len__(self):
        return len(self.data)

    def answersToOneHot(self, indices):
        indices = torch.LongTensor(indices)
        one_hot = torch.DoubleTensor(len(self.entity2idx))
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def relationsToEncode(self, indicies):
        encoded = torch.FloatTensor(len(self.rel2idx))
        encoded.zero_()
        for i in indicies:
            encoded[i] = +1
        return encoded

    def __getitem__(self, index):
        data_point = self.data[index]
        question_text = data_point[1]
        question_ids = [self.word2idx[word] for word in question_text.split()]
        head_id = self.entity2idx[data_point[0].strip()]
        tail_ids = [self.entity2idx[t.strip()] for t in data_point[2] if t.strip() in self.entity2idx]
        tail_onehot = self.answersToOneHot(tail_ids)
        path_ids = [self.rel2idx[rel_name.strip()] for rel_name in data_point[3]]
        path_encoded = self.relationsToEncode(path_ids)
        
        return question_ids, head_id, tail_onehot, path_encoded

    def data_generator(self, data):
        for i in range(len(data)):
            data_sample = data[i]
            head = self.entity2idx[data_sample[0].strip()]
            question = data_sample[1].strip().split(' ')
            encoded_question = [self.word2idx[word.strip()] for word in question]
            if type(data_sample[2]) is str:
                ans = self.entity2idx[data_sample[2]]
            else:
                ans = [self.entity2idx[entity.strip()] for entity in list(data_sample[2]) if entity.strip() in self.entity2idx]

            yield torch.tensor(head, dtype=torch.long),torch.tensor(encoded_question, dtype=torch.long) , ans, torch.tensor(len(encoded_question), dtype=torch.long), data_sample[1]


def _collate_fn(batch):
        sorted_seq = sorted(batch, key=lambda sample: len(sample[0]), reverse=True)
        sorted_seq_lengths = [len(i[0]) for i in sorted_seq]
        longest_sample = sorted_seq_lengths[0]
        minibatch_size = len(batch)
        input_lengths = []
        p_head = []
        p_tail = []
        paths = []
        inputs = torch.zeros(minibatch_size, longest_sample, dtype=torch.long)

        for x in range(minibatch_size):
            sample = sorted_seq[x][0]
            seq_len = len(sample)

            p_head.append(sorted_seq[x][1])
            p_tail.append(sorted_seq[x][2])
            paths.append(sorted_seq[x][3])
            input_lengths.append(seq_len)

            sample = torch.tensor(sample, dtype=torch.long)
            sample = sample.view(sample.shape[0])
            inputs[x].narrow(0,0,seq_len).copy_(sample)

        return inputs, torch.tensor(input_lengths, dtype=torch.long), torch.tensor(p_head, dtype=torch.long), torch.stack(p_tail), torch.stack(paths)

class DataLoader_LSTM(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoader_LSTM, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
