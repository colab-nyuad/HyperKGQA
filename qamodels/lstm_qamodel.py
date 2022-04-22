import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
import torch.nn.utils
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn.init import xavier_normal_
from abc import ABC, abstractmethod
from transformers import *
import random
from qamodels.base_qamodel import Base_QAmodel

class LSTM_QAmodel(Base_QAmodel):

    def __init__(self, args, model, vocab_size):

        super(LSTM_QAmodel, self).__init__(args, model, vocab_size)

        self.n_layers = 1
        self.bidirectional = True
        self.hidden_dim = 256

        self.mid1 = 256
        self.mid2 = 256

        self.ln_model = nn.LSTM(self.rank, self.hidden_dim, self.n_layers, bidirectional=self.bidirectional, batch_first=True)

        self.hidden_dim = self.hidden_dim * 2
        self.lin1 = nn.Linear(self.hidden_dim, self.mid1)
        self.lin2 = nn.Linear(self.mid1, self.mid2)

        if self.hyperbolic_layers:
            self.hidden2c = nn.Linear(self.mid2, 1)
            self.hidden2rel_diag = nn.Linear(self.mid2, self.relation_dim)
            self.relation_dim = self.relation_dim * 2

            if self.context_layer:
                self.hidden2context = nn.Linear(self.mid2, self.relation_dim // 2)
                self.hidden2rel_diag = nn.Linear(self.mid2, self.relation_dim)

        self.hidden2rel = nn.Linear(self.mid2, self.relation_dim)

    def apply_nonLinear(self, input):
        hidden = self.lin1(input)
        hidden = F.relu(hidden)
        hidden = self.lin2(hidden)
        hidden = F.relu(hidden)
        outputs = self.hidden2rel(hidden)

        if self.hyperbolic_layers:
            c = F.softplus(self.hidden2c(hidden))
            rel_diag = self.hidden2rel_diag(hidden)

            if self.context_layer:
                context_vec = self.hidden2context(hidden)
                return outputs, (c, rel_diag, context_vec)
            else:
                return outputs, (c, rel_diag)
        else:
            return outputs, None

    def get_question_embedding(self, question, question_len):
        question_len = question_len.cpu()
        embeds = self.word_embeddings(question)
        packed_output = pack_padded_sequence(embeds, question_len, batch_first=True)
        outputs, (hidden, cell_state) = self.ln_model(packed_output)
        question_embedding = torch.cat([hidden[0,:,:], hidden[1,:,:]], dim=-1)
        return question_embedding


    def get_predictions(self, question, head, question_len):
        pred = super().get_score_ranked(head, question, question_len)
        return torch.sigmoid(pred)
