import torch
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_normal_
from transformers import *

class RoBERTa_PruningModel(nn.Module):

    def __init__(self, args, rel2idx, idx2rel, vocab_size):
        super(RoBERTa_PruningModel, self).__init__()
        self.rel2idx = rel2idx
        self.idx2rel = idx2rel
        self.ls = args.labels_smoothing

        self.roberta_pretrained_weights = 'roberta-base'
        self.roberta_model = RobertaModel.from_pretrained(self.roberta_pretrained_weights)

        self.roberta_dim = 768
        self.hidden2rel = nn.Linear(self.roberta_dim, len(self.rel2idx))
        self.loss = torch.nn.BCELoss(reduction='sum')

    def apply_nonLinear(self, input):
        outputs = self.hidden2rel(input)
        return outputs

    def get_question_embedding(self, question_tokenized, attention_mask):
        roberta_last_hidden_states = self.roberta_model(question_tokenized, attention_mask=attention_mask)[0]
        states = roberta_last_hidden_states.transpose(1,0)
        question_embedding = states[0]
        return question_embedding    

    def get_score_ranked(self, question_tokenized, attention_mask):
        question_embedding = self.get_question_embedding(question_tokenized, attention_mask)
        prediction = self.apply_nonLinear(question_embedding)
        prediction = torch.sigmoid(prediction).squeeze()
        return prediction
