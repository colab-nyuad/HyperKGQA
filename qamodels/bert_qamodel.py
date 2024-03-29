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
from transformers import DistilBertModel, DistilBertConfig
from sentence_transformers import SentenceTransformer
from transformers import RobertaConfig, RobertaModel

class RoBERTa_QAmodel(Base_QAmodel):

    def __init__(self, args, model, vocab_size):

        super(RoBERTa_QAmodel, self).__init__(args, model, vocab_size)

        self.roberta_model = RobertaModel.from_pretrained('roberta-base')

        for param in self.roberta_model.parameters():
            param.requires_grad = True

        # Hidden dimension rank is fixed
        self.lin_dim = 768

        if self.hyperbolic_layers:
            self.hidden2c = nn.Linear(self.lin_dim, 1)
            self.hidden2rel_diag = nn.Linear(self.lin_dim, self.relation_dim)
            self.relation_dim = self.relation_dim * 2

            if self.context_layer:
                self.hidden2context = nn.Linear(self.lin_dim, self.relation_dim // 2)
                self.hidden2rel_diag = nn.Linear(self.lin_dim, self.relation_dim)

        self.hidden2rel = nn.Linear(self.lin_dim, self.relation_dim)

        self.loss_ = torch.nn.KLDivLoss(reduction='sum')

    def loss(self, scores, targets):
        return self.loss_(
            F.log_softmax(scores, dim=1), F.normalize(targets, p=1, dim=1)
        )


    def apply_nonLinear(self, hidden):
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

    def get_question_embedding(self, question_tokenized, attention_mask):
        roberta_last_hidden_states = self.roberta_model(question_tokenized, attention_mask)[0]
        states = roberta_last_hidden_states.transpose(1,0)
        question_embedding = states[0]
        return question_embedding


    def get_predictions(self, question, head, attention_mask):
        pred = super().get_score_ranked(head, question, attention_mask)
        return pred
