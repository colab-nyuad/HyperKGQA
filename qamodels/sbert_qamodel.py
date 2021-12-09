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

class SBERT_QAmodel(Base_QAmodel):

    def __init__(self, args, model, vocab_size):

        super(SBERT_QAmodel, self).__init__(args, model, vocab_size)

        self.ln_model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

        for param in self.ln_model.parameters():
            param.requires_grad = True

        # Hidden dimension rank is fixed
        self.hidden_dim = 768
        self.lin_dim = 512

        self.lin1 = nn.Linear(self.hidden_dim, self.lin_dim)
        self.lin2 = nn.Linear(self.lin_dim, self.lin_dim)

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

    # SentenceTransformer Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


    def get_question_embedding(self, question_tokenized, attention_mask):
        output = self.ln_model(question_tokenized, attention_mask)
        question_embedding = self.mean_pooling(output, attention_mask)
        return question_embedding


    def get_predictions(self, question, head, attention_mask):
        pred = super().get_score_ranked(head, question, attention_mask)
        return pred
