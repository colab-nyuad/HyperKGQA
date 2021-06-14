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

class Base_QAmodel(nn.Module):

    def __init__(self, args, model, vocab_size):
        super(Base_QAmodel, self).__init__()
        self.emb_model = model
        self.loss = torch.nn.BCELoss(reduction='sum')
        self.hidden_dim = args.hidden_dim
        self.ls = args.labels_smoothing
        self.freeze = args.freeze
        self.do_batch_norm = args.do_batch_norm
        self.relation_dim = args.dim
        self.rank = args.dim
        self.word_embeddings = nn.Embedding(vocab_size, self.rank)
        self.emb_model_name = model.__class__.__name__
        self.hyperbolic_layers = False
        self.context_layer = False

        self.act = nn.Softmax(dim=1)
        self.scale = torch.Tensor([1. / np.sqrt(self.relation_dim)]).cuda()

        
        if self.emb_model_name in ['RefH', 'RotH', 'AttH']:
            self.hyperbolic_layers = True
 
        if self.emb_model_name == 'AttH':
            self.context_layer = True

    @abstractmethod
    def apply_nonLinear(self, input):
        pass

    @abstractmethod
    def get_question_embedding(self, question, question_len):
        pass
    
    @abstractmethod
    def calculate_valid_loss(self, samples):
        pass

    @abstractmethod
    def calculate_loss(self, question, head, tail, question_len):
        pass

    def get_score_ranked(self, head, question, question_param, eval_mode=True):

        question_embedding = self.get_question_embedding(question, question_param)
        question_embedding, hyperbolic_layers = self.apply_nonLinear(question_embedding)
        
        if self.hyperbolic_layers:
            if self.context_layer:
                c, rel_diag, context_vec = hyperbolic_layers
                lhs_e = self.emb_model.get_queries(head, question_embedding, c, rel_diag, context_vec)
            else:
                c, rel_diag = hyperbolic_layers
                lhs_e = self.emb_model.get_queries(head, question_embedding, c, rel_diag)
        else:
            lhs_e = self.emb_model.get_queries(head, question_embedding)

        rhs_e = self.emb_model.get_rhs()
        scores = self.emb_model.similarity_score(lhs_e, rhs_e, eval_mode=eval_mode)

        return scores

