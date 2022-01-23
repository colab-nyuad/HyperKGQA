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

class SBERT_PruningModel(nn.Module):

    def __init__(self, args, rel2idx, vocab_size):
        super(SBERT_PruningModel, self).__init__()
        self.rel2idx = rel2idx
        self.ls = args.labels_smoothing

        self.sbert_model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

        self.sbert_dim = 768
        self.hidden2rel = nn.Linear(self.sbert_dim, len(self.rel2idx))
        self.loss = torch.nn.BCELoss(reduction='sum')

    def apply_nonLinear(self, input):
        outputs = self.hidden2rel(input)
        return outputs

        # SentenceTransformer Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


    def get_question_embedding(self, question_tokenized, attention_mask):
        output = self.sbert_model(question_tokenized, attention_mask)
        question_embedding = self.mean_pooling(output, attention_mask)
        return question_embedding


    def get_score_ranked(self, question_tokenized, attention_mask):
        question_embedding = self.get_question_embedding(question_tokenized, attention_mask)
        prediction = self.apply_nonLinear(question_embedding)
        prediction = torch.sigmoid(prediction).squeeze()
        return prediction


