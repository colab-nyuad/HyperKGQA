"""Euclidean Knowledge Graph embedding models where embeddings are in real space."""
import numpy as np
import torch
from torch import nn
from models.base import KGModel
from torch.nn import functional as F

COMP_MODELS = ["CP", "SimplE", "DistMult", "RESCAL", "TuckER"]


class BaseP(KGModel):
    """Compositional Knowledge Graph Embedding models.

    Attributes:
        sim: similarity metric to use (dot product)
    """

    def __init__(self, args, embedding_matrix):
        super(BaseP, self).__init__(args, embedding_matrix)
        self.half_dim = None
        self.half = False

    def similarity_score(self, lhs_e, rhs_e):
        """Compute similarity scores or queries against targets in embedding space."""
        if self.half_dim:
            rhs_e = rhs_e[:, self.half_dim:]
        
        score = lhs_e @ rhs_e.transpose(0, 1)
        
        if self.half:
            score = score * 0.5
        return score

class DistMult(BaseP):
    """DistMult"""

    def __init__(self, args, embedding_matrix, device):
        super(DistMult, self).__init__(args, embedding_matrix)

    def get_queries(self, head, question):
        """Compute embedding and biases of queries."""
        head_e, rel_e = self.get_embeddings(head, question)
        lhs_e = head_e * rel_e
        lhs_e = self.score_dropout(lhs_e)
        return lhs_e


class CP(BaseP):
    
    def __init__(self, args, embedding_matrix, device):
        super(CP, self).__init__(args, embedding_matrix)
        self.half_dim = self.rank // 2

    def get_queries(self, head, question):
        """Compute embedding and biases of queries."""
        
        head_e, rel_e = self.get_embeddings(head, question)
        head_e = head_e[:, :self.half_dim]
        lhs_e = head_e * rel_e
        lhs_e = self.score_dropout(lhs_e)
        return lhs_e

class SimplE(BaseP):

    def __init__(self, args, embedding_matrix, device):
        super(SimplE, self).__init__(args, embedding_matrix)
        self.half = True

    def get_queries(self, head, question):
        head_e, rel_e = self.get_embeddings(head, question)
        lhs_e = head_e * rel_e
        lhs_e = self.score_dropout(lhs_e)
        s_head, s_tail = torch.chunk(lhs_e, 2, dim=1)
        lhs_e = torch.cat([s_tail, s_head], dim=1)
        return lhs_e


class TuckER(BaseP):
    def __init__(self, args, embedding_matrix, device):
        super(TuckER, self).__init__(args, embedding_matrix)
        t_shape = (self.rank, self.rank, self.rank)
        self.W = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, t_shape), dtype=torch.float, device=device, requires_grad=True))

    def get_queries(self, head, rel_e):
        head_e = self.get_query(head)
        head_e = self.ent_dropout(head_e)
        ent_dim = head_e.size(1)
        rel_dim = rel_e.size(1)

        x = head_e.view(-1, 1, ent_dim)
        W_mat = torch.mm(rel_e, self.W.view(rel_dim, -1))
        W_mat = W_mat.view(-1, ent_dim, ent_dim)
        W_mat = self.rel_dropout(W_mat)
        x = torch.bmm(x, W_mat) 
        x = x.view(-1, ent_dim) 
        lhs_e = self.score_dropout(x)

        return lhs_e

class RESCAL(BaseP):

    def __init__(self, args, embedding_matrix, device):
        super(RESCAL, self).__init__(args, embedding_matrix)

    def get_queries(self, head, question):
        head_e, rel_e = self.get_embeddings(head, question)
        ent_dim = head_e.size(1)
        #head_e = head_e.unsqueeze(1)
        head_e = head_e.view(-1, 1, ent_dim)
        rel_e = rel_e.view(-1, ent_dim, ent_dim)
        lhs_e = torch.bmm(head_e, rel_e)
        lhs_e = lhs_e.view(-1, ent_dim)
        lhs_e = self.score_dropout(lhs_e)
        return lhs_e
