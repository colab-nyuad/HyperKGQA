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

    def __init__(self, args):
        super(BaseP, self).__init__(args)
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

    def __init__(self, args, device, sizes):
        super(DistMult, self).__init__(args, sizes)

    def get_queries(self, head, question):
        """Compute embedding and biases of queries."""
        head_e, rel_e = self.get_embeddings(head, question)
        lhs_e = head_e * rel_e
        return lhs_e


class CP(BaseP):
    
    def __init__(self, args, device, sizes):
        super(CP, self).__init__(args, sizes)
        self.half_dim = self.rank // 2

    def get_queries(self, head, question):
        """Compute embedding and biases of queries."""
        
        head_e, rel_e = self.get_embeddings(head, question)
        head_e = head_e[:, :self.half_dim]
        lhs_e = head_e * rel_e
        return lhs_e

class SimplE(BaseP):

    def __init__(self, args, device, sizes):
        super(SimplE, self).__init__(args, sizes)
        self.half = True

    def get_queries(self, head, question):
        head_e, rel_e = self.get_embeddings(head, question)
        lhs_e = head_e * rel_e
        s_head, s_tail = torch.chunk(lhs_e, 2, dim=1)
        lhs_e = torch.cat([s_tail, s_head], dim=1)
        return lhs_e


class TuckER(BaseP):
    def __init__(self, args, device, sizes):
        super(TuckER, self).__init__(args, sizes)
        t_shape = (self.rank, self.rank, self.rank)
        self.W = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, t_shape), dtype=self.data_type, device=device, requires_grad=True))

    def get_queries(self, head, rel_e):
        head_e = self.get_query(head)
        ent_dim = head_e.size(1)
        rel_dim = rel_e.size(1)

        x = head_e.view(-1, 1, ent_dim)
        W_mat = torch.mm(rel_e, self.W.view(rel_dim, -1))
        W_mat = W_mat.view(-1, ent_dim, ent_dim)
        x = torch.bmm(x, W_mat)
        x = x.view(-1, ent_dim)

        return lhs_e

class RESCAL(BaseP):

    def __init__(self, args, device, sizes):
        super(RESCAL, self).__init__(args, sizes)

    def get_queries(self, head, question):
        head_e, rel_e = self.get_embeddings(head, question)
        ent_dim = head_e.size(1)
        head_e = head_e.view(-1, 1, ent_dim)
        rel_e = rel_e.view(-1, ent_dim, ent_dim)
        lhs_e = torch.bmm(head_e, rel_e)
        lhs_e = lhs_e.view(-1, ent_dim)
        return lhs_e
