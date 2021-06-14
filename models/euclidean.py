"""Euclidean Knowledge Graph embedding models where embeddings are in real space."""
import numpy as np
import torch
from torch import nn
from models.base import KGModel
from utils.euclidean import euc_sqdistance, givens_rotations, givens_reflection
from torch.nn import functional as F

EUC_MODELS = ["TransE", "CP", "SimplE", "DistMult", "RESCAL", "TuckER"]



class BaseE(KGModel):
    """Euclidean Knowledge Graph Embedding models.

    Attributes:
        sim: similarity metric to use (dist for distance and dot for dot product)
    """

    def __init__(self, args, embedding_matrix):
        super(BaseE, self).__init__(args, embedding_matrix)
        self.half = False

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        if self.sim == "dot":
            score = lhs_e @ rhs_e.transpose(0, 1)
            if self.half:
                score = score * 0.5
        else:
            score = - euc_sqdistance(lhs_e, rhs_e, eval_mode)
        return score


    def get_embeddings(self, head, question):
        head_e = self.get_query(head)
        head_e = self.ent_dropout(head_e)
        rel_e = self.rel_dropout(question)
        return head_e, rel_e


class TransE(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

    def __init__(self, args, embedding_matrix, device):
        super(TransE, self).__init__(args, embedding_matrix)
        self.sim = "dist"

    def get_queries(self, head, question):
        head_e = self.entity(head)
        lhs_e = head_e + question

        return lhs_e


class CP(BaseE):
    """Canonical tensor decomposition https://arxiv.org/pdf/1806.07297.pdf"""

    def __init__(self, args, embedding_matrix, device):
        super(CP, self).__init__(args, embedding_matrix)
        self.sim = "dot"

    def get_queries(self, head, question):
        """Compute embedding and biases of queries."""
        
        head_e, rel_e = self.get_embeddings(head, question)
        lhs_e = head_e * rel_e
        lhs_e = self.score_dropout(lhs_e)

        return lhs_e

class SimplE(BaseE):
    def __init__(self, args, embedding_matrix, device):
        super(SimplE, self).__init__(args, embedding_matrix)
        self.sim = "dot"
        self.half = True
    
    def get_queries(self, head, question):
        head_e, rel_e = self.get_embeddings(head, question)
        s = head_e * rel_e
        s = self.score_dropout(s)
        s_head, s_tail = torch.chunk(s, 2, dim=1)
        s = torch.cat([s_tail, s_head], dim=1)
        return s
