"""Euclidean Knowledge Graph embedding models where embeddings are in real space."""
import numpy as np
import torch
from torch import nn
from models.base import KGModel
from utils.euclidean import euc_sqdistance
from torch.nn import functional as F

EUC_MODELS = ["TransE"]



class BaseE(KGModel):
    """Euclidean Knowledge Graph Embedding models.

    Attributes:
        sim: similarity metric to use (dist for distance and dot for dot product)
    """

    def __init__(self, args, embedding_matrix):
        super(BaseE, self).__init__(args, embedding_matrix)
        self.half = False
        self._norm = 2

    def similarity_score(self, lhs_e, rhs_e):
        """Compute similarity scores or queries against targets in embedding space."""
        score = -torch.cdist(lhs_e, rhs_e, p=self._norm)
        return score
    

class TransE(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

    def __init__(self, args, embedding_matrix, device):
        super(TransE, self).__init__(args, embedding_matrix)

    def get_queries(self, head, question):
        head_e, rel_e = self.get_embeddings(head, question)
        lhs_e = head_e + rel_e
        lhs_e = self.score_dropout(lhs_e)

        return lhs_e
