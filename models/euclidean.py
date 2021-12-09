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

    def __init__(self, args):
        super(BaseE, self).__init__(args)

    def similarity_score(self, lhs_e, rhs_e):
        """Compute similarity scores or queries against targets in embedding space."""
        score = - euc_sqdistance(lhs_e, rhs_e, eval_mode=True)
        return score
    

class TransE(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

    def __init__(self, args, device):
        super(TransE, self).__init__(args)

    def get_queries(self, head, question):
        head_e, rel_e = self.get_embeddings(head, question)
        lhs_e = head_e + rel_e
        return lhs_e
