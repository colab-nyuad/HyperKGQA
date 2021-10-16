"""Euclidean Knowledge Graph embedding models where embeddings are in complex space."""
import torch
from torch import nn
from models.base import KGModel

COMPLEX_MODELS = ["ComplEx", "RotatE"]


class BaseC(KGModel):
    """Complex Knowledge Graph Embedding models.

    Attributes:
        embeddings: complex embeddings for entities and relations
    """

    def __init__(self, args, embedding_matrix, device):
        """Initialize a Complex KGModel."""
        super(BaseC, self).__init__(args, embedding_matrix)
        self.rank = self.rank // 2
        self.multiplier = 2
        self.bn0 = nn.BatchNorm1d(self.multiplier)
        self.bn2 = nn.BatchNorm1d(self.multiplier)

    def similarity_score(self, lhs_e, rhs_e):
        """Compute similarity scores or queries against targets in embedding space."""
        rhs_e = rhs_e[:, :self.rank], rhs_e[:, self.rank:]
        return lhs_e[0] @ rhs_e[0].transpose(0, 1) + lhs_e[1] @ rhs_e[1].transpose(0, 1)

    def get_embeddings(self, head, question):
        head_e = self.get_query(head)
        head_e = torch.stack([head_e[:, :self.rank], head_e[:, self.rank:]], dim=1)

        if self.do_batch_norm:
            head_e = self.bn0(head_e)

        head_e = self.ent_dropout(head_e)
        head_e = head_e.permute(1, 0, 2)

        rel_e = self.rel_dropout(question)
        rel_e = rel_e[:, :self.rank], rel_e[:, self.rank:]

        return head_e, rel_e

class ComplEx(BaseC):
    """Simple complex model http://proceedings.mlr.press/v48/trouillon16.pdf"""

    def get_queries(self, head, question):
        """Compute embedding and biases of queries."""
        head_e, rel_e = self.get_embeddings(head, question)

        lhs_e = torch.stack([
            head_e[0] * rel_e[0] - head_e[1] * rel_e[1],
            head_e[0] * rel_e[1] + head_e[1] * rel_e[0]
        ], dim=1)

        if self.do_batch_norm:
            lhs_e = self.bn2(lhs_e)

        lhs_e = self.score_dropout(lhs_e)
        lhs_e = lhs_e.permute(1, 0, 2)

        return lhs_e


class RotatE(BaseC):
    """Rotations in complex space https://openreview.net/pdf?id=HkgEQnRqYQ"""

    def get_queries(self, head, question):
        """Compute embedding and biases of queries."""
        head_e, rel_e = self.get_embeddings(head, question)
        rel_norm = torch.sqrt(rel_e[0] ** 2 + rel_e[1] ** 2)
        cos = rel_e[0] / rel_norm
        sin = rel_e[1] / rel_norm
        lhs_e = torch.stack([
            head_e[0] * cos - head_e[1] * sin,
            head_e[0] * sin + head_e[1] * cos
        ], dim=1)

        if self.do_batch_norm:
            lhs_e = self.bn2(lhs_e)

        lhs_e = self.score_dropout(lhs_e)
        lhs_e = lhs_e.permute(1, 0, 2)


        return lhs_e
