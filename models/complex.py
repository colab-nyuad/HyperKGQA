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

    def __init__(self, args, device):
        """Initialize a Complex KGModel."""
        super(BaseC, self).__init__(args)
        self.rank = self.rank // 2
        if self.freeze:
            self.embeddings = nn.ModuleList([nn.Embedding(args.sizes[0], 2 * self.rank, sparse=True, dtype=self.data_type).requires_grad_(False),
                                             nn.Embedding(args.sizes[1], 2 * self.rank, sparse=True, dtype=self.data_type).requires_grad_(False)])
        else:
            self.embeddings = nn.ModuleList([nn.Embedding(args.sizes[0], 2 * self.rank, sparse=True, dtype=self.data_type),
                                             nn.Embedding(args.sizes[1], 2 * self.rank, sparse=True, dtype=self.data_type).requires_grad_(False)])


    def similarity_score(self, lhs_e, rhs_e):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e = lhs_e[:, :self.rank], lhs_e[:, self.rank:]
        rhs_e = rhs_e[:, :self.rank], rhs_e[:, self.rank:]
        score = lhs_e[0] @ rhs_e[0].transpose(0, 1) + lhs_e[1] @ rhs_e[1].transpose(0, 1)
        return score

    def get_embeddings(self, head, question):
        head_e = self.embeddings[0](head)
        if len(head_e.shape) == 1:
            head_e = head_e.unsqueeze(0)
        head_e = head_e[:, :self.rank], head_e[:, self.rank:]
        rel_e = question[:, :self.rank], question[:, self.rank:]

        return head_e, rel_e

    def get_rhs(self):
        return self.embeddings[0].weight

class ComplEx(BaseC):
    """Simple complex model http://proceedings.mlr.press/v48/trouillon16.pdf"""

    def get_queries(self, head, question):
        """Compute embedding and biases of queries."""
        head_e, rel_e = self.get_embeddings(head, question)

        lhs_e = torch.cat([
            head_e[0] * rel_e[0] - head_e[1] * rel_e[1],
            head_e[0] * rel_e[1] + head_e[1] * rel_e[0]
        ], 1)

        return lhs_e

class RotatE(BaseC):
    """Rotations in complex space https://openreview.net/pdf?id=HkgEQnRqYQ"""

    def get_queries(self, head, question):
        """Compute embedding and biases of queries."""
        head_e, rel_e = self.get_embeddings(head, question)
        rel_norm = torch.sqrt(rel_e[0] ** 2 + rel_e[1] ** 2)
        cos = rel_e[0] / rel_norm
        sin = rel_e[1] / rel_norm
        lhs_e = torch.cat([
            head_e[0] * cos - head_e[1] * sin,
            head_e[0] * sin + head_e[1] * cos
        ], 1)

        return lhs_e
