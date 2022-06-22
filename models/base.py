"""Base Knowledge Graph embedding model."""
from abc import ABC, abstractmethod

import torch
from torch import nn


class KGModel(nn.Module, ABC):
    """Base Knowledge Graph Embedding model class.

    Attributes:
        sizes: Tuple[int, int, int] with (n_entities, n_relations, n_entities)
        rank: integer for embedding dimension
        dropout: float for dropout rate
        gamma: torch.nn.Parameter for margin in ranking-based loss
        data_type: torch.dtype for machine precision (single or double)
        bias: string for whether to learn or fix bias (none for no bias)
        init_size: float for embeddings' initialization scale
        entity: torch.nn.Embedding with entity embeddings
        rel: torch.nn.Embedding with relation embeddings
        bh: torch.nn.Embedding with head entity bias embeddings
        bt: torch.nn.Embedding with tail entity bias embeddings
    """

    def __init__(self, args):
        """Initialize KGModel."""
        super(KGModel, self).__init__()
        self.rank = args.dim
        self.freeze = args.freeze
        self.dtype = args.dtype
        self.data_type = torch.double if self.dtype == 'double' else torch.float
        self.gamma = nn.Parameter(torch.Tensor([args.gamma]), requires_grad=False)
        self.entity = nn.Embedding(args.sizes[0], self.rank, dtype=self.data_type)
        self.rel = nn.Embedding(args.sizes[1], self.rank, dtype=self.data_type)
        self.bh = nn.Embedding(args.sizes[0], 1, dtype=self.data_type)
        self.bt = nn.Embedding(args.sizes[0], 1, dtype=self.data_type)
        if self.freeze == True:
            self.rel = self.rel.requires_grad_(False)
            self.entity = self.entity.requires_grad_(False)
            self.bh = self.bh.requires_grad_(False)
            self.bt = self.bt.requires_grad_(False)

    def get_rhs(self):
        return self.entity.weight

    def get_query(self, head):
        head_e = self.entity(head)
        if len(head_e.shape) == 1:
            head_e = head_e.unsqueeze(0)
        return head_e

    def get_embeddings(self, head, question):
        return self.get_query(head), question

    @abstractmethod
    def get_queries(self, head, question, evaluate = False):
        """Compute embedding and biases of queries.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
        Returns:
             lhs_e: torch.Tensor with queries' embeddings (embedding of head entities and relations)
             lhs_biases: torch.Tensor with head entities' biases
        """
        pass

    @abstractmethod
    def similarity_score(self, lhs_e, rhs_e):
        """Compute similarity scores or queries against targets in embedding space.

        Args:
            lhs_e: torch.Tensor with queries' embeddings
            rhs_e: torch.Tensor with targets' embeddings
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            scores: torch.Tensor with similarity scores of queries against targets
        """
        pass
