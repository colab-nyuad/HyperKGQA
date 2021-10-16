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

    def __init__(self, args, embedding_matrix):
        """Initialize KGModel."""
        super(KGModel, self).__init__()
        self.rank = args.dim
        self.freeze = args.freeze
        self.do_batch_norm = args.do_batch_norm
        self.entity = nn.Embedding.from_pretrained(torch.stack(embedding_matrix, dim=0), freeze = self.freeze)
        self.ent_dropout = nn.Dropout(args.ent_dropout)
        self.rel_dropout = nn.Dropout(args.rel_dropout)
        self.score_dropout = nn.Dropout(args.score_dropout)

    def get_rhs(self):
        return self.entity.weight

    def get_query(self, head):
        head_e = self.entity(head)
        if len(head_e.shape) == 1:
            head_e = head_e.unsqueeze(0)

        return head_e

    def get_embeddings(self, head, question):
        head_e = self.get_query(head)
        head_e = self.ent_dropout(head_e)
        rel_e = self.rel_dropout(question)
        return head_e, rel_e

    @abstractmethod
    def get_queries(self, head, question):
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
