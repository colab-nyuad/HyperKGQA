from torch import Tensor
import torch.nn
import torch.nn.functional

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeEmbedder
from kge.model import LookupEmbedder
from kge.misc import round_to_points

from typing import List


class HyperbolicEmbedder(LookupEmbedder):
    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key: str,
        vocab_size: int,
        init_for_load_only=False,
    ):
       

        super().__init__(
            config, dataset, configuration_key, vocab_size, init_for_load_only=init_for_load_only
        )        
 
        # since we doubled relation dim rank for lookup embedder
        self.rank = self.get_option("dim") // 2
        self.init_size = self.get_option("init_size")
        self.use_context = self.get_option("use_context")
        self.data_type = torch.float

        if self.use_context:
            # layer for attention weights
            self.context_vec = torch.nn.Embedding(vocab_size, self.rank)
            self.context_vec.weight.data = self.init_size * torch.randn((vocab_size, self.rank), dtype=self.data_type)   
            self.rank = self.rank * 2

        # diagonal scaling
        self.rel_diag = torch.nn.Embedding(vocab_size, self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((vocab_size, self.rank), dtype=self.data_type) - 1.0

        # multiple curvatures per relation
        self.c_init = torch.ones((vocab_size, 1), dtype=self.data_type)
        self.c = torch.nn.Parameter(self.c_init, requires_grad=True)


    def embed(self, indexes: Tensor) -> Tensor:
        indexes = indexes.long()
        
        embeddings = torch.cat([
            self._postprocess(self._embeddings(indexes)),
            self.rel_diag(indexes),
            self.c[indexes]], 1)
       
        if self.use_context:
            embeddings = torch.cat((embeddings, self.context_vec(indexes)), 1)
 
        return embeddings
