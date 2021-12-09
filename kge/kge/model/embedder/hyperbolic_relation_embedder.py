from torch import Tensor
import torch.nn
import torch.nn.functional

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeEmbedder
from kge.model import LookupEmbedder
from kge.misc import round_to_points

from typing import List


class HyperbolicRelationEmbedder(LookupEmbedder):
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
        self.use_context = self.get_option("use_context")
        self.initialize = self.get_option("initialize")
        self.data_type = torch.double

        if self.use_context:
            # layer for attention weights
            self.context_vec = torch.nn.Embedding(vocab_size, self.rank, dtype=self.data_type)
            self.init(self.context_vec.weight.data)
            self.rank = self.rank * 2

        # diagonal scaling
        self.rel_diag = torch.nn.Embedding(vocab_size, self.rank, dtype=self.data_type)
        self.init(self.rel_diag.weight.data)

        # multiple curvatures per relation
        self.c_init = torch.ones((vocab_size, 1), dtype=self.data_type)
        self.c = torch.nn.Parameter(self.c_init, requires_grad=True)


    def init(self, weights):
        if self.initialize == 'normal_':
            torch.nn.init.normal_(weights, 
                                       mean=self.get_option("initialize_args.normal_.mean"), 
                                       std= self.get_option("initialize_args.normal_.std"))
        elif self.initialize == 'uniform_':
            torch.nn.init.uniform_(weights, a=self.get_option("initialize_args.uniform_.a"))
        elif self.initialize == 'xavier_normal_':
            torch.nn.init.xavier_normal_(weights, gain=self.get_option("initialize_args.xavier_normal_.gain")) 
        elif self.initialize == 'xavier_uniform_':
            torch.nn.init.xavier_uniform_(weights, gain=self.get_option("initialize_args.xavier_uniform_.gain"))

    def embed(self, indexes: Tensor) -> Tensor:
        indexes = indexes.long()

        embeddings = torch.cat([
            self._postprocess(self._embeddings(indexes)),
            self._postprocess(self.rel_diag(indexes)),
            self.c[indexes]], 1)
       
        if self.use_context:
            embeddings = torch.cat((embeddings, self._postprocess(self.context_vec(indexes))), 1)
 
        return embeddings
