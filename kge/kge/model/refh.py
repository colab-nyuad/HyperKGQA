import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from kge.job import Job
from kge.util.hyperbolic import *

class RefHScorer(RelationalScorer):
    r"""Implementation of the Hyperblic AttH KGE scorer.

    Reference: `_

    """

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        
        self.rank = self.get_option("relation_embedder.dim") // 2

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        if combine not in ["sp_", "spo"]:
            raise Exception(
               "Combine {} not supported in RefH's score function".format(combine)
           )
        
        rel_emb = p_emb[:, :2*self.rank]
        rel_diag = p_emb[:, 2*self.rank:3*self.rank]
        c = F.softplus(p_emb[:, 3*self.rank:])

        h_emb = s_emb[:, :self.rank]
        hb = s_emb[:, self.rank:self.rank+1]

        t_emb = o_emb[:, :self.rank]
        tb = o_emb[:, self.rank+1:]
        t_emb = t_emb.to(torch.double)

        if combine == "sp_":
            tb = tb.t()
         
        rel, _ = torch.chunk(rel_emb, 2, dim=1)
        rel = expmap0(rel, c)
        lhs = givens_reflection(rel_diag, h_emb)
        lhs = expmap0(lhs, c)
        res = project(mobius_add(lhs, rel, c), c)
        score = - hyp_distance_(res, t_emb, c, combine=combine) ** 2 + hb + tb
        score = score.to(torch.float)

        return score.view(rel_emb.size(0), -1)

class RefH(KgeModel):
    r"""Implementation of the Hyperbolic KGE model."""

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):

        self._init_configuration(config, configuration_key)
        self.set_option("relation_embedder.dim", self.get_option("relation_embedder.dim") * 2)
        self.set_option("entity_embedder.dim", self.get_option("entity_embedder.dim") + 2)

        super().__init__(
            config=config,
            dataset=dataset,
            scorer=RefHScorer,
            configuration_key=self.configuration_key,
            init_for_load_only=init_for_load_only,
        )

        self.set_option("relation_embedder.dim", self.get_option("relation_embedder.dim") // 2)
        self.set_option("entity_embedder.dim", self.get_option("entity_embedder.dim") - 2)

    def score_so(self, s, o, p=None):
        raise Exception("The hyperbolic model cannot score relations.")


    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        # We overwrite this method to ensure that AttH only predicts towards objects.
        if direction == "o":
            return super().score_spo(s, p, o, direction)
        else:
            raise ValueError("RefH can only score objects")
