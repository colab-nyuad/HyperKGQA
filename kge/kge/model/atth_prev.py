import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from kge.job import Job
from kge.util.hyperbolic import *

class AttHScorer(RelationalScorer):
    r"""Implementation of the Hyperblic AttH KGE scorer.

    Reference: `_

    """

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        
        self.rank = self.get_option("relation_embedder.dim") // 2
        self.act = torch.nn.Softmax(dim=1)
        self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()


    def score_emb(self, s_emb, embeddings, o_emb, combine: str):

        if combine not in ["sp_", "spo"]:
            raise Exception(
               "Combine {} not supported in AttH's score function".format(combine)
           )

        p_emb = embeddings[:, :2*self.rank]
        rel_diag = embeddings[:, 2*self.rank:4*self.rank]
        context_vec = embeddings[:, 4*self.rank:5*self.rank]
        c = F.softplus(embeddings[:, 5*self.rank:])

        rot_mat, ref_mat = torch.chunk(rel_diag, 2, dim=1)
        rot_q = givens_rotations(rot_mat, s_emb).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, s_emb).view((-1, 1, self.rank))
        cands = torch.cat([ref_q, rot_q], dim=1)
        context_vec = context_vec.view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        lhs = expmap0(att_q, c)
        rel, _ = torch.chunk(p_emb, 2, dim=1)
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)
     
        score = - hyp_distance_(res, o_emb, c, combine=combine) ** 2
       
        return score.view(p_emb.size(0), -1)

class AttH(KgeModel):
    r"""Implementation of the Hyperbolic KGE model."""

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):

        self._init_configuration(config, configuration_key)

        self.set_option(
            "relation_embedder.dim", self.get_option("relation_embedder.dim") * 2
        )
        
        init_size = self.get_option('init_size')

        self.set_option(
            "relation_embedder.init_size", init_size
        )
        
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=AttHScorer,
            configuration_key=self.configuration_key,
            init_for_load_only=init_for_load_only,
        )

        self.set_option(
            "relation_embedder.dim", self.get_option("relation_embedder.dim") // 2
        )


    def score_so(self, s, o, p=None):
        raise Exception("The hyperbolic model cannot score relations.")


    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        # We overwrite this method to ensure that AttH only predicts towards objects.
        if direction == "o":
            return super().score_spo(s, p, o, direction)
        else:
            raise ValueError("AttH can only score objects")
