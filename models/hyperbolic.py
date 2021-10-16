"""Hyperbolic Knowledge Graph embedding models where all parameters are defined in tangent spaces."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.base import KGModel
from utils.euclidean import givens_rotations, givens_reflection
from utils.hyperbolic import mobius_add, expmap0, project, hyp_distance_multi_c

HYP_MODELS = ["RotH", "RefH", "AttH"]


class BaseH(KGModel):
    """Trainable curvature for each relationship."""

    def __init__(self, args, embedding_matrix, device):
        """Initialize a Complex KGModel."""
        super(BaseH, self).__init__(args, embedding_matrix)
        self.act = nn.Softmax(dim=1)
        self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).to(device)
 
    def similarity_score(self, lhs_e, rhs_e):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        lhs_e = lhs_e.float()

        return - hyp_distance_multi_c(lhs_e, rhs_e, c, eval_mode=True) ** 2

class RotH(BaseH):
    """Hyperbolic 2x2 Givens rotations"""

    def get_queries(self, head, question, c, rel_diag):
        """Compute embedding and biases of queries."""
        head_e, rel_e = self.get_embeddings(head, question)
        head_e = expmap0(head_e, c)
        tmp = torch.chunk(rel_e, 2, dim=1)
        rel1, rel2 = torch.chunk(rel_e, 2, dim=1)
        rel1 = expmap0(rel1, c)
        rel2 = expmap0(rel2, c)
        lhs = project(mobius_add(head_e, rel1, c), c)
        res1 = givens_rotations(rel_diag, lhs)
        res2 = mobius_add(res1, rel2, c)
        res2 = self.score_dropout(res2)

        return (res2, c)


class RefH(BaseH):
    """Hyperbolic 2x2 Givens reflections"""

    def get_queries(self, head, question, c, rel_diag):
        """Compute embedding and biases of queries."""

        head_e, rel_e = self.get_embeddings(head, question)
        rel, _ = torch.chunk(rel_e, 2, dim=1)
        rel = expmap0(rel, c)
        lhs = givens_reflection(rel_diag, head_e)
        lhs = expmap0(lhs, c)
        res = project(mobius_add(lhs, rel, c), c)
        res = self.score_dropout(res)
        return (res, c)


class AttH(BaseH):
    """Hyperbolic attention model combining translations, reflections and rotations"""

    def get_queries(self, head, question, c, rel_diag, context_vec):
        """Compute embedding and biases of queries."""

        head_e, rel_e = self.get_embeddings(head, question)        
        rot_mat, ref_mat = torch.chunk(rel_diag, 2, dim=1)
        rot_q = givens_rotations(rot_mat, head_e).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head_e).view((-1, 1, self.rank))
        cands = torch.cat([ref_q, rot_q], dim=1).double()
        context_vec = context_vec.view((-1, 1, self.rank)).double()
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        lhs = expmap0(att_q, c)
        rel, _ = torch.chunk(rel_e, 2, dim=1)
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)
        res = self.score_dropout(res)
            
        return (res, c)
