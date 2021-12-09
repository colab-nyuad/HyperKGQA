import os
import argparse
from shutil import copyfile
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import random
import operator
from utils import *
import scipy.sparse.csgraph as csg
from numba import jit, cuda
from estimators.base_estimator import BaseEstimator


class KrackhardtEstimator(BaseEstimator):

    def __init__(self, triplets, entity2idx, rel2idx):
        super(KrackhardtEstimator, self).__init__(triplets, entity2idx, rel2idx)

    def compute_curvature_per_relation(self, samples, rel):
        mask = samples[:, 1] == rel
        samples = samples[mask, :]
        sub_graph = create_graph(samples, self.entity2idx, self.rel2idx, type='directed')
        A = nx.to_numpy_matrix(sub_graph, dtype=int).tolist()
        num = np.sum([A[i][j] * (1-A[j][i]) for i in range(len(A)) for j in range(len(A))])
        curv = num / np.sum(A)
        print(rel, curv)
        return curv


    def compute_graph_curvature(self, samples):
        for k,v in self.rel2idx.items():
           curv = self.compute_curvature_per_relation(samples, k)

