import os
import argparse
from shutil import copyfile
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import random
import operator
from utils.utils import *
import scipy.sparse.csgraph as csg
from numba import jit, cuda
from random import choice
from estimators.base_estimator import BaseEstimator
from scipy.special import comb

class SectionalEstimator(BaseEstimator):

    def __init__(self, triplets, entity2idx, rel2idx):
        super(SectionalEstimator, self).__init__(triplets, entity2idx, rel2idx)

    def sample(self, G, n_samples):
        nodes = list(G)
        nodes.sort()
        curvature = []
        max_iter = 10000
        iter = 0
        idx = 0

        while idx < n_samples:

            # if in max_iter we cannot sample return 0
            if iter == max_iter:
                return [0]

            iter = iter + 1

            m = choice(nodes)
            ngh = list(G.neighbors(m))
            if len(ngh) < 2: continue

            b = choice(ngh)
            c = choice(ngh)
            if b == c: continue
        
            # sample reference node
            a = choice([l for l in nodes if l not in [m,b,c]])

            try:
                bc = len(nx.shortest_path(G, source=b, target=c)) - 1
                ab = len(nx.shortest_path(G, source=a, target=b)) - 1
                ac = len(nx.shortest_path(G, source=a, target=c)) - 1
                am = len(nx.shortest_path(G, source=a, target=m)) - 1

            except nx.NetworkXNoPath:
                continue

            curv = (am**2 + bc**2/4 - (ab**2 + ac**2) / 2) / (2 * am)
            curvature.append(curv)
            
            idx = idx + 1

        return curvature

    def compute_curvature(self, samples):
        sub_graph = create_graph(samples, self.entity2idx, self.rel2idx)
        components = [sub_graph.subgraph(c) for c in nx.connected_components(sub_graph)]
        nodes = [c.number_of_nodes()**3 for c in components]
        total = np.sum(nodes)
        weights = [n/total for n in nodes]
        curvs = [0]

        for idx,c in enumerate(components):
            weight = weights[idx]
            n_samples = int(1000 * weight)
            nv = c.number_of_nodes()
            if nv  > 3:
                # check ration n_samples & number of nodes
                if nv < 15:
                    combs = comb(nv, 4)
                    if combs < n_samples:
                        n_samples = combs
                curv = self.sample(c, n_samples)
            else: 
                curv = [0]
                #if curv is not None:
            curvs.extend(curv)

        return np.mean(curvs), total


    def compute_curvature_per_relation(self, samples, rel):
        mask = samples[:, 1] == rel
        samples = samples[mask, :]
        return self.compute_curvature(samples)

    def compute_graph_curvature(self, samples):
       curvs = []
       curvs_per_relation = {}
       for k,v in self.rel2idx.items():
           curv, n_nodes = self.compute_curvature_per_relation(samples, k)
           curvs.append((curv, n_nodes))
           curvs_per_relation[k] = curv

       # overall graph curvature
       sorted_curvs_per_relation = dict(sorted(curvs_per_relation.items(), key=lambda item: item[1]))
       for k,v in sorted_curvs_per_relation.items():
           print(k, v)
       total = np.sum([c[1] for c in curvs])
       cc = [c[1]/total * c[0] for c in curvs]
       graph_curv = np.sum(cc)
       print("graph curvature", graph_curv)
       return self.rel2idx.keys(), curvs
