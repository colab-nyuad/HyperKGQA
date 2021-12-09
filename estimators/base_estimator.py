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
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseEstimator(nn.Module):

    def __init__(self, triplets, entity2idx, rel2idx):
        super(BaseEstimator, self).__init__()
        self.triplets = triplets
        self.entity2idx = entity2idx
        self.rel2idx = rel2idx

    @abstractmethod
    def compute_curvature_per_relation(self, sapmles, rel):
        pass

    @abstractmethod
    def compute_graph_curvature(self, samples):
        pass
