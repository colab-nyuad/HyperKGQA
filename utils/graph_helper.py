# Should be moved to utility
from multiprocessing import Pool
import networkx as nx
import scipy.sparse.csgraph as csg
import logging
import numpy as np


def dist_sample_rebuild(dm, alpha):
    dist_mat = np.copy(dm)
    n,_ = dist_mat.shape    
  
    keep_dists = np.random.binomial(1,alpha,(n,n))
    
    # sample matrix:
    for i in range(n):
        for j in range(n):
            dist_mat[i,j] = -1 if keep_dists[i,j] == 0 and i!=j else dist_mat[i,j]
       
    # use symmetry first for recovery:
    for i in range(n):
        for j in range(i+1,n):
            if dist_mat[i,j] == -1 and dist_mat[j,i] > 0:
                dist_mat[i,j] = dist_mat[j,i]
            if dist_mat[j,i] == -1 and dist_mat[i,j] > 0:
                dist_mat[j,i] = dist_mat[i,j]
                
    # now let's rebuild it with triangle inequality:
    largest_dist = np.max(dist_mat)
    
    for i in range(n):
        for j in range(i+1,n):
            # missing distance:
            if dist_mat[i,j] == -1:
                dist = largest_dist

                for k in range(n):
                    if dist_mat[i,k] > 0 and dist_mat[j,k] > 0 and dist_mat[i,k]+dist_mat[j,k] < dist:
                        dist = dist_mat[i,k]+dist_mat[j,k]

                    dist_mat[i,j] = dist
                    dist_mat[j,i] = dist
    return dist_mat
