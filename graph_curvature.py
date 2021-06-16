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
from utils.graph_helper import *
import scipy.sparse.csgraph as csg

parser = argparse.ArgumentParser(
    description="Gomputing curvature for Knowledge Graphs"
)

parser.add_argument(
    "--dataset", default='MetaQA', help="KG dataset"
)

parser.add_argument(
    "--kg_type", type=str, default='full', help = "Type of graph (full, sparse)"
)

parser.add_argument(
    "--curvature_type", type=str, default='krackhardt', choices=['krackhardt', 'global_curvature'], help = "Curvature type"
)

parser.add_argument(
    "--relation", type=str, default='all', help="Relation for which compute curvature"
)

qa_data_path = os.environ['DATA_PATH']
kge_path = os.environ['KGE_PATH']

def compute_krackhardt_hierarchy_curvature(samples, entity_dict):
    sub_graph = create_graph(samples, entity_dict)
    A = nx.to_numpy_matrix(sub_graph, dtype=int).tolist()
    num = np.sum([A[i][j] * (1-A[j][i]) for i in range(len(A)) for j in range(len(A))])
    denum = np.sum(A)
    curv = num / denum

    return curv

def compute_krackhardt_hierarchy_score(samples, relations_dict, entity_dict, relation):

    if relation is 'all':
       for k,v in relations_dict.items():
           mask = samples[:, 1] == k
           r_samples = samples[mask, :]
           curv = compute_krackhardt_hierarchy_curvature(r_samples, entity_dict)
           print(v, ': ', curv)
    else:
        print(compute_krackhardt_hierarchy_curvature(samples, entity_dict))


def sample(G, n_samples):

    H = nx.to_scipy_sparse_matrix(G)
    nodes = list(G)
    nodes.sort()
    n = H.shape[0]
    curvature = []
    max_iter = 10000
    iter = 0
    idx = 0

    while idx < n_samples:

        # if in max_iter we cannot sample a triangle check the diameter of the 
        # component, must be at least 3 to sample triangles
        if iter == max_iter:
            d = nx.algorithms.distance_measures.diameter(G)
            if d < 3: return None

        iter = iter + 1

        b = random.randint(0, n-1)
        c = random.randint(0, n-1)
        if b == c: continue
        
        path = nx.shortest_path(G, source=nodes[b], target=nodes[c])
        if len(path) < 3: continue

        middle = len(path) // 2
        m = nodes.index(path[middle])
        l_bc = len(path) - 1
        
        # sample reference node
        a = random.choice([l for l in list(range(n)) if l not in [m,b,c]])
            
        path = nx.shortest_path(G, source=nodes[a], target=nodes[b])
        l_ab = len(path) - 1
           
        path = nx.shortest_path(G, source=nodes[a], target=nodes[c])
        l_ac = len(path) - 1

        path = nx.shortest_path(G, source=nodes[a], target=nodes[m])
        l_am = len(path) - 1

        idx = idx + 1
        curv = (l_am**2 + l_bc**2 / 4 - (l_ab**2 + l_ac**2) / 2) / (2 * l_am)
        curvature.append(curv)
    
    return curvature

def compute_curvature(samples, entity_dict):
    sub_graph = create_graph(samples, entity_dict, type='undirected')
    components = [sub_graph.subgraph(c) for c in nx.connected_components(sub_graph)]
    nodes = [c.number_of_nodes()**3 for c in components]
    total = np.sum(nodes)
    weights = [n/total for n in nodes]
    curvs = [0]

    for idx,c in enumerate(components):
        weight = weights[idx]
        n_samples = int(1000 * weight)
        if n_samples > 0 and c.number_of_nodes() > 3:
            curv = sample(c, n_samples)
            if curv is not None:
                curvs.extend(curv)
    
    return np.mean(curvs), total

def compute_curvature_estimate(samples, relations_dict, entity_dict, relation):

    if relation is 'all':
       curvs = []
       for k,v in relations_dict.items():
           mask = samples[:, 1] == k
           r_samples = samples[mask, :]
           curv, n_nodes = compute_curvature(r_samples, entity_dict)
           curvs.append((curv, n_nodes))
           print(k, curv)

       # overall graph curvature
       total = np.sum([c[1] for c in curvs])
       graph_curv = np.sum([c[1]/total * c[0] for c in curvs])
       print("graph curvature", graph_curv)
    else:
        curv, _ = compute_curvature(samples, entity_dict)
        print(curv)




if __name__ == "__main__":
    
    args = parser.parse_args()
    dataset_path = "{}/data/{}_{}".format(kge_path, args.dataset, args.kg_type)

    ## Reading KG dataset
    triplets = read_kg_triplets(dataset_path, args.relation, type = 'train')
 
    with open('{}/relation_ids.del'.format(dataset_path)) as f:
        relations_dict = {v: int(k) for line in f for (k, v) in [line.strip().split(None, 1)]}

    with open('{}/entity_ids.del'.format(dataset_path)) as f:
        entity_dict = {v: int(k) for line in f for (k, v) in [line.strip().split(None, 1)]}

    
    if args.curvature_type == 'krackhardt':
        compute_krackhardt_hierarchy_score(triplets, relations_dict, entity_dict, args.relation)
    else:
        compute_curvature_estimate(triplets, relations_dict, entity_dict, args.relation)
