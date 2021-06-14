import networkx as nx
from tqdm import tqdm



def krackhardt_hierarchy_score(dataset_path): 

   f = open('../../data/fbwq_full/train.txt', 'r')
triples = []
for line in f:
    line = line.strip().split('\t')
    triples.append(line)

G = nx.Graph()
for t in tqdm(triples):
    e1 = t[0]
    e2 = t[2]
    G.add_node(e1)
    G.add_node(e2)
    G.add_edge(e1, e2)

A = nx.adjacency_matrix(G)

