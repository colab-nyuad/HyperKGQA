import torch, yaml, os
from shutil import copyfile
import networkx as nx
import numpy as np
from tqdm import tqdm
import pickle
from numpy import linalg as LA
import re

def get_vocab(data):
    word2idx = {}
    idx2word = {}
    for d in data:
        sent = d[1]
        for word in sent.split():
            if word not in word2idx:
                idx2word[len(word2idx)] = word
                word2idx[word] = len(word2idx)
    return (word2idx, idx2word, len(word2idx))

def process_qa_file(text_file):
    data_array = []
    with open(text_file, 'r') as data_file:
        for data_line in data_file.readlines():
            data_line = data_line.strip()
            data_line = data_line.strip().split('\t')

            # to ignore questions with missing answers
            if len(data_line) != 3: 
                print(data_line)
                continue

            question = re.sub('\[.+\]', 'NE', data_line[0])
            head = data_line[0].split('[')[1].split(']')[0]
            ans = data_line[1].split('|')
            paths = data_line[2].split('|')
            data_array.append([head, question.strip(), ans, paths])

        return data_array


def read_qa_dataset(hops, dataset_path):
    if hops != 0:
        dataset_path += '/{}hop'.format(hops)
    train_data = '{}/train.txt'.format(dataset_path)
    test_data = '{}/test.txt'.format(dataset_path)
    valid_data = '{}/valid.txt'.format(dataset_path)
    return (train_data, valid_data, test_data)

def read_kg_triplets(dataset, type):
    file = '{}/{}.txt'.format(dataset, type)
    triplets = []
    with open(file, 'r') as data_file:
        for data_line in data_file.readlines():
            data = data_line.strip().split('\t')
            triplets.append(data)
        return np.array(triplets)


def create_graph(triplets, entity2idx, rel2idx, type='undirected'):
    G = nx.MultiGraph() if type == 'undirected' else nx.MultiDiGraph()
    return add_triplets_to_graph(G, triplets, entity2idx, rel2idx)

def add_triplets_to_graph(G, triplets, entity2idx, rel2idx, strip=False):
    for t in tqdm(triplets):
        if strip:
            t[0] = t[0].strip()
            t[2] = t[2].strip()
        e1 = entity2idx[t[0]]
        e2 = entity2idx[t[2]]
        G.add_node(e1)
        G.add_node(e2)
        G.add_edge(e1, e2, name=rel2idx[t[1]], weight=1)
    return G

def get_relations_in_path(G, head, tail):
    try:
        shortest_path = nx.shortest_path(G, head, tail)
        pathGraph = nx.path_graph(shortest_path)
        relations = []
        for ea in pathGraph.edges():
            n_edges = G.number_of_edges(ea[0], ea[1])
            relations.extend([G.edges[ea[0], ea[1], i]['name'] for i in range(n_edges)])
        return relations, pathGraph.edges()
    except nx.exception.NetworkXNoPath:
        return [], None

def read_pruning_file(type, dataset_path, rel2idx, hops):
    if hops != 0:
        dataset_path += '/{}hop'.format(hops)

    with open('{}/pruning_{}.txt'.format(dataset_path, type), 'r') as f:
        data = []
        for line in f:
            line = line.strip().split('\t')
            question = re.sub('\[.+\]', '', line[0])
            rel_list = line[1].split('|')
            rel_id_list = [rel2idx[rel] for rel in rel_list if rel in rel2idx]
            data.append([question, rel_id_list])
        return data

def load_dict(inst_dict):
    inst2idx = {}
    with open(inst_dict, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            line = line[:-1].split('\t')
            inst_id = int(line[0])
            inst_name = line[1]
            inst2idx[inst_name] = inst_id
    return inst2idx

