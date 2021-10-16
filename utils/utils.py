import torch, yaml, os
from shutil import copyfile
import networkx as nx
import numpy as np
from tqdm import tqdm
import pickle
from numpy import linalg as LA
import re

def str2bool(v):
    return int(v)


def prepare_embeddings(embedding_dict):
    entity2idx = {}
    idx2entity = {}
    i = 0
    embedding_matrix = []
    for key, entity in embedding_dict.items():
        entity2idx[key] = i
        idx2entity[i] = key
        i += 1
        embedding_matrix.append(entity)
    
    return (entity2idx, idx2entity, embedding_matrix)


def get_embeddings(embedder, dict_):
    e = {}
    f = open(dict_, 'r')
    for line in f:
        line = line[:-1].split('\t')
        ent_id = int(line[0])
        ent_name = line[1]
        e[ent_name] = embedder._embeddings(torch.LongTensor([ent_id]))[0]
    else:
        f.close()
        return e


def preprocess_entities_relations(kge_model, embedding_path):
    entity_dict = '{}/entity_ids.del'.format(embedding_path)
    embedder = kge_model._entity_embedder
    e = get_embeddings(embedder, entity_dict)

    relation_dict = '{}/relation_ids.del'.format(embedding_path)
    embedder = kge_model._relation_embedder
    if hasattr(embedder, 'base_embedder'):
        embedder = embedder.base_embedder
    print(embedder._embeddings)
    r = get_embeddings(embedder, relation_dict)
    return (e, r)


def get_vocab(data):
    word_to_ix = {}
    maxLength = 0
    idx2word = {}
    for d in data:
        sent = d[1]

        for word in sent.split():
            if word not in word_to_ix:
                idx2word[len(word_to_ix)] = word
                word_to_ix[word] = len(word_to_ix)
        length = len(sent.split())
        if length > maxLength:
            maxLength = length
    return (word_to_ix, idx2word, maxLength)


def process_text_file(text_file):
    data_array = []
    heads = []
    with open(text_file, 'r') as data_file:
        for data_line in data_file.readlines():
            data_line = data_line.strip()
            if data_line == '': continue
            data_line = data_line.strip().split('\t')
            
            # to ignore questions with missing answers
            if len(data_line) != 2: 
                print(data_line)
                continue
                        
            question = re.sub('\[.+\]', 'NE', data_line[0])
            head = data_line[0].split('[')[1].split(']')[0]
            ans = data_line[1].split('|')
            data_array.append([head, question.strip(), ans])
            heads.append(head)
                
        return data_array, set(heads)


def writeToFile(lines, fname):
    f = open(fname, 'w')
    for line in lines:
        f.write(line + '\n')
    else:
        f.close()
        print('Wrote to ', fname)


def read_qa_dataset(hops, dataset_path):
    if hops != 0:
        dataset_path += '/{}hop'.format(hops)
    train_data = '{}/train.txt'.format(dataset_path) 
    test_data = '{}/test.txt'.format(dataset_path)
    valid_data = '{}/valid.txt'.format(dataset_path)
    return (train_data, valid_data, test_data)

def generate_yaml(args, yaml_file):
    default_config = {}
    default_config['job'] = {'device': 'cuda' if args.use_cuda else 'cpu'}
    default_config['dataset'] = {'name': '{}_{}'.format(args.dataset, args.kg_type)}
    default_config['train'] = {}
    default_config['train']['optimizer'] = args.optimizer
    default_config['train']['max_epochs'] = args.max_epochs
    default_config['train']['batch_size'] = args.batch_size
    default_config['train']['optimizer_args'] = {'lr': args.learning_rate_kge}
    default_config['valid'] = {}
    default_config['valid']['every'] = args.valid_every
    default_config['valid']['early_stopping'] = {'patience': args.patience}
    default_config['eval'] = {'batch_size': args.batch_size}
    default_config['model'] = 'reciprocal_relations_model'
    default_config['reciprocal_relations_model'] = {}
    default_config['reciprocal_relations_model']['base_model'] = {'type' : args.model.lower()}
    default_config['lookup_embedder'] = {'dim':args.dim}
    with open(yaml_file, 'w') as (outfile):
        yaml.dump(default_config, outfile, default_flow_style=False)


def copy_embeddings(checkpoints_path, args, qa_data_path):
    emb_path = '{}/pretrained_models/{}'.format(qa_data_path, args.dataset)
    emb_dir = '{}/{}_{}_{}_{}'.format(emb_path, args.model, args.dataset, args.kg_type, args.dim)
    if not os.path.exists(emb_dir):
        os.makedirs(emb_dir)
    files_to_copy = ['checkpoint_best.pt', 'entity_ids.del', 'relation_ids.del']
    for f in files_to_copy:
        copyfile('{}/{}'.format(checkpoints_path, f), '{}/{}'.format(emb_dir, f))
    else:
        return emb_dir

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


def get_relations_in_path(G, head, tail, hops):
    try:
        shortest_path = nx.shortest_path(G, head, tail)
        if (hops == 0 and 2 <= len(shortest_path) <= 3) or (hops != 0 and len(shortest_path) == hops+1):
            pathGraph = nx.path_graph(shortest_path)
            relations = []
            for ea in pathGraph.edges():
                n_edges = G.number_of_edges(ea[0], ea[1])
                relations.extend([G.edges[ea[0], ea[1], i]['name'] for i in range(n_edges)])
            return relations
        else: return []
    except nx.exception.NetworkXNoPath:
        return []

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


def read_dict(dict_):
    with open(dict_) as f:
        dictionary_ = {int(k): v for line in f for (k, v) in [line.strip().split(None, 1)]}
    return dictionary_


def read_inv_dict(dict_):
    with open(dict_) as f:
        dictionary_ = {v: int(k) for line in f for (k, v) in [line.strip().split(None, 1)]}
    return dictionary_

def group_questions_by_chain(dataset_path, hops):
    if hops != 0: 
        dataset_path += '/{}hop'.format(hops)
    
    rel_qq = {}
    with open('{}/pruning_test.txt'.format(dataset_path), 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            question = re.sub('\[.+\]', 'NE', line[0])
            if line[1] in rel_qq.keys():
                rel_qq[line[1]].append(question)
            else:
                rel_qq[line[1]] = [question]

        return rel_qq

