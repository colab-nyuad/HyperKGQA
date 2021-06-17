import torch, yaml, os
from shutil import copyfile
import networkx as nx
import numpy as np
from tqdm import tqdm

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
    else:
        return (
         entity2idx, idx2entity, embedding_matrix)


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


def process_text_file(text_file, check_length, split=False):
    data_array = []
    with open(text_file, 'r') as data_file:
        for data_line in data_file.readlines():
            data_line = data_line.strip()
            if data_line == '':
                continue
            data_line = data_line.strip().split('\t')
            if check_length and len(data_line) != 2:
                continue

            question = data_line[0].split('[')
            question_1 = question[0]
            question_2 = question[1].split(']')
            head = question_2[0].strip()
            question_2 = question_2[1]
            question = question_1 + 'NE' + question_2
            ans = data_line[1].split('|')
            data_array.append([head, question.strip(), ans])
    
        if split == False:
            return data_array
        data = []
        for line in data_array:
            head = line[0]
            question = line[1]
            tails = line[2]
            for tail in tails:
                data.append([head, question, tail])
                
        return data


def writeToFile(lines, fname):
    f = open(fname, 'w')
    for line in lines:
        f.write(line + '\n')
    else:
        f.close()
        print('Wrote to ', fname)


def get_MetaQA_dataset(hops, qa_data_path, kg_type):
    hops = hops + 'hop'
    metaqa_data_path = '{}/QA_data/MetaQA'.format(qa_data_path)
    train_dp = '{}/qa_train_{}'.format(metaqa_data_path, hops)
    train_dp = train_dp + '_half.txt' if kg_type == 'half' else train_dp + '.txt'
    valid_dp = '{}/qa_dev_{}.txt'.format(metaqa_data_path, hops)
    test_dp = '{}/qa_test_{}.txt'.format(metaqa_data_path, hops)
    return (train_dp, valid_dp, test_dp, False)


def get_fbwq_dataset(qa_data_path):
    fbwq_data_path = '{}/QA_data/WebQuestionsSP'.format(qa_data_path)
    train_dp = '{}/qa_train_webqsp.txt'.format(fbwq_data_path)
    valid_dp = '{}/qa_test_webqsp.txt'.format(fbwq_data_path)
    test_dp = valid_dp
    return (train_dp, valid_dp, test_dp, True)


def read_qa_dataset(dataset, hops, qa_data_path, kg_type):
    if 'fbwq' in dataset:
        return get_fbwq_dataset(qa_data_path)
    if 'MetaQA' in dataset:
        return get_MetaQA_dataset(hops, qa_data_path, kg_type)
    print('Dataset is not supported, create a read function')


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
    emb_path = '{}/pretrained_models/embeddings/{}'.format(qa_data_path, args.dataset)
    emb_dir = '{}/{}_{}_{}_{}'.format(emb_path, args.model, args.dataset, args.kg_type, args.dim)
    if not os.path.exists(emb_dir):
        os.makedirs(emb_dir)
    files_to_copy = ['checkpoint_best.pt', 'entity_ids.del', 'relation_ids.del']
    for f in files_to_copy:
        copyfile('{}/{}'.format(checkpoints_path, f), '{}/{}'.format(emb_dir, f))
    else:
        return emb_dir

def read_kg_triplets(dataset, relation, type):
    file = '{}/{}.txt'.format(dataset, type)

    triplets = []
    with open(file, 'r') as data_file:
        for data_line in data_file.readlines():
            data = data_line.strip().split('\t')
            if relation == 'all' or (relation != 'all' and data[1] == relation):
                triplets.append(data)
        return np.array(triplets)

def create_graph(samples, entity_dict, type='directed'):
    if type is 'directed':
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    for t in tqdm(samples):
        e1 = entity_dict[t[0].strip()]
        e2 = entity_dict[t[2].strip()]
        G.add_node(e1)
        G.add_node(e2)
        G.add_edge(e1, e2, name=t[1], weight=1)
    return G


def get_relations_in_path(G, e1, e2):
    path = nx.shortest_path(G, e1, e2)
    print(path)
    return set(relations)
