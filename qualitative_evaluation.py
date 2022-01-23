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
import estimators
import pandas as pd

from estimators import SectionalEstimator
import dataloaders
import optimizers
import qamodels
from optimizers import QAOptimizer
import seaborn as sns
import matplotlib.pyplot as plt
import csv

parser = argparse.ArgumentParser(
    description="Gomputing curvature for Knowledge Graphs"
)

parser.add_argument(
    "--kg_type", type=str, default='full', help = "Type of graph (full, sparse)"
)

parser.add_argument(
    "--hops", type=int, default=0, help = "Number of edges to reach the answer"
)

parser.add_argument(
    "--model", default='RotH', choices=["RefH", "RotH", "AttH", "ComplEx"], help="Embedding model"
)

parser.add_argument(
    '--step', default=0, type=float, help=''
)

parser.add_argument(
    '--dtype', type=str, default='double'
)


data_path = os.environ['DATA_PATH']
checkpoints = os.environ['CHECKPOINTS']
kg_data_path = os.environ['KGDATAPATH']
curvatures_path = os.environ['CURVS_PATH']


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

if __name__ == "__main__":

    args = parser.parse_args()

    args.dataset = 'MetaQA'
    args.qa_nn_type = 'LSTM'
    args.reg = 0
    args.rel_gamma = 0
    args.similarity_type = ""
    dims = [20, 50, 100, 200, 400]

    kg_data_path = '{}/{}/{}'.format(kg_data_path, args.dataset, args.kg_type)
    train_triplets = read_kg_triplets(kg_data_path, type = 'train')
    valid_triplets = read_kg_triplets(kg_data_path, type = 'valid')
    test_triplets = read_kg_triplets(kg_data_path, type = 'test')
    triplets = np.vstack((train_triplets, valid_triplets))
    triplets = np.vstack((triplets, test_triplets))

    entity2idx = pickle.load(open('{}/entities_dict.pickle'.format(kg_data_path), 'rb'))
    rel2idx = pickle.load(open('{}/relations_dict.pickle'.format(kg_data_path), 'rb'))
    idx2rel = {v: k for k, v in rel2idx.items()}

    curv_estimator = SectionalEstimator(triplets, entity2idx, rel2idx)

    ## Reading QA dataset
    qa_dataset_path = '{}/{}'.format(data_path, args.dataset)
    train_data, _, test_data = read_qa_dataset(args.hops, qa_dataset_path)
    train_samples, heads = process_text_file(train_data)
    test_samples, _ = process_text_file(test_data)

    rel_qq = group_questions_by_chain(qa_dataset_path, args.hops)

    ### compute curvature
    rel_qq_file = "{}/{}_{}_hops_{}.csv".format(curvatures_path, args.dataset, args.kg_type, args.hops)

    with open(rel_qq_file, 'w') as f:
        writer = csv.writer(f)
        for r in rel_qq.keys():
            sr = r.split('|')
            sr_triplets = [t for t in triplets if t[1] in sr]
            curv, _ = curv_estimator.compute_curvature(sr_triplets)
            print(r, curv)
            writer.writerow([r, curv])

    ## read computed curvatures
    with open(rel_qq_file) as f:
        rel_qq_curvs = {k: float(v) for line in f for (k, v) in [line.strip().split(',')]}

    ## Create QA dataset
    word2idx, idx2word, vocab_size = get_vocab(train_samples)
    dataset = getattr(dataloaders, 'Dataset_{}'.format(args.qa_nn_type))(train_samples, word2idx, entity2idx)
    data_loader = getattr(dataloaders, 'DataLoader_{}'.format(args.qa_nn_type))(dataset, batch_size=20, shuffle=True, num_workers=20)

    ## Loading pretrained qa models
    checkpoint_path = '{}/{}/{}/'.format(checkpoints, args.dataset, args.kg_type)
    qa_models = []
    device = 'cuda:0'
    for dim in dims:
        qa_model = torch.load('{}/{}/{}/{}_{}.pt'.format(checkpoint_path, args.model, dim, args.model, args.hops))
        qa_model.emb_model.device = device
        qa_model.to(device)
        qa_models.append(qa_model)

    ## Create QA optimizer
    args.use_relation_matching = False
    args.max_epochs = 100
    args.batch_size = 100
    qa_optimizer = QAOptimizer(args, qa_models[0], None, None, dataset, device)

    rq_performance = {}
    total_length = len(test_samples)

    #merge rel_qq with tets_samples
    for k,v in rel_qq.items():
        rq_performance[rel_qq_curvs[k]] = []
        cq_test_samples = [t for t in test_samples if t[1] in v]
        l = len(cq_test_samples)
        if l > 0:
            for qa_model in qa_models:
                qa_optimizer.model = qa_model
                score, _ = qa_optimizer.calculate_valid_loss(cq_test_samples)
                rq_performance[rel_qq_curvs[k]].append(score * total_length)
            rq_performance[rel_qq_curvs[k]].append(l)


    rq_frame = pd.DataFrame.from_dict(rq_performance, orient='index').sort_index()
    columns = ['{}'.format(dim) for dim in dims]
    columns.append('Size')
    rq_frame.columns = columns
    print(rq_frame)
    if args.step != 0:
        min_ = np.min(rq_frame.index)
        max_ = np.max(rq_frame.index)
        groupped_rq_frame = rq_frame.groupby(pd.cut(rq_frame.index, np.arange(min_, max_, args.step))).sum()
        groupped_rq_frame = groupped_rq_frame.loc[(groupped_rq_frame!=0).any(1)]
        groupped_rq_frame = groupped_rq_frame.iloc[:,:len(dims)].div(groupped_rq_frame.Size, axis=0)
        groupped_rq_frame['index'] = groupped_rq_frame.index
        groupped_rq_frame['index'] =  groupped_rq_frame['index'].apply(lambda x: x.mid)
        rq_frame = groupped_rq_frame.set_index(groupped_rq_frame['index'])
        rq_frame = rq_frame.drop(['index'], axis=1)
        print(rq_frame)
    else:
        rq_frame = rq_frame.iloc[:,:len(dims)].div(rq_frame.Size, axis=0)
    print(rq_frame)
    rq_frame = rq_frame.drop(columns=['Size'])

    sns.lineplot(data=rq_frame)
    plt.xlabel("Curvature")
    plt.ylabel("Hits@1")
    plt.savefig("output.png")



