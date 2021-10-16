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

from kge.model import KgeModel
from kge.util.io import load_checkpoint
from kge.config import Config
from kge.dataset import Dataset

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
    "--dataset", default='MetaQA', help="KG dataset"
)

parser.add_argument(
    "--kg_type", type=str, default='full', help = "Type of graph (full, sparse)"
)

parser.add_argument(
    "--hops", type=int, default=0, help = "Number of edges to reach the answer"
)

parser.add_argument(
    "--models", nargs="+", default=["AttH", "RotH", "ComplEx", "TransE"], help="Embedding model"
)

parser.add_argument(
    "--dim", default=20, type=int, help="Embedding dimension"
)

parser.add_argument(
    '--qa_nn_type', default='LSTM', choices=['LSTM', 'RoBERTa']
)

parser.add_argument(
    '--step', default=0.0, type=float, help=''
)

data_path = os.environ['DATA_PATH']
kge_path = os.environ['KGE_PATH']
curvatures_path = os.environ['CURVS_PATH']
checkpoints = os.environ['CHECKPOINTS']

if __name__ == "__main__":

    args = parser.parse_args()
    dataset_path = "{}/data/{}_{}".format(kge_path, args.dataset, args.kg_type)
    qa_dataset_path = '{}/QA_data/{}'.format(data_path, args.dataset)
    train_triplets = read_kg_triplets(dataset_path, type = 'train')
    valid_triplets = read_kg_triplets(dataset_path, type = 'valid')
    test_triplets = read_kg_triplets(dataset_path, type = 'test')
    triplets = np.vstack((train_triplets, valid_triplets))
    triplets = np.vstack((triplets, test_triplets))
    rel2idx = read_inv_dict('{}/relation_ids.del'.format(dataset_path))
    idx2rel = read_dict('{}/relation_ids.del'.format(dataset_path))
    entity2idx = read_inv_dict('{}/entity_ids.del'.format(dataset_path))
    curv_estimator = SectionalEstimator(triplets, entity2idx, rel2idx)


    train_data, _, test_data = read_qa_dataset(args.hops, qa_dataset_path)
    train_samples, _ = process_text_file(train_data)
    test_samples, _ = process_text_file(test_data)
    rel_qq = group_questions_by_chain(qa_dataset_path, args.hops)

    ### compute curvature
    rel_qq_file = "{}/{}_{}_hops{}.csv".format(curvatures_path, args.dataset, args.kg_type, args.hops)
    '''
    with open(rel_qq_file, 'w') as f:
        writer = csv.writer(f)
        for r in rel_qq.keys():
            sr = r.split('|')
            sr_triplets = [t for t in triplets if t[1] in sr]
            curv, _ = curv_estimator.compute_curvature(sr_triplets)
            print(r, curv)
            writer.writerow([r, curv])
    '''
    ## read computed curvatures
    with open(rel_qq_file) as f:
        rel_qq_curvs = {k: float(v) for line in f for (k, v) in [line.strip().split(',')]}

    ## Create QA dataset
    word2idx,idx2word, max_len = get_vocab(train_samples)
    vocab_size = len(word2idx)
    dataset = getattr(dataloaders, 'Dataset_{}'.format(args.qa_nn_type))(train_samples, word2idx, entity2idx)
    data_loader = getattr(dataloaders, 'DataLoader_{}'.format(args.qa_nn_type))(dataset, batch_size=20, shuffle=True, num_workers=20)

    ## Creat QA model
    qa_models = []
    device = torch.device(0)
    for model in args.models:
        checkpoint_path = "{}/{}_{}_{}_{}_{}.pt".format(checkpoints, args.dataset, args.kg_type, model, args.dim, args.hops)
        qa_model = torch.load(checkpoint_path)
        qa_model.to(device)
        qa_models.append(qa_model)

    ## Create QA optimizer
    args.use_relation_matching = False
    args.max_epochs = 100
    args.batch_size = 100
    qa_optimizer = QAOptimizer(args, qa_models[0], None, None, dataset, device)

    rq_performance = {}
    #merge rel_qq with tets_samples
    for k,v in rel_qq.items():
        rq_performance[rel_qq_curvs[k]] = []
        cq_test_samples = [t for t in test_samples if t[1] in v]
        l = len(cq_test_samples)
        if l > 0:
            for qa_model in qa_models:
                qa_optimizer.model = qa_model
                score = qa_optimizer.calculate_valid_loss(cq_test_samples)
                rq_performance[rel_qq_curvs[k]].append(score * l)
            rq_performance[rel_qq_curvs[k]].append(l)


    rq_frame = pd.DataFrame.from_dict(rq_performance, orient='index').sort_index()
    columns = args.models
    columns.append('Len')
    rq_frame.columns = columns
    min_ = np.min(rq_frame.index)
    max_ = np.max(rq_frame.index)
    groupped_rq_frame = rq_frame.groupby(pd.cut(rq_frame.index, np.arange(min_, max_, args.step))).sum()
    groupped_rq_frame = groupped_rq_frame.loc[(groupped_rq_frame!=0).any(1)]
    groupped_rq_frame = groupped_rq_frame.iloc[:,:len(args.models)].div(groupped_rq_frame.Len, axis=0)
    groupped_rq_frame['index'] = groupped_rq_frame.index
    groupped_rq_frame['index'] =  groupped_rq_frame['index'].apply(lambda x: x.mid)
    groupped_rq_frame = groupped_rq_frame.set_index(groupped_rq_frame['index'])
    groupped_rq_frame = groupped_rq_frame.drop(['Len', 'index'], axis=1)
    print(groupped_rq_frame)
    sns.lineplot(data=groupped_rq_frame)
    plt.savefig("output.png")
