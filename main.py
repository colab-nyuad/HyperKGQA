import os
import random
import argparse
import yaml
from shutil import copyfile
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
from tqdm import tqdm
import argparse
import operator
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import networkx as nx
import ast
import json
import models
import optimizers

from kge.model import KgeModel
from qamodels import SBERT_QAmodel
from kge.util.io import load_checkpoint
from relation_matching_models import RelationMatchingModel
from utils.utils import *
from models import all_models
from dataloaders import CheckpointLoader, QADataset, QADataLoader
from optimizers import QAOptimizer, RelationMatchingOptimizer

import xgboost as xgb
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser(
    description="Graph Embedding for Question Answering over Knowledge Graphs"
)

parser.add_argument(
    "--dataset", default='MetaQA', help="KG dataset"
)

parser.add_argument(
    "--hops", type=int, default=0, help = "Number of edges to reach the answer"
)

parser.add_argument(
    "--kg_type", type=str, default='full', help = "Type of graph (full, sparse)"
)

parser.add_argument(
    "--model", default="ComplEx", choices=all_models, help="Embedding model"
)

parser.add_argument(
    "--regularizer", choices=["F2"], default="F2", help="Regularizer"
)

parser.add_argument(
    "--reg", default=0.0, type=float, help="Regularization weight"
)

parser.add_argument(
    "--optimizer", choices=["SparseAdam", "Adagrad", "Adam"], default="Adam", help="Optimizer"
)

parser.add_argument(
    "--max_epochs", default=100, type=int, help = "Maximum number of epochs to train"
)

parser.add_argument(
    "--patience", default=10, type=int, help="Number of epochs before early stopping"
)

parser.add_argument(
    "--valid_every", default=3, type=int, help="Number of epochs before validation"
)

parser.add_argument(
    "--batch_size", default=128, type=int, help="Batch size"
)

parser.add_argument(
    "--learning_rate", default=0.0005, type=float, help="Learning rate for KGQA"
)

parser.add_argument(
    "--freeze", default=False, type=bool, help="Freeze weights of trained embeddings"
)

parser.add_argument(
    '--use_cuda', type=bool, default=True, help = "Use gpu"
)

parser.add_argument(
    '--gpu', type=int, default=0, help = "Which gpu to use"
)

parser.add_argument(
    '--num_workers', type=int, default=1, help=""
)

parser.add_argument(
    "--dim", default=5, type=int, help="Embedding dimension"
)

parser.add_argument(
    '--labels_smoothing', type=float, default=0.0, help = "Perform label smoothing"
)

parser.add_argument(
    '--decay', type=float, default=1.0
)

parser.add_argument(
    '--checkpoint_type', default='ldh', choices=['libkge', 'ldh'], help = "Checkpoint type"
)

parser.add_argument(
    '--rel_gamma', type=float, default=0.0, help = "Hyperparameter for relation matching"
)

parser.add_argument(
    '--use_relation_matching', default=False, type=bool, help="Use relation matching"
)

# Exporting enviromental variables

data_path = os.environ['DATA_PATH']
checkpoints = os.environ['CHECKPOINTS']
kg_data_path = os.environ['KGDATAPATH']
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#-------------------------------------

def train(optimizer, model, data_loader, train_samples, valid_samples, test_samples, args, checkpoint_path):

    best_score = -float("inf")
    no_update = 0
    eps = 0.0001

    phases = ['train'] * args.valid_every
    phases.append('valid')

    for epoch in range(args.max_epochs):
        for phase in phases:

            if phase == 'train':
                model.train()
                loader = tqdm(data_loader, total=len(data_loader), unit="batches")
                score = optimizer.train(loader, epoch)

            elif phase=='valid':
                model.eval()
                score, _ = optimizer.calculate_valid_loss(valid_samples)

                if score > best_score + eps and epoch < args.max_epochs:
                    best_score = score
                    no_update = 0

                    print("Validation accuracy increased from previous epoch", score)
                    test_score, _ = optimizer.calculate_valid_loss(test_samples)
                    print('Test score for best valid so far:', test_score)
                    torch.save(model, '{}'.format(checkpoint_path))
                    print('Model saved')

                elif (score < best_score + eps) and (no_update < args.patience):
                    no_update +=1
                    print("Validation accuracy decreases to %f from %f, %d more epoch to check"%(score, best_score, args.patience-no_update))

                if no_update == args.patience or epoch == args.max_epochs-1:
                    print("Model has exceed patience or reached maximum epochs")
                    return


def evaluate_with_relation_matching(dataset_path, qa_optimizer, test_samples, pruning_model, dataset, entity2idx, rel2idx):
    triples = read_kg_triplets(dataset_path, type = 'train')
    G = create_graph(triples, entity2idx, rel2idx, type='directed')
    idx2rel = {v: k for k, v in rel2idx.items()}
    qa_optimizer.compute_score_with_relation_matching(test_samples, G, pruning_model, dataset, idx2rel)

if __name__ == "__main__":
    args = parser.parse_args()

    ## Loading trained pretrained KG embeddings
    embedding_path = ('{}/{}/{}/{}/{}'.format(checkpoints, args.dataset, args.kg_type, args.model, args.dim))
    dataset_path = "{}/{}/{}".format(kg_data_path, args.dataset, args.kg_type)
    loader = CheckpointLoader(embedding_path, dataset_path, args.checkpoint_type)
    device = torch.device(args.gpu if args.use_cuda else "cpu")
    entity2idx, rel2idx = loader.load_parameters(args)
    embed_model = getattr(models, args.model)(args, device)
    loader.load_data(args, entity2idx, rel2idx, embed_model)

    ## Reading QA dataset
    qa_dataset_path = '{}/{}'.format(data_path, args.dataset)
    train_data, valid_data, test_data = read_qa_dataset(args.hops, qa_dataset_path)
    train_samples = process_qa_file(train_data)
    valid_samples = process_qa_file(valid_data)
    test_samples = process_qa_file(test_data)

    ## Creating QA dataset 
    print('Creating QA dataset')
    word2idx,idx2word, max_len = get_vocab(np.vstack((np.vstack((train_samples, valid_samples)), test_samples)))
    vocab_size = len(word2idx)
    dataset = QADataset(train_samples, word2idx, entity2idx, rel2idx)
    data_loader = QADataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    ## Creating QA model
    print('Creating QA model')
    qa_model = SBERT_QAmodel(args, embed_model, vocab_size, rel2idx)
    qa_model.to(device)

    ## Creating QA optimizer
    print('Creating QA optimizer')
    regularizer = getattr(optimizers, args.regularizer)(args.reg)
    optimizer = getattr(torch.optim, args.optimizer)(qa_model.parameters(), lr=args.learning_rate)
    qa_optimizer = QAOptimizer(args, qa_model, optimizer, regularizer, dataset, device)

    ## Training QA model
    checkpoint_path =  "{}/{}_{}.pt".format(embedding_path, args.model, args.hops)
    train(qa_optimizer, qa_model, data_loader, train_samples, valid_samples, test_samples, args, checkpoint_path)
    
    if args.use_relation_matching == True:
        ## Train relation matching model
        qa_optimizer.model = torch.load(checkpoint_path).to(device)
        checkpoint_rels_path =  "{}/relation_matching_model_{}.pt".format(embedding_path, args.hops)
        model = RelationMatchingModel(args, rel2idx, vocab_size, qa_optimizer.model.ln_model).to(device)
        optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
        relation_matching_optimizer = RelationMatchingOptimizer(args, model, optimizer, regularizer, dataset, device)
        train(relation_matching_optimizer, model, data_loader, train_samples, valid_samples, test_samples, args, checkpoint_rels_path)
        
        relation_matching_model = torch.load(checkpoint_rels_path).to(device)
        evaluate_with_relation_matching(dataset_path, qa_optimizer, test_samples, relation_matching_model, dataset, entity2idx, rel2idx)
