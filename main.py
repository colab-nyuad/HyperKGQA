import os
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

import dataloaders
import optimizers
import qamodels
import pruning_models
import models

from kge.model import KgeModel
from kge.util.io import load_checkpoint

from utils.utils import *
from models import all_models
from dataloaders import CheckpointLoader
from optimizers import QAOptimizer, PruningOptimizer

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
    "--regularizer", choices=["L3"], default="L3", help="Regularizer"
)

parser.add_argument(
    "--reg", default=0.0, type=float, help="Regularization weight"
)

parser.add_argument(
    "--optimizer", choices=["Adagrad", "Adam"], default="Adam", help="Optimizer"
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
    '--num_workers', type=int, default=15, help=""
)

parser.add_argument(
    "--dim", default=20, type=int, help="Embedding dimension"
)

parser.add_argument(
    '--labels_smoothing', type=float, default=0.0, help = "Perform label smoothing"
)

parser.add_argument(
    '--decay', type=float, default=1.0
)

parser.add_argument(
    '--qa_nn_type', default='LSTM', choices=['LSTM', 'RoBERTa', 'SBERT']
)

parser.add_argument(
    '--use_relation_matching', type=str, help = "Use relation matching for QA task"
)

parser.add_argument(
    '--checkpoint_type', default='libkge', choices=['libkge', 'ldh'], help = "Checkpoint type"
)


# Exporting enviromental variables

data_path = os.environ['DATA_PATH']
checkpoints = os.environ['CHECKPOINTS']
kg_data_path = os.environ['KGDATAPATH']
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#-------------------------------------

def train(optimizer, model, data_loader, scheduler, train_samples, valid_samples, test_samples, args, checkpoint_path):

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
                scheduler.step()

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

def evaluate_with_relation_matching(dataset_path, qa_optimizer, test_samples, pruning_model, dataset, entity2idx, rel2idx, relation_matrix):
    train_triplets = read_kg_triplets(dataset_path, type = 'train')
    valid_triplets = read_kg_triplets(dataset_path, type = 'valid')
    test_triplets = read_kg_triplets(dataset_path, type = 'test')
    triplets = np.vstack((train_triplets, valid_triplets))
    triplets = np.vstack((triplets, test_triplets))
    
    G = create_graph(triplets, entity2idx, rel2idx)
    idx2rel = {v: k for k, v in rel2idx.items()}
    qa_optimizer.compute_score_with_relation_matching(test_samples, G, pruning_model, dataset, relation_matrix, idx2rel)


def train_relation_matching_model(args, qa_dataset_path, device, rel2idx, word2idx, vocab_size):
    checkpoint_prm_path =  "{}/{}/{}/pruning_model_{}.pt".format(checkpoints, args.dataset, args.kg_type, args.hops)
    
    train_samples = read_pruning_file('train', qa_dataset_path, rel2idx, args.hops)
    valid_samples = read_pruning_file('valid', qa_dataset_path, rel2idx, args.hops)
    test_samples = read_pruning_file('test', qa_dataset_path, rel2idx, args.hops)

    dataset = getattr(dataloaders, 'DatasetPruning_{}'.format(args.qa_nn_type))(train_samples, rel2idx, word2idx)
    data_loader = getattr(dataloaders, 'DataLoaderPruning_{}'.format(args.qa_nn_type))(dataset, batch_size=args.batch_size, shuffle=True, num_workers=15)
    model = (getattr(pruning_models, '{}_PruningModel'.format(args.qa_nn_type))(args, rel2idx, vocab_size)).to(device)
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
    scheduler = ExponentialLR(optimizer, args.decay)
    optimizer = PruningOptimizer(args, model, optimizer, regularizer, dataset, device)
    train(optimizer, model, data_loader, scheduler, train_samples, valid_samples, test_samples, args, checkpoint_prm_path)

    return checkpoint_prm_path, dataset


if __name__ == "__main__":
    args = parser.parse_args()

    ## Reading QA dataset
    qa_dataset_path = '{}/{}'.format(data_path, args.dataset)
    train_data, valid_data, test_data = read_qa_dataset(args.hops, qa_dataset_path)
    train_samples, heads = process_text_file(train_data)
    valid_samples, _ = process_text_file(valid_data)
    test_samples, _ = process_text_file(test_data)

    ## Loading trained pretrained KG embeddings
    embedding_path = ('{}/{}/{}/{}/{}'.format(checkpoints, args.dataset, args.kg_type, args.model, args.dim))
    dataset_path = "{}/{}/{}".format(kg_data_path, args.dataset, args.kg_type)
    loader = CheckpointLoader(embedding_path, dataset_path, args.checkpoint_type)
    device = torch.device(args.gpu if args.use_cuda else "cpu")
    entity2idx, rel2idx = loader.load_checkpoint(args)
    embed_model = getattr(models, args.model)(args, device)
    loader.load_data(embed_model)

    ## Create QA dataset 
    print('Process QA dataset')
    word2idx,idx2word, max_len = get_vocab(train_samples)
    vocab_size = len(word2idx)
    dataset = getattr(dataloaders, 'Dataset_{}'.format(args.qa_nn_type))(train_samples, word2idx, entity2idx)
    data_loader = getattr(dataloaders, 'DataLoader_{}'.format(args.qa_nn_type))(dataset, batch_size=args.batch_size, shuffle=True, num_workers=20)
    
    ## Creat QA model
    print('Creating QA model')
    qa_model = getattr(qamodels, '{}_QAmodel'.format(args.qa_nn_type))(args, embed_model, vocab_size)
    qa_model.to(device)

    ## Create QA optimizer
    print('Creating QA optimizer')
    regularizer = getattr(optimizers, args.regularizer)(args.reg)
    optimizer = getattr(torch.optim, args.optimizer)(qa_model.parameters(), lr=args.learning_rate)
    scheduler = ExponentialLR(optimizer, args.decay)
    qa_optimizer = QAOptimizer(args, qa_model, optimizer, regularizer, dataset, device)

    ## Train the model
    checkpoint_path =  "{}/{}_{}.pt".format(embedding_path, args.model, args.hops)
    train(qa_optimizer, qa_model, data_loader, scheduler, train_samples, valid_samples, test_samples, args, checkpoint_path)

    if args.use_relation_matching:
        qa_optimizer.model = torch.load(checkpoint_path).to(device)
        checkpoint_prm_path, dataset = train_relation_matching_model(args, qa_dataset_path, device, rel2idx, word2idx, vocab_size)
        model = torch.load(checkpoint_prm_path).to(device)
        evaluate_with_relation_matching(kg_data_path, qa_optimizer, test_samples, model, dataset, entity2idx, rel2idx, relation_matrix)
