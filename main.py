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

from kge.model import KgeModel
from kge.util.io import load_checkpoint
from kge.config import Config
from kge.dataset import Dataset

import dataloaders
import optimizers
import qamodels
import pruning_models
import models
from utils.utils import *
from models import all_models
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
    "--dim", default=200, type=int, help="Embedding dimension"
)

parser.add_argument(
    "--batch_size", default=128, type=int, help="Batch size"
)

parser.add_argument(
    "--kg_batch_size", default=512, type=int, help="Batch size for KG"
)

parser.add_argument(
    "--ent_dropout", default=0.0, type=float, help="Entity Dropout rate"
)

parser.add_argument(
    "--rel_dropout", default=0.0, type=float, help="Relation Dropout rate"
)

parser.add_argument(
    "--score_dropout", default=0.0, type=float, help="Score Dropout rate"
)

parser.add_argument(
    "--nn_dropout", default=0.0, type=float, help="NN Dropout rate"
)

parser.add_argument(
    "--learning_rate_kge", default=0.1, type=float, help="Learning rate for Embeddings"
)

parser.add_argument(
    "--learning_rate_kgqa", default=0.0005, type=float, help="Learning rate for KGQA"
)

parser.add_argument(
    "--hidden_dim", default=256, type=int, help="Hidden dimension"
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
    '--do_batch_norm', type=bool, default=False, help = "Do batch normalization"
)

parser.add_argument(
    '--labels_smoothing', type=float, default=0.0, help = "Perform label smoothing"
)

parser.add_argument(
    "--init_size", default=1e-3, type=float, help="Initial embeddings' scale"
)

parser.add_argument(
    '--decay', type=float, default=1.0
)

parser.add_argument(
    '--embeddings', type=str, help = "Path to the folder with computed embeddings for KG"
)

parser.add_argument(
    '--qa_nn_type', default='LSTM', choices=['LSTM', 'RoBERTa']
)

parser.add_argument(
    '--use_relation_matching', type=str, help = "Use relation matching for QA task"
)


# Exporting enviromental variables

data_path = os.environ['DATA_PATH']
checkpoints = os.environ['CHECKPOINTS']
kge_path = os.environ['KGE_PATH']
config_files = os.environ['CONFIG_FILES']

#-------------------------------------

def train(optimizer, model, scheduler, train_samples, valid_samples, test_samples, args, checkpoint_path):
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
                score = optimizer.calculate_valid_loss(valid_samples)

                if score > best_score + eps and epoch < args.max_epochs:
                    best_score = score
                    no_update = 0

                    print("Validation accuracy increased from previous epoch", score)
                    test_score = optimizer.calculate_valid_loss(test_samples)
                    print('Test score for best valid so far:', test_score)
                    torch.save(model, '{}'.format(checkpoint_path))
                    print('Model saved')

                elif (score < best_score + eps) and (no_update < args.patience):
                    no_update +=1
                    print("Validation accuracy decreases to %f from %f, %d more epoch to check"%(score, best_score, args.patience-no_update))

                if no_update == args.patience or epoch == args.max_epochs-1:
                    print("Model has exceed patience or reached maximum epochs")
                    return

if __name__ == "__main__":
    args = parser.parse_args()
    kge_data_path = "{}/data".format(kge_path)
    dataset_path = "{}/{}_{}".format(kge_data_path, args.dataset, args.kg_type)
    config_name = '{}_{}_{}_{}'.format(args.dataset, args.kg_type, args.model, args.dim)

    ## Checkpoints folder
    if not os.path.exists(checkpoints):
        os.makedirs(checkpoints)

    ## Compute embeddings for KG
    if args.embeddings:
        embedding_path = args.embeddings
    else:
        # Process dataset according to LibKGE format
        cmd = "python {}/preprocess/preprocess_default.py {}".format(kge_data_path, dataset_path)
        os.system(cmd)

        yaml_file = '{}/config.yaml'.format(dataset_path)
        config_file = '{}/{}/{}/{}/{}'.format(config_files, args.dataset, args.model, args.kg_type, config_name)

        # Check for configuration file
        if os.path.exists(config_file):
            copyfile(config_file, yaml_file)
            print("Found the corresponding configuration file ", config_file)

        # If not found create configuration file from arguments
        else:
            generate_yaml(args, yaml_file)
        
        cmd = "kge resume {}".format(dataset_path)
        os.system(cmd)

        # copy teh best checkpoint and remove files created by LibKGE
        embedding_path = copy_embeddings(dataset_path, args, data_path)
        cmd = "{}/clean.sh {}".format(kge_data_path, dataset_path)
        os.system(cmd)

    ## Reading QA dataset
    qa_dataset_path = '{}/QA_data/{}'.format(data_path, args.dataset)
    train_data, valid_data, test_data = read_qa_dataset(args.hops, qa_dataset_path)
    train_samples, heads = process_text_file(train_data)
    valid_samples, _ = process_text_file(valid_data)
    test_samples, _ = process_text_file(test_data)

    ## Loading trained KG embeddings 
    print('Loading kg embeddings from', embedding_path)
    checkpoint = ('{}/checkpoint_best.pt'.format(embedding_path))
    kge_checkpoint = load_checkpoint(checkpoint)
    kge_model = KgeModel.create_from(kge_checkpoint)
    print('Loading entities and relations')
    e, r = preprocess_entities_relations(kge_model, embedding_path)
    entity2idx, idx2entity, embedding_matrix = prepare_embeddings(e)
    rel2idx, idx2rel, relation_matrix = prepare_embeddings(r)
    print(len(relation_matrix))

    ## Create QA dataset 
    print('Process QA dataset')
    word2idx,idx2word, max_len = get_vocab(train_samples)
    vocab_size = len(word2idx)
    dataset = getattr(dataloaders, 'Dataset_{}'.format(args.qa_nn_type))(train_samples, word2idx, entity2idx)
    data_loader = getattr(dataloaders, 'DataLoader_{}'.format(args.qa_nn_type))(dataset, batch_size=args.batch_size, shuffle=True, num_workers=20)
    
    ## Creat QA model
    print('Creating QA model')
    device = torch.device(args.gpu if args.use_cuda else "cpu")
    embed_model = getattr(models, args.model)(args, embedding_matrix, device)
    qa_model = getattr(qamodels, '{}_QAmodel'.format(args.qa_nn_type))(args, embed_model, vocab_size)
    checkpoint_path =  "{}/{}_{}.pt".format(checkpoints, config_name, args.hops)
    qa_model.to(device)

    ## Create QA optimizer
    print('Creating QA optimizer')
    regularizer = getattr(optimizers, args.regularizer)(args.reg)
    optimizer = getattr(torch.optim, args.optimizer)(qa_model.parameters(), lr=args.learning_rate_kgqa)
    scheduler = ExponentialLR(optimizer, args.decay)
    qa_optimizer = QAOptimizer(args, qa_model, optimizer, regularizer, dataset, device)
    train(qa_optimizer, qa_model, scheduler, train_samples, valid_samples, test_samples, args, checkpoint_path)

    if args.use_relation_matching:
          checkpoint_path_pr =  "{}/{}_pruning_model_{}_{}.pt".format(checkpoints, args.dataset, args.kg_type, args.hops)
          train_samples_pr = read_pruning_file('train', qa_dataset_path, rel2idx, args.hops)
          pr_dataset = getattr(dataloaders, 'DatasetPruning_{}'.format(args.qa_nn_type))(train_samples_pr, rel2idx, idx2rel, word2idx)
          
          #-------- Train pruning model ------------#
          
          if not os.path.exists(checkpoint_path_pr):
              valid_samples_pr = read_pruning_file('valid', qa_dataset_path, rel2idx, args.hops)
              test_samples_pr = read_pruning_file('test', qa_dataset_path, rel2idx, args.hops)
              data_loader = getattr(dataloaders, 'DataLoaderPruning_{}'.format(args.qa_nn_type))(pr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=15)
              pr_model = (getattr(pruning_models, '{}_PruningModel'.format(args.qa_nn_type))(args, rel2idx, idx2rel, vocab_size)).to(device)
              optimizer = getattr(torch.optim, args.optimizer)(pr_model.parameters(), lr=args.learning_rate_kgqa)
              scheduler = ExponentialLR(optimizer, args.decay)
              pr_optimizer = PruningOptimizer(args, pr_model, optimizer, regularizer, pr_dataset, device, idx2rel)
              train(pr_optimizer, pr_model, scheduler, train_samples_pr, valid_samples_pr, test_samples_pr, args, checkpoint_path_pr)
          
          #-----------------------------------------#

          ## Load KG triplets
          train_triplets = read_kg_triplets(dataset_path, type = 'train')
          valid_triplets = read_kg_triplets(dataset_path, type = 'valid')
          test_triplets = read_kg_triplets(dataset_path, type = 'test')
          triplets = np.vstack((train_triplets, valid_triplets))
          triplets = np.vstack((triplets, test_triplets))
          G = create_graph(triplets, entity2idx, rel2idx)
          pruning_model = torch.load(checkpoint_path_pr).to(device)
          qa_optimizer.compute_accuracy(test_samples, G, pruning_model, pr_dataset, relation_matrix, idx2rel, idx2entity)
