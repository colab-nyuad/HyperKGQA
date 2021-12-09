import torch
import random
import os
import unicodedata
import re
import time
import json
import pickle

from utils.utils import extract_embeddings
from torch import nn
from kge.model import KgeModel
from kge.util.io import load_checkpoint
from collections import defaultdict
from tqdm import tqdm
import numpy as np

class CheckpointLoader():
    def __init__(self, embedding_path, kg_path, checkpoint_type):
        self.embedding_path = embedding_path
        self.kg_path = kg_path
        self.checkpoint_type = checkpoint_type

    def load_checkpoint(self, args):
        if self.checkpoint_type == 'ldh':
            return self.load_ldh_checkpoint(args)
        else:
            return self.load_libkge_checkpoint(args)
    
    def load_libkge_checkpoint(self, args):
        checkpoint = '{}/checkpoint_best.pt'.format(self.embedding_path)
        kge_checkpoint = load_checkpoint(checkpoint)
        kge_model = KgeModel.create_from(kge_checkpoint)

        entities_dict = '{}/entity_ids.del'.format(self.kg_path)
        bias = True if kge_model._entity_embedder.dim > args.dim else False
        entity2idx, idx2entity, self.embedding_matrix, self.bh, self.bt = extract_embeddings(kge_model._entity_embedder, entities_dict, bias=bias)

        relation_dict = '{}/relation_ids.del'.format(self.kg_path)
        rel2idx, idx2rel, self.relation_matrix, _, _ = extract_embeddings(kge_model._relation_embedder, relation_dict)

        self.freeze = args.freeze
        args.sizes = (len(self.embedding_matrix), len(self.relation_matrix) // 2, len(self.embedding_matrix))
        if hasattr(kge_model, 'init_size'):
            args.init_size = kge_model.init_size
        args.dtype = 'float'
        args.gamma = 0

        return entity2idx, rel2idx

    def load_ldh_checkpoint(self, args):
        config = json.load(open('{}/config.json'.format(self.embedding_path)))
    
        args.sizes = config['sizes']
        args.init_size = config['init_size']
        args.gamma = config['gamma']
        args.dtype = config['dtype']

        entity2idx = pickle.load(open('{}/entities_dict.pickle'.format(self.kg_path), 'rb'))
        rel2idx = pickle.load(open('{}/relations_dict.pickle'.format(self.kg_path), 'rb'))

        return entity2idx, rel2idx

    def load_data(self, embed_model):
        if self.checkpoint_type == 'ldh':
            checkpoint = '{}/checkpoint.pt'.format(self.embedding_path)
            embed_model.load_state_dict(torch.load(checkpoint))
            self.relation_matrix = embed_model.rel.weight.data.cpu()
        else:
            if hasattr(embed_model, 'embeddings'):
                embed_model.embeddings[0] = nn.Embedding.from_pretrained(torch.stack(self.embedding_matrix, dim=0), freeze = self.freeze)
                embed_model.embeddings[1] = nn.Embedding.from_pretrained(torch.stack(self.relation_matrix, dim=0), freeze = True)
            else:
                embed_model.entity = nn.Embedding.from_pretrained(torch.stack(self.embedding_matrix, dim=0), freeze = self.freeze)
                embed_model.rel = nn.Embedding.from_pretrained(torch.stack(self.relation_matrix, dim=0), freeze = True)
            
            if len(self.bh) > 0:
                embed_model.bh = nn.Embedding.from_pretrained(torch.stack(self.bh, dim=0), freeze = self.freeze)
                embed_model.bt = nn.Embedding.from_pretrained(torch.stack(self.bt, dim=0), freeze = self.freeze)

        return self.relation_matrix
