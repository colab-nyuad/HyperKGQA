import torch
import random
import os
import unicodedata
import re
import time
import json
import pickle

from utils.utils import load_dict
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

    def load_parameters(self, args):
        if self.checkpoint_type == 'ldh':
            config = json.load(open('{}/config.json'.format(self.embedding_path)))
            args.sizes = config['sizes']
            args.init_size = config['init_size']
            args.gamma = config['gamma']
            args.dtype = config['dtype']
            entity2idx = pickle.load(open('{}/entities_dict.pickle'.format(self.kg_path), 'rb'))
            rel2idx = pickle.load(open('{}/relations_dict.pickle'.format(self.kg_path), 'rb'))
        else:
            entity2idx = load_dict('{}/entity_ids.del'.format(self.kg_path))
            rel2idx = load_dict('{}/relation_ids.del'.format(self.kg_path))
            args.sizes = (len(entity2idx), len(rel2idx) // 2, len(entity2idx))
            args.init_size = 1e-3
            args.dtype = 'float'
            args.gamma = 0
        
        rr = list(rel2idx.keys())
        for i, r in enumerate(rr):
            rel2idx[r + '_inv'] = i + len(rr)
        print(rel2idx)
        return entity2idx, rel2idx

    def load_data(self, args, entity2idx, rel2idx, embed_model):
        print('start loading')
        checkpoint = '{}/model.pt'.format(self.embedding_path)
        if self.checkpoint_type == 'ldh':
            checkpoint = '{}/model.pt'.format(self.embedding_path)
            embed_model.load_state_dict(torch.load(checkpoint))
        else:
            kge_checkpoint = load_checkpoint(checkpoint)
            kge_model = KgeModel.create_from(kge_checkpoint)
            print('checkpoint loaded')

            bias = True if kge_model._entity_embedder.dim > args.dim else False
            entity_embedder = kge_model._entity_embedder
            relation_embedder = kge_model._relation_embedder

            if hasattr(entity_embedder, 'base_embedder'):
                entity_embedder = entity_embedder.base_embedder
                relation_embedder = relation_embedder.base_embedder
            
            if bias == True:
                embed_model.bh = nn.Embedding.from_pretrained(entity_embedder._embeddings.weight[:, -2:-1], freeze = args.freeze)
                embed_model.bt = nn.Embedding.from_pretrained(entity_embedder._embeddings.weight[:, -1:], freeze = args.freeze)
            print('biases loaded')

            if hasattr(embed_model, 'embeddings'):
                embed_model.embeddings[0] = nn.Embedding.from_pretrained(entity_embedder._embeddings.weight, freeze = args.freeze)
                embed_model.embeddings[1] = nn.Embedding.from_pretrained(relation_embedder._embeddings.weight, freeze=args.freeze)
            else:
                embed_model.entity = nn.Embedding.from_pretrained(entity_embedder._embeddings.weight[:, :-2], freeze = args.freeze)
                embed_model.rel = nn.Embedding.from_pretrained(relation_embedder._embeddings.weight, freeze = args.freeze)

        print('embeddings and relations loaded')
