"""Knowledge Graph embedding model optimizer."""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
import pickle
from scipy import spatial
import networkx as nx
from collections import Counter
from utils.utils import get_relations_in_path
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from utils.utils import get_pool
import random
import operator

class QAOptimizer(object):
    """Knowledge Graph embedding model optimizer.
    KGOptimizers performs loss computations for one phase
    Attributes:
        model: models.base.KGModel
        regularizer: regularizers.Regularizer
        optimizer: torch.optim.Optimizer
        batch_size: An integer for the training batch size
    """

    def __init__(self, args, model, optimizer, regularizer, dataset, device):
        self.model = model
        self.hops = args.hops
        self.thresh = 0.5
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.reg_weight = args.reg
        self.dataset = dataset
        self.device = device
        self.batch_size = args.batch_size
        self.max_epochs = args.max_epochs
        self.gamma = args.rel_gamma

    def calculate_loss(self, question, head, tail, question_param, path):
        pred, weights = self.model.get_predictions(question, head, question_param)
        p_tail = tail

        if self.model.ls:
            p_tail = ((1.0-self.model.ls)*p_tail) + (1.0/p_tail.size(1))

        loss_scores = self.model.loss(pred, p_tail)

        #loss_weights = self.model.rel_loss(weights, path)
        #loss_scores += loss_weights

        if self.reg_weight > 0:
           loss_scores = loss_scores + self.regularizer.forward(self.model.parameters())

        return loss_scores

    def calculate_valid_loss(self, samples):
        data_gen = self.dataset.data_generator(samples)
        total_correct = 0

        for i in tqdm(range(len(samples))):
            d = next(data_gen)
            passed = 0
            head = d[0].to(self.device)
            question_tokenized = d[1].unsqueeze(0).to(self.device)
            ans = d[2]
            attention_mask = d[3].unsqueeze(0).to(self.device)

            if head < 0:
                continue

            scores, ww = self.model.get_score_ranked(head, question_tokenized, attention_mask)
            top_2 = torch.topk(scores, k=2, largest=True, sorted=True)
            top_2_idx = top_2[1].tolist()[0]
            head_idx = head.tolist()

            if top_2_idx[0] == head_idx:
                pred_ans = top_2_idx[1]
            else:
                pred_ans = top_2_idx[0]

            if type(ans) is int:
                ans = [ans]

            if pred_ans in ans:
                total_correct += 1

        accuracy = total_correct/len(samples)
        print(accuracy)

        return accuracy


    def compute_score(self, candidate_rels, pruning_rels):
        return len(candidate_rels & pruning_rels) / len(pruning_rels.union(candidate_rels))

    def compute_score_with_relation_matching(self, samples, G, model, dataset, idx2rel):
        data_gen = self.dataset.data_generator(samples)
        total_correct = 0

        for i in tqdm(range(len(samples))):
            d = next(data_gen)

            head = d[0].to(self.device)
            if head < 0: continue
            question_tokenized = d[1].unsqueeze(0).to(self.device)
            ans = d[2]
            attention_mask = d[3].unsqueeze(0).to(self.device)

            scores, _ = self.model.get_score_ranked(head, question_tokenized, attention_mask)
            scores, candidates =  torch.topk(scores, k=50, largest=True)
            scores = scores.squeeze(0).cpu().detach().numpy()
            candidates = candidates.squeeze(0).cpu().detach().numpy()

            if candidates[0] == head:
                candidates = candidates[1:]
                scores = scores[1:]

            question_embedding = self.model.get_question_embedding(question_tokenized, attention_mask)
            rel_scores = model.get_score_ranked(question_tokenized, attention_mask)

            pruning_rels_scores, pruning_rels_torch = torch.topk(rel_scores, 5)
            pruning_rels = [p.item() for s, p in zip(pruning_rels_scores, pruning_rels_torch) if s > self.thresh]

            if len(pruning_rels) > 0:
                for j, tail in enumerate(candidates):
                    candidate_rels, _ = get_relations_in_path(G, head.item(), tail)
                    rm_scores = []
                    for cr in candidate_rels:
                        rm_scores.append(self.compute_score(set(cr), set(pruning_rels)))
                    if len(rm_scores) > 0:
                        scores[j] = scores[j] + self.gamma*max(rm_scores)

            pred_ans = candidates[np.argmax(scores)]

            if type(ans) is int:
                ans = [ans]

            if pred_ans in ans:
                total_correct += 1

        accuracy = total_correct/len(samples)
        print(accuracy)
        return accuracy

    def train(self, loader, epoch):

        running_loss = 0

        for i_batch, a in enumerate(loader):

            question = a[0].to(self.device)
            question_param = a[1].to(self.device)
            head = a[2].to(self.device)
            tail = a[3].to(self.device)
            path = a[4].to(self.device)

            loss = self.calculate_loss(question, head, tail, question_param, path)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            loader.set_postfix(Loss=running_loss/((i_batch+1)*self.batch_size), Epoch=epoch)
            loader.set_description('{}/{}'.format(epoch, self.max_epochs))
            loader.update()

        return running_loss



