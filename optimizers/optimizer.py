"""Knowledge Graph embedding model optimizer."""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
import pickle
from scipy import spatial
import networkx as nx
from utils.utils import get_relations_in_path
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity

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
        self.thresh = 0.5 # 0.4 for freebase
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.dataset = dataset
        self.device = device
        self.batch_size = args.batch_size
        self.max_epochs = args.max_epochs
        self.gamma = 10 # 5 for half # 10 for full
        self.score_func = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.use_relation_matching = args.use_relation_matching
    
    def calculate_loss(self, question, head, tail, question_param):
        pred = self.model.get_predictions(question, head, question_param)
        p_tail = tail

        if self.model.ls:
            p_tail = ((1.0-self.model.ls)*p_tail) + (1.0/p_tail.size(1))
        loss = self.model.loss(pred, p_tail)

        if not self.model.freeze:
            loss = loss + self.regularizer.forward(self.model.emb_model.entity.weight)

        return loss

    def calculate_valid_loss(self, samples):
        data_gen = self.dataset.data_generator(samples)
        total_correct = 0
        predicted_answers = []

        for i in tqdm(range(len(samples))):
            d = next(data_gen)

            head = d[0].to(self.device)
            question_tokenized = d[1].unsqueeze(0).to(self.device)
            ans = d[2]
            attention_mask = d[3].unsqueeze(0).to(self.device)

            scores = self.model.get_score_ranked(head, question_tokenized, attention_mask)
            
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
            
            predicted_answers.append(pred_ans)

        print(total_correct)
        accuracy = total_correct/len(samples)
        return accuracy, predicted_answers

    def compute_score(self, candidate_rels, pruning_rels, relation_matrix):
        score = []
        for p in pruning_rels:
            score.append(np.sum([self.score_func(relation_matrix[p], relation_matrix[c]).item() for c in candidate_rels]) / len(candidate_rels))
        return np.sum(score)

    def compute_score_with_relation_matching(self, samples, G, model, dataset, relation_matrix, idx2rel): 
        data_gen = self.dataset.data_generator(samples) 
        total_correct = 0 
        if self.hops != 0:
            neighborhood = self.hops + 1
        else: neighborhood=2

        for i in tqdm(range(len(samples))):
            d = next(data_gen)

            head = d[0].to(self.device)
            question_tokenized = d[1].unsqueeze(0).to(self.device)
            ans = d[2]
            attention_mask = d[3].unsqueeze(0).to(self.device)

            scores = self.model.get_score_ranked(head, question_tokenized, attention_mask)
            scores, candidates =  torch.topk(scores, k=20, largest=True)
            scores = scores.squeeze(0).cpu().detach().numpy()
            candidates = candidates.squeeze(0).cpu().detach().numpy()
            
            question_tokenized, attention_mask = dataset.tokenize_question(samples[i][1].replace('NE', ''))
            question_tokenized = question_tokenized.unsqueeze(0).to(self.device)
            attention_mask = attention_mask.unsqueeze(0).to(self.device)
            rel_scores = model.get_score_ranked(question_tokenized, attention_mask)
            
            pruning_rels_scores, pruning_rels_torch = torch.topk(rel_scores, 5)
            pruning_rels = [p.item() for s, p in zip(pruning_rels_scores, pruning_rels_torch) if s > self.thresh]

            if candidates[0] == head:
                candidates = candidates[1:]
                scores = scores[1:]

            recomputed_scores = {}
            recomputed_scores[0] = scores[0]
            for i, tail in enumerate(candidates):
                candidate_rels = get_relations_in_path(G, head.item(), tail, neighborhood)
                if len(candidate_rels) > 0:
                    score = self.compute_score(candidate_rels, pruning_rels, relation_matrix)
                    recomputed_scores[i] = scores[i] + self.gamma * score

            pred_ans = candidates[max(recomputed_scores, key=recomputed_scores.get)]

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

            loss = self.calculate_loss(question, head, tail, question_param)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            loader.set_postfix(Loss=running_loss/((i_batch+1)*self.batch_size), Epoch=epoch)
            loader.set_description('{}/{}'.format(epoch, self.max_epochs))
            loader.update()
        
        return running_loss
