"""Knowledge Graph embedding model optimizer."""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
import pickle

class RelationMatchingOptimizer(object):
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
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.dataset = dataset
        self.device = device
        self.batch_size = args.batch_size
        self.max_epochs = args.max_epochs
    
    def calculate_loss(self, question, question_param, rel_one_hot):
        pred = self.model.get_score_ranked(question, question_param)
        rel = rel_one_hot

        if self.model.ls:
            rel = ((1.0-self.model.ls)*rel) + (1.0/rel.size(1))

        loss = self.model.loss(pred, rel)
        return loss


    def calculate_valid_loss(self, samples, checkpoint_path = None):
        data_gen = self.dataset.data_generator(samples)
        total_correct = 0
        predicted_ans = []

        for i in tqdm(range(len(samples))):
            d = next(data_gen)

            question = d[1].unsqueeze(0).to(self.device)
            question_param = d[3].unsqueeze(0).to(self.device)
            rel_id_list = d[4]
            scores = self.model.get_score_ranked(question, question_param)
            pred = torch.topk(scores, k=1)[1]

            if pred in rel_id_list:
                total_correct = total_correct + 1
            
            predicted_ans.append(pred)

        print(total_correct)
        accuracy = total_correct/len(samples)
        return accuracy, predicted_ans

    def train(self, loader, epoch):

        running_loss = 0

        for i_batch, a in enumerate(loader):
            question = a[0].to(self.device)
            question_param = a[1].to(self.device)
            rel_one_hot = a[4].to(self.device)
            rel_one_hot[rel_one_hot > 1] = 1

            loss = self.calculate_loss(question, question_param, rel_one_hot)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            loader.set_postfix(Loss=running_loss/((i_batch+1)*self.batch_size), Epoch=epoch)
            loader.set_description('{}/{}'.format(epoch, self.max_epochs))
            loader.update()
        
        return running_loss

