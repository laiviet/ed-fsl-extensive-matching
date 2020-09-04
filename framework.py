import os
import sys

import torch
from torch import optim, nn
import random
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from models.sentence_encoder import LSTMEncoder, GRUEncoder
from models.transformer import TransformerSentenceEncoder
from models.gcn import GCNN, CNN
import datetime


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)


def euclidean(s1, s2):
    return torch.mean(torch.pow(s1 - s2, 2))


def kl(s1, s2):
    kl_loss = torch.nn.KLDivLoss()
    s1 = torch.softmax(s1, dim=-1)
    s2 = torch.softmax(s2, dim=-1)
    return kl_loss(torch.log(s1), s2) + kl_loss(torch.log(s2), s1)


def cosine(s1, s2):
    return 1 - torch.mean(torch.cosine_similarity(s1, s2, -1))


class FewShotREModel(nn.Module):
    def __init__(self, args):
        '''
        sentence_encoder: Sentence encoder

        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)

        self.args = args
        self.device = args.device
        self.cost = nn.CrossEntropyLoss()
        self.intra_cost = None
        self.inter_cost = None
        self.hidden_size = 100
        args.hidden_size = self.hidden_size

        if args.encoder == 'trans':
            print('Transforer Encoder')
            self.sentence_encoder = TransformerSentenceEncoder(args)
        elif args.encoder == 'gru':
            print('GRU Encoder')
            self.sentence_encoder = GRUEncoder(args)
        elif args.encoder == 'lstm':
            print('LSTM Encoder')
            self.sentence_encoder = LSTMEncoder(args)
        elif args.encoder == 'gcn':
            print('GCN Encoder')
            self.sentence_encoder = GCNN(args)
        else:
            print('CNN Encoder')
            # self.sentence_encoder = CNNSentenceEncoder(args)
            self.sentence_encoder = CNN(args)
        #
        #
        if self.args.betaf == 'euclidean':
            self.intra_distance = euclidean
        elif self.args.betaf == 'cosine':
            self.intra_distance = cosine
        elif self.args.gammaf == 'kl':
            self.intra_distance = kl
        else:
            self.intra_distance = self.learnable

        if self.args.gammaf == 'euclidean':
            self.inter_distance = euclidean
        elif self.args.gammaf == 'cosine':
            self.inter_distance = cosine
        elif self.args.gammaf == 'kl':
            self.inter_distance = kl
        else:
            self.inter_distance = self.learnable
        self.distance_fc = nn.Sequential(
            nn.Linear(4 * self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1),
        )
        self.init()

        print(self)

    def init(self):
        self.distance_fc.apply(init_weights)

    def forward(self, support, query, negative=None):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        return: logits, pred
        '''
        raise NotImplementedError

    def loss(self, label):
        # print('| framework > loss > self.logits', tuple(self.logits.shape))
        # print('| framework > loss > label', tuple(label.shape))
        logits = self.logits.view(-1, self.logits.shape[-1])
        targets = label.view(-1).to(self.device)

        return self.cost(logits, targets)

    def loloss(self):
        raise NotImplemented

    def learnable(self, a, b):
        x = torch.cat([a, b], dim=-1)
        return torch.sum(torch.sigmoid(self.distance_fc(x)))

    def intra_loss(self):
        B, N, K, D = self.support.shape

        support = self.support.view(-1, K, D)  # BN x K x D
        s1 = support.unsqueeze(dim=1).expand(-1, K, -1, -1)  # BN x K x K x D
        s2 = support.unsqueeze(dim=2).expand(-1, -1, K, -1)  # BN x K x K x D
        return self.intra_distance(s1, s2)

    def inter_loss(self):
        if len(self.prototypes.shape) ==3:
            B, N, D = self.prototypes.shape
            s1 = self.prototypes.unsqueeze(dim=1).expand(-1, N, -1, -1)  # B x N x N x D
            s2 = self.prototypes.unsqueeze(dim=2).expand(-1, -1, N, -1)  # B x N x N x D
            return self.inter_distance(s1, s2)
        else:
            B, NQ, N, D = self.prototypes.shape
            s1 = self.prototypes.unsqueeze(dim=2).expand(-1,-1, N, -1, -1)  # B x N x N x D
            s2 = self.prototypes.unsqueeze(dim=3).expand(-1,-1, -1, N, -1)  # B x N x N x D
            return self.inter_distance(s1, s2)


    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        acc = torch.mean((pred.view(-1).cpu() == label.view(-1).cpu()).type(torch.FloatTensor))
        acc = acc.detach().cpu().numpy()
        return acc

    def fscore(self, predictions, groundtruths):
        # print('| fscore > predictions: ', tuple(predictions.shape))
        # print('| fscore > groundtruths: ', tuple(groundtruths.shape))
        predictions = predictions.cpu().view(-1).numpy()
        groundtruths = groundtruths.cpu().view(-1).numpy()

        def transfer(v, t):
            y = []
            for x in v:
                if x < t:
                    y.append(x + 1)
                else:
                    y.append(0)
            return np.array(y)

        predictions = transfer(predictions, self.args.way)
        groundtruths = transfer(groundtruths, self.args.way)
        zeros = np.zeros(predictions.shape, dtype='int')
        numPred = np.sum(np.not_equal(predictions, zeros))
        numKey = np.sum(np.not_equal(groundtruths, zeros))
        predictedIds = np.nonzero(predictions)
        preds_eval = predictions[predictedIds]
        keys_eval = groundtruths[predictedIds]
        correct = np.sum(np.equal(preds_eval, keys_eval))
        # print('correct : {}, numPred : {}, numKey : {}'.format(correct, numPred, numKey))
        precision = correct / numPred if numPred > 0 else 0.0
        recall = correct / numKey
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0. else 0.0
        return f1


class FewShotREFramework:

    def __init__(self, model, train_data_loader, val_data_loader, test_data_loader, args):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.model = model
        self.device = model.device
        self.train_data_loader = iter(train_data_loader)
        self.val_data_loader = iter(val_data_loader)
        self.test_data_loader = iter(test_data_loader)
        self.args = args

        current_time = str(datetime.datetime.now().time())
        self.log_dir = 'logs/{}-{}-way-{}-shot-{}'.format(args.model, args.way, args.shot, current_time)
        # self.writer = SummaryWriter(log_dir=self.log_dir)
        self.init_optimizer()
        for k, v in self.args.__dict__.items():
            v = str(v) if not isinstance(v, str) else v
            # self.writer.add_text(tag=k, text_string=v, global_step=0)

    def get_main_setting(self):
        return self.args.way, self.args.shot, self.args.query

    def get_coefficient(self):
        return self.args.alpha, self.args.beta, self.args.gamma

    def init_optimizer(self):
        self.scheduler = None
        self.parameters_to_optimize = filter(lambda x: x.requires_grad, self.model.parameters())
        if self.args.optimizer == 'adadelta':
            print('Optimizer: Adadelta')
            self.optimizer = optim.Adadelta(self.parameters_to_optimize, self.args.lr)
        else:
            print('Optimizer: SGD')
            self.optimizer = optim.SGD(self.parameters_to_optimize, self.args.lr, weight_decay=self.args.lr)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.lr_step_size)

    def train(self, train_iter=5000, val_iter=1000, test_iter=1000, val_step=200):
        """

        :param train_iter:
        :param val_iter:
        :param val_step:
        :param test_iter:
        :return:
        """
        self.model.train()
        alpha, beta, gamma = self.get_coefficient()
        best_val, best_test = 0.0, 0.0
        ce, loloss, intra, inter = 0, 0, 0, 0
        print('Training with negative examples')
        for it in range(train_iter):
            support, query, negative, label = next(self.train_data_loader)
            if it == 0:
                print('-' * 80)
                for k, v in support.items():
                    print(k, v.shape)
                print('----')
                for k, v in query.items():
                    print(k, v.shape)
                print('----')
                for k, v in negative.items():
                    print(k, v.shape)
                print('----')
                print('target', label.shape)
                print('-' * 80)
            self.model.forward(support, query, negative)
            ce = self.model.loss(label)
            total = 0
            total += ce
            if alpha > 0.0:
                _loloss = self.model.loloss()
                scale = ce.detach() / _loloss.detach()
                loloss = alpha * _loloss * scale
                total += loloss
            if beta > 0.0:
                _intra = self.model.intra_loss()
                scale = ce.detach() / _intra.detach()
                intra = beta * _intra * scale
                total += intra
            if gamma > 0.0:
                _inter = self.model.inter_loss()
                scale = ce.detach() / _inter.detach()
                inter = gamma * _inter * scale
                total -= inter

            # print('@ {} | {} {} {} {}'.format(it, ce, loloss, intra, inter))

            self.optimizer.zero_grad()
            total.backward()
            # if self.args.model == 'relation':
            torch.nn.utils.clip_grad_norm_(self.parameters_to_optimize, 0.1)

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            if it % 50 == 0:
                print('@ {} | {:.5f} {:.5f} {:.5f} {:.5f}'.format(it, ce, loloss, intra, inter))

            if it % val_step == 0 and it > 0:
                self.model.eval()
                valid_acc = self.eval(self.val_data_loader, val_iter)
                test_acc = self.eval(self.test_data_loader, test_iter)
                # self.writer.add_scalar('loss/ce', ce, it)
                # self.writer.add_scalar('loss/lol', loloss, it)
                # self.writer.add_scalar('loss/intra', intra, it)
                # self.writer.add_scalar('loss/inter', inter, it)
                # self.writer.add_scalar('perf/valid', valid_acc, it)
                # self.writer.add_scalar('perf/test', test_acc, it)

                if valid_acc > best_val:
                    best_val = valid_acc
                    best_test = test_acc
                    print('> Valid/Test:\t{:.4f}\t{:.4f} -> Best'.format(valid_acc, test_acc))
                else:
                    print('> Valid/Test:\t{:.4f}\t{:.4f}'.format(valid_acc, test_acc))
                self.model.train()

        # self.writer.add_scalar('perf/best/valid', best_val)
        # self.writer.add_scalar('perf/best/test', best_test)

    def eval(self, dataloader, eval_iter):
        """

        :param dataloader:
        :param eval_iter:
        :return: accuracy or f-score
        """
        random.seed(3456)
        iter_right = 0.0
        iter_sample = 0.0
        for it in range(eval_iter):
            support, query, negative, label = next(dataloader)
            self.model(support, query, negative)
            right = self.model.fscore(self.model.pred, label)
            iter_right += right
            iter_sample += 1
        return iter_right / iter_sample * 100.0
