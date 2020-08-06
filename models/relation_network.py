import sys

sys.path.append('..')
import torch
from torch import nn
import framework
from torch.nn import functional as F


class RelationNetwork(framework.FewShotREModel):

    def __init__(self, args):
        framework.FewShotREModel.__init__(self, args)


    def loloss(self):
        """

        :return:
        """
        support = self.support[:, :, 2:, :]  # B x Ns x K-2 x D
        query = self.support[:, :, 0:2, :].contiguous()  # B x 2Ns x D
        logits, _ = self.do_relation(support, query)
        B, N, C = logits.shape
        targets = torch.LongTensor([x for x in range(C)] * (B * N // C)).to(self.device)

        # print('| proto > loloss > logits', tuple(self.logits.shape))
        # print('| proto > loloss > targets', tuple(targets.shape))

        logits = logits.view(-1, logits.shape[-1])
        targets = targets.view(-1).to(self.device)

        return self.cost(logits, targets)

    def do_relation(self, support, query):
        """

        :param support: B x Ns x K x D
        :param query:   B x Nq x Q x D
        :return:
        """

        # B, N, K, D = support.shape
        Ns = support.shape[1]
        B, Nq, Q, D = query.shape

        query = query.view(B, Nq * Q, D)

        prototypes = torch.mean(support, 2)  # (B, Ns, K, D) -> (B, Ns, D)
        _prototypes = prototypes.unsqueeze(dim=1).expand(-1, Nq * Q, -1, -1)  # B x NqQ x Ns x D
        query = query.unsqueeze(dim=2).expand(-1, -1, Ns, -1)  # B x NqQ x N x D

        # print('Relation: query', tuple(query.shape))
        # print('Relation: prototypes', tuple(prototypes.shape))

        abs = torch.abs(_prototypes-query)
        prod = _prototypes * query
        concatenate_rep = torch.cat([_prototypes, query, abs, prod], dim=3)

        logits = self.distance_fc(concatenate_rep).squeeze(dim=3)
        # print('RelationNetwork > logits: ', tuple(logits.shape))
        return logits, prototypes

    def do_relation_with_negative(self, support, query, negative):
        """

        :param support: B x Ns x K x D
        :param query:   B x Nq x Q x D
        :return:
        """

        # B, N, K, D = support.shape
        B, Nq, Q, D = query.shape

        query = query.view(B, Nq * Q, D)

        positive_prototypes = support.mean(dim=2)  # (B, Ns, K, D) -> (B, Ns, D)
        negative_prototypes = negative.mean(dim=2) # (B, Ns, K, D) -> (B, Ns, D)

        prototypes = torch.cat([positive_prototypes, negative_prototypes], dim=1)

        _prototypes = prototypes.unsqueeze(dim=1).expand(-1, Nq * Q, -1, -1)  # B x NqQ x Ns x D
        Ns = _prototypes.shape[2]
        query = query.unsqueeze(dim=2).expand(-1, -1, Ns, -1)  # B x NqQ x N x D

        # print('Relation: query', tuple(query.shape))
        # print('Relation: prototypes', tuple(prototypes.shape))

        abs = torch.abs(_prototypes - query)
        prod = _prototypes * query
        concatenate_rep = torch.cat([_prototypes, query, abs, prod], dim=3)

        logits = self.distance_fc(concatenate_rep).squeeze(dim=3)
        # print('RelationNetwork > logits: ', tuple(logits.shape))
        return logits, prototypes

    def forward(self, support, query, negative=None):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''
        # print('N/K/Q: ', N, K, Q)

        self.support = self.sentence_encoder(support)  # (B, Ns, K, D), where D is the hidden size
        self.query = self.sentence_encoder(query)  # (B, Nq, K, D)
        negative = self.sentence_encoder(negative)
        self.logits, self.prototypes = self.do_relation_with_negative(self.support, self.query, negative)

        _, self.pred = torch.max(self.logits.view(-1, self.logits.shape[-1]), 1)
