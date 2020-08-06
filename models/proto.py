import sys

sys.path.append('..')
import torch
from torch import nn
import framework
from torch.nn import functional as F


class Proto(framework.FewShotREModel):

    def __init__(self, args):
        framework.FewShotREModel.__init__(self, args)

    def __dist__(self, x, y, dim):
        return (torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        # print('Proto > S: ', tuple(S.shape))
        # print('Proto > Q: ', tuple(Q.shape))
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def loloss(self):
        """

        :return:
        """
        B, Ns, K, D = self.support.shape
        support = self.support[:, :, 2:, :]  # B x Ns x K-2 x D
        query = self.support[:, :, 0:2, :].contiguous().view(B, Ns * 2, D)  # B x 2Ns x D
        logits, _ = self.do_proto(support, query)
        B, N, C = logits.shape
        targets = torch.LongTensor([x for x in range(C)] * (B * N // C)).to(self.device)

        # print('| proto > loloss > logits', tuple(self.logits.shape))
        # print('| proto > loloss > targets', tuple(targets.shape))

        logits = logits.view(-1, logits.shape[-1])
        targets = targets.view(-1).to(self.device)

        return self.cost(logits, targets)

    def do_proto(self, support, query):
        """

        :param support: B x N x K x D
        :param query:   B x NQ x D
        :return:
        """
        # Prototypical Networks
        prototypes = torch.mean(support, 2)  # Calculate prototype for each class
        # print('| proto > prototypes', tuple(prototypes.shape))

        logits = -self.__batch_dist__(prototypes, query)
        # print('| proto > logits', tuple(logits.shape))
        return logits, prototypes

    def do_proto_with_negative(self, support, query, negative):
        # Prototypical Networks
        positive_prototypes = torch.mean(support, 2)  # Calculate prototype for each class
        # print('| proto > positive_prototypes', tuple(positive_prototypes.shape))
        negative_prototypes = torch.mean(negative, 2)
        # print('| proto > negative_prototypes', tuple(negative_prototypes.shape))

        prototypes = torch.cat([positive_prototypes, negative_prototypes], dim=1)
        # print('| proto > prototypes', tuple(prototypes.shape))

        logits = -self.__batch_dist__(prototypes, query)
        return logits, prototypes

    def forward(self, support, query, negative):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''
        # print('N/K/Q: ', N, K, Q)

        self.support = self.sentence_encoder(support)  # (B, Ns, K, D), where D is the hidden size
        # print("Proto > support", tuple(self.support.shape))
        self.query = self.sentence_encoder(query)  # (B, Nq, K, D)
        # print("Proto > query", tuple(self.query.shape))

        B, Ns, K, D = self.support.shape
        B, Nq, Q, D = self.query.shape
        self.query = self.query.view(B, -1, D)  # B, Nq*K, D

        # print("Proto > query", tuple(self.query.shape))

        # print('Negative example')
        negative = self.sentence_encoder(negative)
        self.logits, self.prototypes = self.do_proto_with_negative(self.support, self.query, negative)

        _, self.pred = torch.max(self.logits.view(-1, self.logits.shape[-1]), 1)


class ProtoHATT(Proto):

    def __init__(self, args):
        framework.FewShotREModel.__init__(self, args)
        self.drop = nn.Dropout()
        K = args.shot
        # for instance-level attention
        self.fc = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # for feature-level attention
        self.conv1 = nn.Conv2d(1, 32, (K, 1), padding=(K // 2, 0))
        self.conv2 = nn.Conv2d(32, 64, (K, 1), padding=(K // 2, 0))
        self.conv_final = nn.Conv2d(64, 1, (K, 1), stride=(K, 1))
        self.self_conv_final = nn.Conv2d(64, 1, (K - 1, 1), stride=(K - 1, 1))
        print('Created ProtoHATT model')

    def __dist__(self, x, y, dim, score=None):
        if score is None:
            return (torch.pow(x - y, 2)).sum(dim)
        else:
            return (torch.pow(x - y, 2) * score).sum(dim)

    def __batch_dist__(self, S, Q, score=None):
        return self.__dist__(S, Q.unsqueeze(2), 3, score)

    def loloss(self):
        """

        :return:
        """
        B, Ns, K, D = self.support.shape
        support = self.support[:, :, 2:, :]  # B x Ns x K-2 x D
        query = self.support[:, :, 0:2, :].contiguous().view(B, Ns * 2, D)  # B x 2Ns x D
        logits, _ = self.do_proto(support, query)
        B, N, C = logits.shape
        targets = torch.LongTensor([x for x in range(C)] * (B * N // C)).to(self.device)

        # print('| proto > loloss > logits', tuple(self.logits.shape))
        # print('| proto > loloss > targets', tuple(targets.shape))

        logits = logits.view(-1, logits.shape[-1])
        targets = targets.view(-1).to(self.device)

        return self.cost(logits, targets)

    def do_proto_with_negative(self, support, query):
        """

        :param support:
        :param query:
        :return:
        """

        B, N, K, D = support.shape
        query = query.view(B,-1,D)
        NQ = query.shape[1]

        # instance-level attention


        support = support.unsqueeze(1).expand(-1, NQ, -1, -1, -1)  # (B, NQ, 2*N, K, D)
        # print("ProtoHATT > support", tuple(support.shape))
        # print("Proto > query", tuple(query.shape))

        support_for_att = self.fc(support)
        query_for_att = self.fc(query.unsqueeze(2).unsqueeze(3).expand(-1, -1, N, K, -1))

        ins_att_score = F.softmax(torch.tanh(support_for_att * query_for_att).sum(-1), dim=-1)  # (B, NQ, N, K)
        support_proto = (support * ins_att_score.unsqueeze(4).expand(-1, -1, -1, -1, D)).sum(
            3)  # (B, NQ, N, D)

        logits = -self.__batch_dist__(support_proto, query)

        # print('| ProtoHATT > support_proto: ', tuple(support_proto.shape))
        return logits, support_proto

    def forward(self, support, query, negative):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''
        self.support = self.sentence_encoder(support)  # (B, Ns, K, D), where D is the hidden size
        # print("ProtoHATT > support", tuple(self.support.shape))
        self.query = self.sentence_encoder(query)  # (B, Nq, K, D)
        # print("ProtoHATT > query", tuple(self.query.shape))
        negative = self.sentence_encoder(negative)
        # print("ProtoHATT > negative", tuple(negative.shape))

        support = torch.cat([self.support, negative],dim=1)
        B, Ns, K, D = support.shape


        self.logits, self.prototypes = self.do_proto_with_negative(support, self.query)
        _, self.pred = torch.max(self.logits.view(-1, Ns), 1)
