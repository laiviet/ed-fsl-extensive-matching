import framework
import torch
import torch.nn as nn
import torch.nn.functional as F


class CompareNetwork2(nn.Module):

    def __init__(self, n_features):
        super(CompareNetwork2, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * n_features, n_features),
            nn.Sigmoid(),
            nn.Linear(n_features, 1)
        )

    def forward(self, S, Q):
        B, NQ, N, D = S.shape
        # print(S.shape)
        # print(Q.shape)
        Q = Q.unsqueeze(2).expand(-1, -1, N, -1)
        features = torch.cat([torch.abs(S - Q), S * Q], dim=3)
        score = self.fc(features).squeeze(3)
        return score


class CompareNetwork4(nn.Module):

    def __init__(self, n_features):
        super(CompareNetwork4, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4 * n_features, n_features),
            nn.Sigmoid(),
            nn.Linear(n_features, 1)
        )

    def forward(self, S, Q):
        B, NQ, N, D = S.shape
        # print(S.shape)
        # print(Q.shape)
        Q = Q.unsqueeze(2).expand(-1, -1, N, -1)
        features = torch.cat([S, Q, torch.abs(S - Q), S * Q], dim=3)
        score = self.fc(features).squeeze(3)
        return score


class Relation2(framework.FewShotREModel):

    def __init__(self, sentence_encoder, shots, hidden_size=230):
        super(Relation2, self).__init__(sentence_encoder)

        self.hidden_size = hidden_size
        self.drop = nn.Dropout()

        # for instance-level attention
        self.fc = nn.Linear(hidden_size, hidden_size, bias=True)
        # for feature-level attention
        self.conv1 = nn.Conv2d(1, 32, (shots, 1), padding=(shots // 2, 0))
        self.conv2 = nn.Conv2d(32, 64, (shots, 1), padding=(shots // 2, 0))
        self.conv_final = nn.Conv2d(64, 1, (shots, 1), stride=(shots, 1))

        self.compare = CompareNetwork2(hidden_size)

    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''
        support = self.sentence_encoder(support)  # (B * N * K, D), where D is the hidden size
        query = self.sentence_encoder(query)  # (B * N * Q, D)
        support = support.view(-1, N, K, self.hidden_size)  # (B, N, K, D)
        query = query.view(-1, N * Q, self.hidden_size)  # (B, N * Q, D)

        B, N, K, _ = support.shape  # Batch size
        NQ = query.size(1)  # Num of instances for each batch in the query set

        prototypes = support.mean(2).unsqueeze(1).expand(-1, NQ, -1, -1)

        # Compare network
        logits = self.compare(prototypes, query)
        _, pred = torch.max(logits.view(-1, N), 1)

        return logits, pred


class Relation4(framework.FewShotREModel):

    def __init__(self, sentence_encoder, shots, hidden_size=230):
        framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.drop = nn.Dropout()

        # for instance-level attention
        self.fc = nn.Linear(hidden_size, hidden_size, bias=True)
        # for feature-level attention
        self.conv1 = nn.Conv2d(1, 32, (shots, 1), padding=(shots // 2, 0))
        self.conv2 = nn.Conv2d(32, 64, (shots, 1), padding=(shots // 2, 0))
        self.conv_final = nn.Conv2d(64, 1, (shots, 1), stride=(shots, 1))

        self.compare = CompareNetwork4(hidden_size)

    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''
        support = self.sentence_encoder(support)  # (B * N * K, D), where D is the hidden size
        query = self.sentence_encoder(query)  # (B * N * Q, D)
        support = support.view(-1, N, K, self.hidden_size)  # (B, N, K, D)
        query = query.view(-1, N * Q, self.hidden_size)  # (B, N * Q, D)

        B, N, K, _ = support.shape  # Batch size
        NQ = query.size(1)  # Num of instances for each batch in the query set

        # feature-level attention
        # fea_att_score = support.view(B * N, 1, K, self.hidden_size)  # (B * N, 1, K, D)
        # fea_att_score = F.relu(self.conv1(fea_att_score))  # (B * N, 32, K, D)
        # fea_att_score = F.relu(self.conv2(fea_att_score))  # (B * N, 64, K, D)
        # fea_att_score = self.drop(fea_att_score)
        # fea_att_score = self.conv_final(fea_att_score)  # (B * N, 1, 1, D)
        # fea_att_score = F.relu(fea_att_score)
        # fea_att_score = fea_att_score.view(B, N, self.hidden_size).unsqueeze(1)  # (B, 1, N, D)
        # print('| support', tuple(support.shape))
        # print('| query', tuple(query.shape))

        # instance-level attention
        # support_for_att = self.fc(support).view(B, N * K, -1).transpose(dim0=1, dim1=2)
        # # print('| support_for_att', tuple(support_for_att.shape))
        # query_for_att = self.fc(query)
        # # print('| query_for_att', tuple(query_for_att.shape))
        # score = torch.bmm(query_for_att, support_for_att).view(B, NQ, N, K)
        # # print('| score', tuple(score.shape))
        #
        # ins_att_score = F.softmax(score, dim=3)  # (B, NQ, N, K)
        # # print('| ins_att_score', tuple(ins_att_score.shape))
        #
        # support = support.unsqueeze(1).expand(-1, NQ, -1, -1, -1)  # B, NQ, N, K, D
        # support = support * ins_att_score.unsqueeze(4)  # (B, NQ, N, K, D)
        # # print('| support', tuple(support.shape))
        # protypes = support.mean(3)  # (B, NQ, N, D)
        # print('| protypes', tuple(protypes.shape))

        prototypes = support.mean(2).unsqueeze(1).expand(-1, NQ, -1, -1)

        # Compare network
        logits = self.compare(prototypes, query)
        _, pred = torch.max(logits.view(-1, N), 1)

        return logits, pred


class RelationHatt2(framework.FewShotREModel):

    def __init__(self, sentence_encoder, shots, hidden_size=230):
        framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.drop = nn.Dropout()

        # for instance-level attention
        self.fc = nn.Linear(hidden_size, hidden_size, bias=True)
        # for feature-level attention
        self.conv1 = nn.Conv2d(1, 32, (shots, 1), padding=(shots // 2, 0))
        self.conv2 = nn.Conv2d(32, 64, (shots, 1), padding=(shots // 2, 0))
        self.conv_final = nn.Conv2d(64, 1, (shots, 1), stride=(shots, 1))

        self.compare = CompareNetwork2(hidden_size)

    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''
        support = self.sentence_encoder(support)  # (B * N * K, D), where D is the hidden size
        query = self.sentence_encoder(query)  # (B * N * Q, D)
        support = support.view(-1, N, K, self.hidden_size)  # (B, N, K, D)
        query = query.view(-1, N * Q, self.hidden_size)  # (B, N * Q, D)

        B, N, K, _ = support.shape  # Batch size
        NQ = query.size(1)  # Num of instances for each batch in the query set

        # feature-level attention
        fea_att_score = support.view(B * N, 1, K, self.hidden_size)  # (B * N, 1, K, D)
        fea_att_score = F.relu(self.conv1(fea_att_score))  # (B * N, 32, K, D)
        fea_att_score = F.relu(self.conv2(fea_att_score))  # (B * N, 64, K, D)
        fea_att_score = self.drop(fea_att_score)
        fea_att_score = self.conv_final(fea_att_score)  # (B * N, 1, 1, D)
        fea_att_score = F.relu(fea_att_score)
        fea_att_score = fea_att_score.view(B, N, self.hidden_size).unsqueeze(1)  # (B, 1, N, D)

        # instance-level attention
        support = support.unsqueeze(1).expand(-1, NQ, -1, -1, -1)  # (B, NQ, N, K, D)
        support_for_att = self.fc(support)
        query_for_att = self.fc(query.unsqueeze(2).unsqueeze(3).expand(-1, -1, N, K, -1))
        ins_att_score = F.softmax(torch.tanh(support_for_att * query_for_att).sum(-1), dim=-1)  # (B, NQ, N, K)
        support_proto = (support * ins_att_score.unsqueeze(4).expand(-1, -1, -1, -1, self.hidden_size)).sum(
            3)  # (B, NQ, N, D)

        # Compare network
        logits = self.compare(support_proto, query)
        _, pred = torch.max(logits.view(-1, N), 1)

        return logits, pred


class RelationHatt4(framework.FewShotREModel):

    def __init__(self, sentence_encoder, shots, hidden_size=230):
        framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.drop = nn.Dropout()

        # for instance-level attention
        self.fc = nn.Linear(hidden_size, hidden_size, bias=True)
        # for feature-level attention
        self.conv1 = nn.Conv2d(1, 32, (shots, 1), padding=(shots // 2, 0))
        self.conv2 = nn.Conv2d(32, 64, (shots, 1), padding=(shots // 2, 0))
        self.conv_final = nn.Conv2d(64, 1, (shots, 1), stride=(shots, 1))

        self.compare = CompareNetwork4(hidden_size)

    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''
        support = self.sentence_encoder(support)  # (B * N * K, D), where D is the hidden size
        query = self.sentence_encoder(query)  # (B * N * Q, D)
        support = support.view(-1, N, K, self.hidden_size)  # (B, N, K, D)
        query = query.view(-1, N * Q, self.hidden_size)  # (B, N * Q, D)

        B, N, K, _ = support.shape  # Batch size
        NQ = query.size(1)  # Num of instances for each batch in the query set

        # feature-level attention
        fea_att_score = support.view(B * N, 1, K, self.hidden_size)  # (B * N, 1, K, D)
        fea_att_score = F.relu(self.conv1(fea_att_score))  # (B * N, 32, K, D)
        fea_att_score = F.relu(self.conv2(fea_att_score))  # (B * N, 64, K, D)
        fea_att_score = self.drop(fea_att_score)
        fea_att_score = self.conv_final(fea_att_score)  # (B * N, 1, 1, D)
        fea_att_score = F.relu(fea_att_score)
        fea_att_score = fea_att_score.view(B, N, self.hidden_size).unsqueeze(1)  # (B, 1, N, D)

        # instance-level attention
        support = support.unsqueeze(1).expand(-1, NQ, -1, -1, -1)  # (B, NQ, N, K, D)
        support_for_att = self.fc(support)
        query_for_att = self.fc(query.unsqueeze(2).unsqueeze(3).expand(-1, -1, N, K, -1))
        ins_att_score = F.softmax(torch.tanh(support_for_att * query_for_att).sum(-1), dim=-1)  # (B, NQ, N, K)
        support_proto = (support * ins_att_score.unsqueeze(4).expand(-1, -1, -1, -1, self.hidden_size)).sum(
            3)  # (B, NQ, N, D)

        # Compare network
        logits = self.compare(support_proto, query)
        _, pred = torch.max(logits.view(-1, N), 1)

        return logits, pred
