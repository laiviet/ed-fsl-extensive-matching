import torch
import framework


class Matching(framework.FewShotREModel):

    def __init__(self, args):
        framework.FewShotREModel.__init__(self, args)

    def loloss(self):
        """

        :return:
        """
        support = self.support[:, :, 2:, :].contiguous()  # B x Ns x K-2 x D
        query = self.support[:, :, 0:2, :].contiguous()  # B x 2Ns x D
        logits, _ = self.do_matching(support, query)
        B, N, C = logits.shape
        targets = torch.LongTensor([x for x in range(C)] * (B * N // C)).to(self.device)

        # print('| proto > loloss > logits', tuple(self.logits.shape))
        # print('| proto > loloss > targets', tuple(targets.shape))

        logits = logits.view(-1, logits.shape[-1])
        targets = targets.view(-1).to(self.device)

        return self.cost(logits, targets)

    def do_matching(self, support, query):
        """

        :param support:
        :param query:
        :return:
        """
        # print('| Matching: support', tuple(support.shape))
        # print('| Matching: query', tuple(query.shape))

        # B, Ns, K, D = support.shape
        B, Nq, Q, D = query.shape
        query = query.view(B, -1, D)  # (B, Nq * Q, D)
        # print('| Matching: query', tuple(query.shape))

        # Matching Network
        prototypes = torch.mean(support, 2)  # (B, Ns, K, D) -> (B, Ns, D)
        # print('| Matching: prototypes', tuple(prototypes.shape))

        _prototypes = torch.transpose(prototypes, dim0=1, dim1=2)  # (B, Ns, D) -> (B, D, Ns)

        logits = torch.bmm(query, _prototypes)  # B x NqQ x Ns

        return logits, prototypes

    def do_matching_with_negative(self, support, query, negative):
        # B, Ns, K, D = support.shape
        B, Nq, Q, D = query.shape
        # print('| Matching: support', tuple(support.shape))
        query = query.view(B, Nq * Q, D)  # (B, Nq * Q, D)
        # print('| Matching: query', tuple(query.shape))

        # Matching Network
        positive_proto = torch.mean(support, 2)  # (B, Ns, K, D) -> (B, Ns, D)
        negative_proto = torch.mean(negative, 2)  # (B, Ns, K, D) -> (B, Ns, D)
        # print('| Matching: proto', tuple(proto.shape))

        self.prototypes = torch.cat([positive_proto, negative_proto], dim=1)
        # print('| Matching: self.prototypes', tuple(self.prototypes.shape))
        # print('| Matching: query', tuple(query.shape))

        # query = query.unsqueeze(dim=2)
        # prototypes = self.prototypes.unsqueeze(dim=1)

        # print('| Matching: prototypes', tuple(prototypes.shape))
        # print('| Matching: query', tuple(query.shape))
        _prototypes = torch.transpose(self.prototypes, dim0=1, dim1=2)  # B x Ns x D -> B x D x Ns

        logits = torch.bmm(query, _prototypes)  # B x NqQ x Ns

        # logits = torch.nn.functional.cosine_similarity(query, prototypes, -1)  # B x NqQ x Ns
        return logits, self.prototypes

    def forward(self, support, query, negative=None):
        """

        :param support:
        :param query:
        :param negative:
        :return:
        """
        self.support = self.sentence_encoder(support)  # (B, Ns, K, D), where D is the hidden size
        self.query = self.sentence_encoder(query)  # (B, Nq, K, D)

        # print('| Matching: self.support ', tuple(self.support.shape))
        # print('| Matching: self.query ', tuple(self.query.shape))

        self.negative = self.sentence_encoder(negative)  # B x O x D
        # print('| Matching: self.negative ', tuple(self.negative.shape))

        self.logits, self.prototypes = self.do_matching_with_negative(self.support, self.query, self.negative)

        _, self.pred = torch.max(self.logits.view(-1, self.logits.shape[-1]), 1)
