import torch
import torch.nn as nn
import torch.nn.functional as F
from models.embedding import *


class CNNSentenceEncoder(nn.Module):

    def __init__(self, args):
        super(CNNSentenceEncoder, self).__init__()
        self.device = args.device
        self.hidden_size = args.hidden_size
        self.max_length = args.max_length

        self.embedder = Embedding(args.vectors,
                                  tune_embedding=args.tune_embedding,
                                  device=args.device)

        self.embedding_dim = self.embedder.size
        self.conv = nn.Conv1d(self.embedding_dim, self.hidden_size, 3, padding=1)
        self.pool = nn.MaxPool1d(args.max_length)
        self.relu = nn.ReLU()
        self.init()

    def init(self):
        torch.nn.init.xavier_uniform(self.conv.weight)

    def forward(self, inputs):
        embedding = self.embedder(inputs)
        shape = list(tuple(embedding.shape))
        L, D = shape[-2], shape[-1]
        # print('| CNNSentenceEncoder > embedding', tuple(embedding.shape))
        x = self.conv(embedding.view(-1, L, D).transpose(1, 2))
        x = self.relu(x)
        x = self.pool(x)
        # print('| CNNEncoder > x2', tuple(x.shape))
        ori = tuple(shape[:-2] + [-1])
        x = x.view(ori)
        # print('| CNNEncoder > x3', tuple(x.shape))
        return x


class GRUEncoder(nn.Module):

    def __init__(self, args):
        super(GRUEncoder, self).__init__()

        self.device = args.device
        self.hidden_size = args.hidden_size
        self.window = args.window
        self.max_length = args.max_length

        self.embedder = Embedding(args.vectors,
                                  tune_embedding=args.tune_embedding,
                                  device=args.device)
        self.gru = nn.GRU(input_size=self.embedder.size,
                          hidden_size=args.hidden_size,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)
        self.linear = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.non_linear = nn.Tanh()

    def initHidden(self, batch_size):
        h0 = torch.zeros(2 * 1, batch_size, self.hidden_size).to(self.device)
        return h0

    def introduceLocalEmbedding(self, frep, embed, anchors):
        assert type(frep) == list
        window = self.window
        assert window >= 0

        if window == 0:
            frep = frep[0] if len(frep) == 1 else torch.cat(frep, 1)
            return frep
        if window > 0:
            zeros = torch.zeros(embed.size(0), window, embed.size(2)).float().to(self.device)
            padded = torch.cat([zeros, embed, zeros], 1)

            ids = []
            for i in range(2 * window + 1):
                ids += [(anchors + i).long().view(-1, 1)]
            ids = torch.cat(ids, 1)
            ids = ids.unsqueeze(2).expand(ids.size(0), ids.size(1), embed.size(2))

            res = padded.gather(1, ids)
            res = res.view(res.size(0), -1)

            frep += [res]
            frep = torch.cat(frep, 1)

        return frep

    def select_anchor(self, emb, anchor_index):
        """

        :param emb: B x L x D
        :param anchor_index: B
        :return:
        """
        B, L, D = emb.shape
        u = torch.tensor([x for x in range(L)]).unsqueeze(0)
        v = anchor_index.view(B, 1)
        mask = (u == v).unsqueeze(dim=2).to(self.device)
        x = torch.masked_select(emb, mask).view(-1, D)
        return x

    def forward(self, inputs):
        embedding = self.embedder(inputs)
        shape = list(embedding.shape)

        embedding = embedding.view(tuple([-1] + shape[-2:]))
        anchors = inputs['anchor_index'].view(-1)

        # print('| GRUEncoder: anchors > ', tuple(anchors.shape))
        # print('| GRUEncoder: embedding > ', tuple(embedding.shape))
        b = 1
        for x in shape[:-2]:
            b *= x
        h0 = self.initHidden(b)
        hidden_states, _ = self.gru(embedding, h0)

        # print('| GRUEncoder: hidden_states > ', tuple(hidden_states.shape))

        rnnRep = self.select_anchor(hidden_states, anchors)
        # print('| GRUEncoder: rnnRep > ', tuple(rnnRep.shape))

        x = rnnRep.view(tuple(shape[:-2] + [2 * self.hidden_size]))
        # print('| GRUEncoder: x > ', tuple(x.shape))
        x = self.linear(x)
        x = self.non_linear(x)

        return x


class LSTMEncoder(nn.Module):

    def __init__(self, args):
        super(LSTMEncoder, self).__init__()

        self.device = args.device
        self.hidden_size = args.hidden_size
        self.window = args.window
        self.max_length = args.max_length

        self.embedder = Embedding(args.vectors,
                                  tune_embedding=args.tune_embedding,
                                  device=args.device)
        self.gru = nn.LSTM(input_size=self.embedder.size,
                           hidden_size=args.hidden_size,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=True)
        self.linear = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.non_linear = nn.Tanh()

    def initHidden(self, batch_size):
        h0 = torch.zeros(2 * 1, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(2 * 1, batch_size, self.hidden_size).to(self.device)
        return h0, c0

    def introduceLocalEmbedding(self, frep, embed, anchors):
        assert type(frep) == list
        window = self.window
        assert window >= 0

        if window == 0:
            frep = frep[0] if len(frep) == 1 else torch.cat(frep, 1)
            return frep
        if window > 0:
            zeros = torch.zeros(embed.size(0), window, embed.size(2)).float().to(self.device)
            padded = torch.cat([zeros, embed, zeros], 1)

            ids = []
            for i in range(2 * window + 1):
                ids += [(anchors + i).long().view(-1, 1)]
            ids = torch.cat(ids, 1)
            ids = ids.unsqueeze(2).expand(ids.size(0), ids.size(1), embed.size(2))

            res = padded.gather(1, ids)
            res = res.view(res.size(0), -1)

            frep += [res]
            frep = torch.cat(frep, 1)

        return frep

    def select_anchor(self, emb, anchor_index):
        """

        :param emb: B x L x D
        :param anchor_index: B
        :return:
        """
        B, L, D = emb.shape
        u = torch.tensor([x for x in range(L)]).unsqueeze(0)
        v = anchor_index.view(B, 1)
        mask = (u == v).unsqueeze(dim=2).to(self.device)
        x = torch.masked_select(emb, mask).view(-1, D)
        return x

    def forward(self, inputs):
        embedding = self.embedder(inputs)
        shape = list(embedding.shape)

        embedding = embedding.view(tuple([-1] + shape[-2:]))
        anchors = inputs['anchor_index'].view(-1)

        # print('| GRUEncoder: anchors > ', tuple(anchors.shape))
        # print('| GRUEncoder: embedding > ', tuple(embedding.shape))
        b = 1
        for x in shape[:-2]:
            b *= x
        h0 = self.initHidden(b)
        hidden_states, _ = self.gru(embedding, h0)

        # print('| GRUEncoder: hidden_states > ', tuple(hidden_states.shape))

        rnnRep = self.select_anchor(hidden_states, anchors)
        # print('| GRUEncoder: rnnRep > ', tuple(rnnRep.shape))

        x = rnnRep.view(tuple(shape[:-2] + [2 * self.hidden_size]))
        # print('| GRUEncoder: x > ', tuple(x.shape))
        x = self.linear(x)
        x = self.non_linear(x)

        return x