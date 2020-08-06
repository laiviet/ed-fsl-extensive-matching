import torch.nn as nn
import torch
import math
from copy import deepcopy
import torch.nn.functional as F
from models.embedding import Embedding
import numpy as np

import framework


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # if mask is not None:
    #
    #     scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def select_anchor(emb, anchor_index, device):
    """

    :param emb: B x L x D
    :param anchor_index: B
    :return:
    """
    B, L, D = emb.shape
    # print('| select_anchor > device', device)
    u = torch.tensor([x for x in range(L)]).unsqueeze(0).to(device)
    v = anchor_index.view(B, 1)
    mask = (u == v).unsqueeze(dim=2)
    x = torch.masked_select(emb, mask).view(-1, D)
    return x


class InteractLayer(nn.Module):

    def __init__(self, d_model, device='cpu'):
        super(InteractLayer, self).__init__()
        self.device = device
        self.to_s = nn.Linear(d_model, d_model)
        self.to_q = nn.Linear(d_model, d_model)

    def forward(self, s, q, s_anchor, q_anchor, s_mask, q_mask, B, N, K, Q):
        """

        :param s:
        :param q:
        :param s_anchor:
        :param q_anchor:
        :param s_mask:
        :param q_mask:
        :return:
        """

        # print('InteractLayer > s ', tuple(s.shape))
        # print('InteractLayer > q ', tuple(q.shape))
        # print('InteractLayer > s_anchor ', tuple(s_anchor.shape))
        # print('InteractLayer > q_anchor ', tuple(q_anchor.shape))
        # print('InteractLayer > s_mask ', tuple(s_mask.shape))
        # print('InteractLayer > q_mask ', tuple(q_mask.shape))

        _, L, D = s.shape

        s_anchor_emb = select_anchor(s, s_anchor, self.device)
        q_anchor_emb = select_anchor(q, q_anchor, self.device)
        s_anchor_emb = self.to_q(s_anchor_emb).view(B, N * K, D).unsqueeze(dim=2).unsqueeze(dim=3)
        q_anchor_emb = self.to_s(q_anchor_emb).view(B, N * Q, D).unsqueeze(dim=1).unsqueeze(dim=3)

        s = s.view(B, N * K, L, D).unsqueeze(dim=2).expand(-1, -1, N * Q, -1, -1)  # B x NK x NQ x L x D
        q = q.view(B, N * Q, L, D).unsqueeze(dim=1).expand(-1, N * K, -1, -1, -1)  # B x NK x NQ x L x D

        print('InteractLayer > s ', tuple(s.shape))
        print('InteractLayer > q ', tuple(q.shape))

        # print('InteractLayer > s_anchor_emb ', tuple(s_anchor_emb.shape))
        # print('InteractLayer > q_anchor_emb ', tuple(q_anchor_emb.shape))

        s = s * q_anchor_emb
        q = q * s_anchor_emb

        return s, q


class CNNInteractLayer(nn.Module):

    def __init__(self, d_model, device='cpu'):
        super(CNNInteractLayer, self).__init__()
        self.device = device

        self.cnns = nn.ModuleList([nn.Conv1d(d_model * 2, 150, 2, padding=1),
                                   nn.Conv1d(d_model * 2, 150, 3, padding=1),
                                   nn.Conv1d(d_model * 2, 150, 4, padding=2),
                                   nn.Conv1d(d_model * 2, 150, 5, padding=2)])
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(31)

    def forward(self, s, q, B, N, K, Q, L=31):
        """

        :param s:
        :param q:
        :param s_anchor:
        :param q_anchor:
        :param s_mask:
        :param q_mask:
        :return:
        """

        D = s.shape[-1]

        # print('InteractLayer > s ', tuple(s.shape))
        # print('InteractLayer > q ', tuple(q.shape))

        s = s.view(B, N * K, L, -1)
        q = q.view(B, N * Q, L, -1)

        s = s.unsqueeze(dim=2).expand(-1, -1, N * Q, -1, -1)
        q = q.unsqueeze(dim=1).expand(-1, N * K, -1, -1, -1)

        # print('InteractLayer > s ', tuple(s.shape))
        # print('InteractLayer > q ', tuple(q.shape))

        sq = torch.cat((s, q), 4).view(-1, L, D * 2).transpose(dim0=1, dim1=2).contiguous()

        # print('InteractLayer > sq ', tuple(sq.shape))

        x = [cnn(sq) for cnn in self.cnns]
        s = [x[0][:, 0:75, :-1], x[1][:, 0:75, :], x[2][:, 0:75, :-1], x[3][:, 0:75, :]]
        q = [x[0][:, 75:, :-1], x[1][:, 75:, :], x[2][:, 75:, :-1], x[3][:, 75:, :]]

        s = self.relu(torch.cat(s, dim=1))
        q = self.relu(torch.cat(q, dim=1))
        # print('InteractLayer > s ', tuple(s.shape))

        s = self.pooling(s).squeeze(dim=2)
        q = self.pooling(q).squeeze(dim=2)
        return s, q


class TransInteractProto(framework.FewShotREModel):

    def __init__(self, word_vec_mat, max_length=31, pos_embedding_dim=50, N=1,
                 d_model=512, d_ff=2048, heads=10, dropout=0.1, device='cpu'):
        super(TransInteractProto, self).__init__()

        self.n_layers = N
        self.d_model = d_model
        self.device = device
        self.embedding = Embedding(word_vec_mat, max_length,
                                   pos_embedding_dim=pos_embedding_dim,
                                   device=device)

        # Modules
        att = MultiHeadedAttention(heads, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        sublayer = SublayerConnection(d_model, dropout)

        # For Encoder layer
        self.self_attn = clones(att, N)
        self.feed_forward = clones(ff, N)
        self.sublayer_1 = clones(sublayer, N)
        self.sublayer_2 = clones(sublayer, N)
        self.norm = LayerNorm(d_model)

        # for post interact ver1
        # self.post_self_attn = clones(att, N)
        # self.post_feed_forward = clones(ff, N)
        # self.post_sublayer_1 = clones(sublayer, N)
        # self.post_sublayer_2 = clones(sublayer, N)
        # self.post_norm = LayerNorm(d_model)
        #

        # for for ver 3
        self.post_self_attn = clones(att, N)
        self.post_feed_forward = clones(ff, N)
        self.post_sublayer_1 = clones(sublayer, N)
        self.post_sublayer_2 = clones(sublayer, N)
        self.post_norm = LayerNorm(d_model)



        # for Interact layer
        # self.interact_layer = InteractLayer(d_model, device=device)
        # self.interact_layer = CNNInteractLayer(d_model)

        # For proto
        # self.fc = nn.Linear(d_model, 230)
        self.fc_ver3 = nn.Linear(d_model * 2, 1)

    def forward(self, support, query, N, K, Q):
        return self.ver3(support, query, N, K, Q)

    def ver1(self, support, query, N, K, Q, L=31):
        """
        Out of Memory

        :param support: dict
        :param query: dict
        :param N: way
        :param K: shot
        :param Q: query
        :return:

        """

        B = support['length'].shape[0]
        D = self.d_model

        s_mask = support['mask'].view(-1, L).to(self.device)
        q_mask = query['mask'].view(-1, L).to(self.device)

        s_anchor = support['anchor_index'].view(-1).to(self.device)
        q_anchor = query['anchor_index'].view(-1).to(self.device)

        s, q = self.embedding(support), self.embedding(query)  # B*N*K x L x D,  B*N*Q x L x D

        # print('| TransInteractProto > s [1]', tuple(s.shape))
        # print('| TransInteractProto > q [1]', tuple(q.shape))

        # Intra sentence encoder

        for i in range(self.n_layers):
            q = self.sublayer_1[i](q, lambda x: self.self_attn[i](q, q, q, q_mask))
            q = self.sublayer_2[i](q, self.feed_forward[i])
        q = self.norm(q)

        for i in range(self.n_layers):
            s = self.sublayer_1[i](s, lambda x: self.self_attn[i](s, s, s, s_mask))
            # print('| TransInteractProto > s', tuple(s.shape))
            s = self.sublayer_2[i](s, self.feed_forward[i])
            # print('| TransInteractProto > s', tuple(s.shape))
        s = self.norm(s)

        # Inter connection
        # s = s.view(B, N * K, L, D)
        # q = q.view(B, N * Q, L, D)

        s, q = self.interact_layer(s, q, s_anchor, q_anchor, s_mask, q_mask, B, N, K, Q)
        # print('| TransInteractProto > s [2]', tuple(s.shape))
        # print('| TransInteractProto > q [2]', tuple(q.shape))

        s = s.view(-1, L, D)
        q = q.view(-1, L, D)

        for i in range(self.n_layers):
            s = self.post_sublayer_1[i](s, lambda x: self.post_self_attn[i](s, s, s, s_mask))
            # print('| TransInteractProto > s', tuple(s.shape))
            s = self.post_sublayer_2[i](s, self.post_feed_forward[i])
            # print('| TransInteractProto > s', tuple(s.shape))
        s = self.post_norm(s)

        for i in range(self.n_layers):
            q = self.post_sublayer_1[i](q, lambda x: self.post_self_attn[i](q, q, q, q_mask))
            q = self.post_sublayer_2[i](q, self.post_feed_forward[i])
        q = self.post_norm(q)

        s_anchor = s_anchor.view(B, N * K).unsqueeze(dim=2).expand(-1, -1, N * Q).contiguous().view(-1)
        q_anchor = q_anchor.view(B, N * Q).unsqueeze(dim=1).expand(-1, N * K, -1).contiguous().view(-1)

        s = select_anchor(s, s_anchor, self.device)
        q = select_anchor(q, q_anchor, self.device)

        print('| TransInteractProto > s [3]', tuple(s.shape))
        print('| TransInteractProto > q [3]', tuple(q.shape))

        logits = torch.pow(s - q, 2).sum(dim=1)
        logits = logits.view(B, N, K, N * Q).mean(dim=2)
        logits = torch.transpose(logits, dim0=1, dim1=2).contiguous()
        _, preds = logits.max(dim=2)

        return logits, preds

    def ver2(self, support, query, N, K, Q, L=31):
        """
        Similar to the Transproto Model

        :param support: dict
        :param query: dict
        :param N: way
        :param K: shot
        :param Q: query
        :return:

        """

        B = support['length'].shape[0]
        D = self.d_model

        s_mask = support['mask'].view(-1, L).to(self.device)
        q_mask = query['mask'].view(-1, L).to(self.device)

        s_anchor = support['anchor_index'].view(-1).to(self.device)
        q_anchor = query['anchor_index'].view(-1).to(self.device)

        s, q = self.embedding(support), self.embedding(query)  # B*N*K x L x D,  B*N*Q x L x D

        print('| TransInteractProto > s [1]', tuple(s.shape))
        print('| TransInteractProto > q [1]', tuple(q.shape))

        # Intra sentence encoder

        for i in range(self.n_layers):
            s = self.sublayer_1[i](s, lambda x: self.self_attn[i](s, s, s, s_mask))
            # print('| TransInteractProto > s', tuple(s.shape))
            s = self.sublayer_2[i](s, self.feed_forward[i])
            # print('| TransInteractProto > s', tuple(s.shape))
        s = self.norm(s)

        for i in range(self.n_layers):
            q = self.sublayer_1[i](q, lambda x: self.self_attn[i](q, q, q, q_mask))
            q = self.sublayer_2[i](q, self.feed_forward[i])
        q = self.norm(q)

        # Inter connection
        # s = s.view(B, N * K, L, D)
        # q = q.view(B, N * Q, L, D)

        # s, q = self.interact_layer(s, q, B, N, K, Q)
        # print('| TransInteractProto > s [2]', tuple(s.shape))
        # print('| TransInteractProto > q [2]', tuple(q.shape))
        #
        # s_anchor = s_anchor.view(B, N * K).unsqueeze(dim=2).expand(-1, -1, N * Q).contiguous().view(-1)
        # q_anchor = q_anchor.view(B, N * Q).unsqueeze(dim=1).expand(-1, N * K, -1).contiguous().view(-1)
        #
        s = select_anchor(s, s_anchor, self.device)
        q = select_anchor(q, q_anchor, self.device)

        # print('| TransInteractProto > s [2]', tuple(s.shape))
        # print('| TransInteractProto > q [2]', tuple(q.shape))

        s = s.view(B, N, K, -1).mean(dim=2).unsqueeze(dim=1)  # B x 1 x N x D
        q = q.view(B, N * Q, -1).unsqueeze(dim=2)  # B x NQ x 1 x D

        # print('| TransInteractProto > s [3]', tuple(s.shape))
        # print('| TransInteractProto > q [3]', tuple(q.shape))

        logits = -torch.pow(s - q, 2).sum(dim=3)  # B x NQ x N
        _, preds = logits.max(dim=2)
        return logits, preds

    def ver3(self, support, query, N, K, Q, L=31):
        """
        Concat two sentence into one

        :param support: dict
        :param query: dict
        :param N: way
        :param K: shot
        :param Q: query
        :return:

        """
        # print('| TransInteractProto > support["indices"]', tuple(support["indices"].shape))
        # print('| TransInteractProto > support["mask"]', tuple(support["mask"].shape))
        # print('| TransInteractProto > support["anchor_index"]', tuple(support["anchor_index"].shape))
        # print('| TransInteractProto > query["indices"]', tuple(query["indices"].shape))
        # print('| TransInteractProto > query["mask"]', tuple(query["mask"].shape))
        # print('| TransInteractProto > query["anchor_index"]', tuple(query["anchor_index"].shape))

        B = support['length'].shape[0]
        D = self.d_model

        s_mask = support['mask'].to(self.device)
        q_mask = query['mask'].to(self.device)

        s_anchor = support['anchor_index'].view(-1).to(self.device)
        q_anchor = query['anchor_index'].view(-1).to(self.device)

        s = self.embedding(support)  # B*N*K x L x D
        q = self.embedding(query)  # ,  B*N*Q x L x D

        print('| TransInteractProto > s [1]', tuple(s.shape))
        print('| TransInteractProto > q [1]', tuple(q.shape))

        # Support sentence encoder
        for i in range(self.n_layers):
            s = self.sublayer_1[i](s, lambda x: self.self_attn[i](s, s, s, s_mask))
            # print('| TransInteractProto > s', tuple(s.shape))
            s = self.sublayer_2[i](s, self.feed_forward[i])
            # print('| TransInteractProto > s', tuple(s.shape))
        s = self.norm(s)

        s_anchor_emb = select_anchor(s, s_anchor, self.device)
        s_anchor_emb = s_anchor_emb.view(B, N, K, D).mean(dim=2).unsqueeze(dim=1).unsqueeze(dim=3) # B x 1 x N x 1 x D
        print('| TransInteractProto > s_anchor_emb', tuple(s_anchor_emb.shape))
        q = q.view(B, N*Q, L, D).unsqueeze(dim=2) # B x NQ x 1 x L x D
        print('| TransInteractProto > q', tuple(q.shape))

        q = q * s_anchor_emb
        q=  q.view(-1, L, D)

        # Query encoder
        for i in range(self.n_layers):
            q = self.post_sublayer_1[i](q, lambda x: self.post_self_attn[i](q, q, q, q_mask))
            q = self.post_sublayer_2[i](q, self.post_feed_forward[i])
        q = self.post_norm(q)

        # Inter connection
        # s = s.view(B, N * K, L, D)
        # q = q.view(B, N * Q, L, D)

        # s, q = self.interact_layer(s, q, B, N, K, Q)
        # print('| TransInteractProto > s [2]', tuple(s.shape))
        # print('| TransInteractProto > q [2]', tuple(q.shape))
        #
        # s_anchor = s_anchor.view(B, N * K).unsqueeze(dim=2).expand(-1, -1, N * Q).contiguous().view(-1)
        q_anchor = q_anchor.view(B, N * Q).unsqueeze(dim=1).expand(-1, N * K, -1).contiguous().view(-1)
        #
        s = s_anchor_emb
        q = select_anchor(q, q_anchor, self.device)

        print('| TransInteractProto > s [2]', tuple(s.shape))
        print('| TransInteractProto > q [2]', tuple(q.shape))

        s = s.view(B, N, K, -1).mean(dim=2).unsqueeze(dim=1).expand(-1, N * Q, -1, -1)  # B x NQ x N x D
        q = q.view(B, N * Q, N*K, -1)  # B x NQ x N x D

        print('| TransInteractProto > s [3]', tuple(s.shape))
        print('| TransInteractProto > q [3]', tuple(q.shape))

        logits = -torch.pow(s - q, 2).sum(dim=3)  # B x NQ x N
        _, preds = logits.max(dim=2)
        return logits, preds
