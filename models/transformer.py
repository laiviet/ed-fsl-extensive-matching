import torch.nn as nn
import torch
import math
from copy import deepcopy
import torch.nn.functional as F
from models.embedding import Embedding
import numpy as np
import types
import framework


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


def filter_weight(weights, prefix):
    l = len(prefix) + 1
    selected = {k[l:]: v for k, v in weights.items() if k.startswith(prefix)}
    return selected


def merge(weight_dicts):
    all = {}
    for d in weight_dicts:
        for k, v in d.items():
            all[k] = v
    return all


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

    def functional_forward(self, x, mask, weights):
        for i, layer in enumerate(self.layers):
            p = 'layers.' + str(i)  # prefix
            print('-> ', p)
            layer_weights = filter_weight(weights, p)
            x = layer.functional_forward(x, mask, layer_weights)
        norm_weights = filter_weight(weights, 'norm')
        print('-> Encoder.norm')
        return self.norm.functional_forward(x, norm_weights)


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

    def functional_forward(self, x, weights):
        # print('Layer norm:')
        # print(weights.keys())

        print('Use: norm.a_2 ')
        print('Use: norm.b_2 ')
        a2 = weights['a_2']
        b2 = weights['b_2']

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return a2 * (x - mean) / (std + self.eps) + b2


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

    def functional_forward(self, x, sublayer, weights):
        # print('Sublayer Connection:')
        # print(weights.keys())
        norm_weights = filter_weight(weights, 'norm')
        norm = self.norm.functional_forward(x, norm_weights)
        if isinstance(sublayer, types.FunctionType):
            before_dropout = sublayer(norm)
        else:
            before_dropout = sublayer.functional_forward(norm, weights)

        return x + self.dropout(before_dropout)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.ff = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        att = self.self_attn(x, x, x, mask)
        x = self.sublayer[0](x, lambda x: att)
        return self.sublayer[1](x, self.ff)

    def functional_forward(self, x, mask, weights):
        self_att_weight = filter_weight(weights, 'self_attn')
        att = self.self_attn.functional_forward(x, x, x, self_att_weight, mask)
        sublayer_0_weights = filter_weight(weights, 'sublayer.0')
        x = self.sublayer[0].functional_forward(x, lambda x: att, sublayer_0_weights)
        sublayer_1_weights = filter_weight(weights, 'sublayer.1')
        ff_weights = filter_weight(weights, 'ff')

        _w = merge([sublayer_1_weights, ff_weights])

        x = self.sublayer[1].functional_forward(x, self.ff, _w)
        return x


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
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def functional_forward(self, query, key, value, weights, mask=None):
        for k in weights.keys():
            print('Use: ', k, '     multihead attention')
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = F.linear(query, weights['linears.0.weight'], weights['linears.0.bias'])
        key = F.linear(key, weights['linears.1.weight'], weights['linears.1.bias'])
        value = F.linear(value, weights['linears.2.weight'], weights['linears.2.bias'])

        query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = key.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = value.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return F.linear(x, weights['linears.3.weight'], weights['linears.3.bias'])


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

    def functional_forward(self, x, weights):
        print('Use w_1.weight')
        print('Use w_1.bias')
        print('Use w_2.weight')
        print('Use w_2.bias')
        x = F.linear(x, weights['w_1.weight'], weights['w_1.bias'])
        x = self.dropout(F.relu(x))
        x = F.linear(x, weights['w_2.weight'], weights['w_2.bias'])
        return x


class Transformer(nn.Module):
    def __init__(self, N=2, d_model=512, d_ff=2048, heads=10, dropout=0.1, device='cpu'):
        super().__init__()
        self.N = N
        self.device = device
        attn = MultiHeadedAttention(heads, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), dropout), N)

    def forward(self, x, mask):
        x = self.encoder(x, mask)
        return x

class TransformerSentenceEncoder(nn.Module):
    def __init__(self, args, d_ff=2048, heads=10, dropout=0.1):
        super().__init__()
        self.device = args.device
        self.embedding = Embedding(args.vectors,
                                  tune_embedding=args.tune_embedding,
                                  device=args.device)
        d_model= self.embedding.size
        attn = MultiHeadedAttention(heads, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), dropout), args.trans_n)
        # self.linear = nn.Linear(d_model*31 , d_model)
        # self.relu = nn.ReLU()

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
        mask = inputs['mask'].to(self.device)
        anchor = inputs['anchor_index'].view(-1)
        # print('| TransformerSentenceEncoder > anchor', tuple(anchor.shape))
        # print('| TransformerSentenceEncoder > mask1', tuple(mask.shape))

        mask = mask.view(-1, mask.shape[-1])
        # print('| TransformerSentenceEncoder > mask2', tuple(mask.shape))

        x = self.embedding(inputs)
        shape = list(x.shape)
        x = x.view(tuple([-1] + shape[-2:]))
        # print('| TransformerSentenceEncoder > x', tuple(x.shape))
        x = self.encoder(x, mask)

        x = self.select_anchor(x, anchor)
        x = x.view(tuple(shape[:-2] + [shape[-1]]))

        # print('| TransformerSentenceEncoder > x', tuple(x.shape))
        return x
