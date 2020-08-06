import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from collections import OrderedDict


def retrieveLocalEmbeddings(embed, anchors, window, device):
    zeros = torch.zeros(embed.size(0), window, embed.size(2)).float().to(device)
    zeros = Variable(zeros)

    padded = torch.cat([zeros, embed, zeros], 1)

    ids = []
    for i in range(2 * window + 1):
        ids += [(anchors + i).long().view(-1, 1)]
    ids = torch.cat(ids, 1)
    ids = ids.unsqueeze(2).expand(ids.size(0), ids.size(1), embed.size(2))

    res = padded.gather(1, ids)
    res = res.view(res.size(0), -1)
    return res


def clipTwoDimentions(mat, norm=3.0, device='cpu'):
    col_norms = ((mat ** 2).sum(0, keepdim=True)) ** 0.5
    desired_norms = col_norms.clamp(0.0, norm)
    scale = desired_norms / (1e-7 + col_norms)
    res = mat * scale
    res = res.to(device)
    return res


class BaseNet(nn.Module):

    def __init__(self, arguments):
        super(BaseNet, self).__init__()

        self.args = arguments

        self.word_embedding = nn.Embedding(self.args.embs['word'].shape[0], self.args.embs['word'].shape[1])

        self.embedding_input_dim = self.args.embs['word'].shape[1]

        if self.args.use_position:
            self.dist_embedding = nn.Embedding(self.args.embs['dist'].shape[0], self.args.embs['dist'].shape[1])
            self.embedding_input_dim += self.args.embs['dist'].shape[1]

        if self.args.use_ner:
            self.ner_embedding = nn.Embedding(self.args.embs['ner'].shape[0], self.args.embs['ner'].shape[1])
            self.embedding_input_dim += self.args.embs['ner'].shape[1]

        self.final_representation_dropout = nn.Dropout(self.args.final_representation_dropout_rate)

    def init_weights(self):
        self.word_embedding.weight = nn.Parameter(torch.from_numpy(self.args.embs['word']).float())

        if self.args.use_position:
            self.dist_embedding.weight = nn.Parameter(torch.from_numpy(self.args.embs['dist']).float())

        if self.args.use_ner:
            self.ner_embedding.weight = nn.Parameter(torch.from_numpy(self.args.embs['ner']).float())

    def createFcModule(self, dim_rep):

        if self.args.local_window > 0:
            dim_rep += (1 + 2 * self.args.local_window) * self.args.embs['word'].shape[1]

        rep_hids = [dim_rep] + self.args.final_feed_forward + [self.args.num_classes]

        ofcs = OrderedDict()
        for i, (ri, ro) in enumerate(zip(rep_hids, rep_hids[1:])):
            ofcs['finalRep_' + str(i)] = nn.Linear(ri, ro)
            # ofcs['finalNL_' + str(i)] = nn.Tanh()
        self.fc = nn.Sequential(ofcs)

    def clipEmbeddings(self):
        if self.args.use_position: self.dist_embedding.weight.data = clipTwoDimentions(self.dist_embedding.weight.data,
                                                                                       norm=self.args.norm_lim,
                                                                                       device=self.args.device)
        if self.args.use_ner: self.ner_embedding.weight.data = clipTwoDimentions(self.ner_embedding.weight.data,
                                                                                 norm=self.args.norm_lim,
                                                                                 device=self.args.device)

    def computeInputLengthMask(self, inputs):
        words = inputs['word']

        length = (words != 0).sum(1).long().to(self.args.device)

        mask = (words != 0).float().to(self.args.device)

        return length, mask

    def computeInputRep(self, inputs):
        inWord_embeddings = self.word_embedding(inputs['word'])

        inRep = [inWord_embeddings]

        inRep += [self.dist_embedding(inputs['dist'])]

        inRep = inRep[0] if len(inRep) == 1 else torch.cat(inRep, 2)

        length, mask = self.computeInputLengthMask(inputs)

        return inRep, length, mask

    def introduceLocalEmbedding(self, frep, inputs):

        assert type(frep) == list

        inWord_embeddings = self.word_embedding(inputs['word'])

        if self.args.local_window > 0:
            local_rep = retrieveLocalEmbeddings(inWord_embeddings, inputs['iniPos'], self.args.local_window,
                                                self.args.device)
            frep += [local_rep]

        frep = frep[0] if len(frep) == 1 else torch.cat(frep, 1)

        return frep

    def computeProbDist(self, rep):
        rep = self.final_representation_dropout(rep)
        out = self.fc(rep)
        # return [F.log_softmax(out, dim=1), rep]
        return [out, rep]


#### CNN ####
class CNN(BaseNet):

    def __init__(self, arguments):
        super(CNN, self).__init__(arguments)

        # if self.args['useRelDep']: embedding_input_dim += self.args['numRelDep']

        self.convs = nn.ModuleList([nn.Conv2d(1, self.args.cnn_kernel_number, (K, self.embedding_input_dim)) for K in
                                    self.args.cnn_kernel_sizes])

        self.dim_rep = self.args.cnn_kernel_number * len(self.args.cnn_kernel_sizes)

        self.createFcModule(self.dim_rep)

    def forward(self, inputs):
        inRep, _, _ = self.computeInputRep(inputs)

        inRep = inRep.unsqueeze(1)  # (B,1,T,D)

        convRep = [torch.tanh(conv(inRep)).squeeze(3) for conv in self.convs]  # [(B,Co,T), ...]*len(Ks)

        pooledRep = [F.max_pool1d(cr, cr.size(2)).squeeze(2) for cr in convRep]  # [(B,Co), ...]*len(Ks)

        frep = self.introduceLocalEmbedding(pooledRep, inputs)

        return self.computeProbDist(frep)


#### Nonconsecutive CNN ####

class nonConsecutiveConvLayer2(nn.Module):

    def __init__(self, feature_map, length, dim, device):
        super(nonConsecutiveConvLayer2, self).__init__()

        self.feature_map = feature_map
        self.length = length
        self.dim = dim
        self.device = device

        self.window = 2

        # fan_in = self.window * self.dim
        # fan_out = self.feature_map * self.window * self.dim / self.length #(length - window + 1)
        # W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.Ws = []
        for i in range(self.window):
            conv_W = nn.Parameter(torch.FloatTensor(self.dim, self.feature_map).to(self.device))
            nn.init.xavier_uniform_(conv_W.data)
            self.Ws += [conv_W]

        self.conv_b = nn.Parameter(torch.zeros(self.feature_map).to(self.device))

        # self.init_weights()

    def init_weights(self):
        for W in self.Ws: nn.init.xavier_uniform_(W.data)
        # nn.init.xavier_uniform(self.conv_b)

    def recurrence(self, x, hidden1, hidden2):
        ati = x @ self.Ws[0]
        m1 = ati.max(hidden1)
        ati = hidden1 + (x @ self.Ws[1])
        m2 = ati.max(hidden2)

        return m1, m2

    def forward(self, input, batch_first=True):
        if batch_first: input = input.transpose(0, 1)

        hiddens = []
        for i in range(self.window):
            hidden = Variable(torch.zeros(input.size(1), self.feature_map)).to(self.device)
            hiddens += [hidden]

        for i in range(input.size(0)):
            hiddens[0], hiddens[1] = self.recurrence(input[i], hiddens[0], hiddens[1])

        res = torch.tanh(hiddens[1] + self.conv_b.unsqueeze(0))

        return res


class nonConsecutiveConvLayer3(nn.Module):

    def __init__(self, feature_map, length, dim, device):
        super(nonConsecutiveConvLayer3, self).__init__()

        self.feature_map = feature_map
        self.length = length
        self.dim = dim
        self.device = device

        self.window = 3

        # fan_in = self.window * self.dim
        # fan_out = self.feature_map * self.window * self.dim / self.length #(length - window + 1)
        # W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.Ws = []
        for i in range(self.window):
            conv_W = nn.Parameter(torch.FloatTensor(self.dim, self.feature_map).to(self.device))
            nn.init.xavier_uniform_(conv_W.data)

            self.Ws += [conv_W]

        self.conv_b = nn.Parameter(torch.zeros(self.feature_map).to(self.device))

        # self.init_weights()

    def init_weights(self):
        for W in self.Ws: nn.init.xavier_uniform_(W.data)
        # nn.init.xavier_uniform(conv_b)

    def recurrence(self, x, hidden1, hidden2, hidden3):
        ati = x @ self.Ws[0]
        m1 = ati.max(hidden1)
        ati = hidden1 + (x @ self.Ws[1])
        m2 = ati.max(hidden2)
        ati = hidden2 + (x @ self.Ws[2])
        m3 = ati.max(hidden3)

        return m1, m2, m3

    def forward(self, input, batch_first=True):
        if batch_first: input = input.transpose(0, 1)

        hiddens = []
        for i in range(self.window):
            hidden = Variable(torch.zeros(input.size(1), self.feature_map)).to(self.device)
            hiddens += [hidden]

        for i in range(input.size(0)):
            hiddens[0], hiddens[1], hiddens[2] = self.recurrence(input[i], hiddens[0], hiddens[1], hiddens[2])

        res = torch.tanh(hiddens[2] + self.conv_b.unsqueeze(0))

        return res


class nonConsecutiveConvLayer4(nn.Module):

    def __init__(self, feature_map, length, dim, device):
        super(nonConsecutiveConvLayer4, self).__init__()

        self.feature_map = feature_map
        self.length = length
        self.dim = dim
        self.device = device

        self.window = 4

        # fan_in = self.window * self.dim
        # fan_out = self.feature_map * self.window * self.dim / self.length #(length - window + 1)
        # W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.Ws = []
        for i in range(self.window):
            conv_W = nn.Parameter(torch.FloatTensor(self.dim, self.feature_map).to(self.device))
            nn.init.xavier_uniform_(conv_W.data)

            self.Ws += [conv_W]

        self.conv_b = nn.Parameter(torch.zeros(self.feature_map).to(self.device))

        # self.init_weights()

    def init_weights(self):
        for W in self.Ws: nn.init.xavier_uniform_(W.data)
        # nn.init.xavier_uniform(conv_b)

    def recurrence(self, x, hidden1, hidden2, hidden3, hidden4):
        ati = x @ self.Ws[0]
        m1 = ati.max(hidden1)
        ati = hidden1 + (x @ self.Ws[1])
        m2 = ati.max(hidden2)
        ati = hidden2 + (x @ self.Ws[2])
        m3 = ati.max(hidden3)
        ati = hidden3 + (x @ self.Ws[3])
        m4 = ati.max(hidden4)

        return m1, m2, m3, m4

    def forward(self, input, batch_first=True):
        if batch_first: input = input.transpose(0, 1)

        hiddens = []
        for i in range(self.window):
            hidden = Variable(torch.zeros(input.size(1), self.feature_map)).to(self.device)
            hiddens += [hidden]

        for i in range(input.size(0)):
            hiddens[0], hiddens[1], hiddens[2], hiddens[3] = self.recurrence(input[i], hiddens[0], hiddens[1],
                                                                             hiddens[2], hiddens[3])

        res = torch.tanh(hiddens[3] + self.conv_b.unsqueeze(0))

        return res


class nonConsecutiveConvLayer5(nn.Module):

    def __init__(self, feature_map, length, dim, device):
        super(nonConsecutiveConvLayer5, self).__init__()

        self.feature_map = feature_map
        self.length = length
        self.dim = dim
        self.device = device

        self.window = 5

        # fan_in = self.window * self.dim
        # fan_out = self.feature_map * self.window * self.dim / self.length #(length - window + 1)
        # W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.Ws = []
        for i in range(self.window):
            conv_W = nn.Parameter(torch.FloatTensor(self.dim, self.feature_map).to(self.device))
            nn.init.xavier_uniform_(conv_W.data)

            self.Ws += [conv_W]

        self.conv_b = nn.Parameter(torch.zeros(self.feature_map).to(self.device))

        # self.init_weights()

    def init_weights(self):
        for W in self.Ws: nn.init.xavier_uniform_(W.data)
        # nn.init.xavier_uniform(conv_b)

    def recurrence(self, x, hidden1, hidden2, hidden3, hidden4, hidden5):
        ati = x @ self.Ws[0]
        m1 = ati.max(hidden1)
        ati = hidden1 + (x @ self.Ws[1])
        m2 = ati.max(hidden2)
        ati = hidden2 + (x @ self.Ws[2])
        m3 = ati.max(hidden3)
        ati = hidden3 + (x @ self.Ws[3])
        m4 = ati.max(hidden4)
        ati = hidden4 + (x @ self.Ws[4])
        m5 = ati.max(hidden5)

        return m1, m2, m3, m4, m5

    def forward(self, input, batch_first=True):
        if batch_first: input = input.transpose(0, 1)

        hiddens = []
        for i in range(self.window):
            hidden = Variable(torch.zeros(input.size(1), self.feature_map)).to(self.device)
            hiddens += [hidden]

        for i in range(input.size(0)):
            hiddens[0], hiddens[1], hiddens[2], hiddens[3], hiddens[4] = self.recurrence(input[i], hiddens[0],
                                                                                         hiddens[1], hiddens[2],
                                                                                         hiddens[3], hiddens[4])

        res = torch.tanh(hiddens[4] + self.conv_b.unsqueeze(0))

        return res


class NCNN(BaseNet):

    def __init__(self, arguments):
        super(NCNN, self).__init__(arguments)

        self.convs = nn.ModuleList([eval('nonConsecutiveConvLayer' + str(K))(self.args.cnn_kernel_number,
                                                                             self.args.sent_len,
                                                                             self.embedding_input_dim,
                                                                             self.args.device)
                                    for K in self.args.cnn_kernel_sizes])

        self.dim_rep = self.args.cnn_kernel_number * len(self.args.cnn_kernel_sizes)

        self.createFcModule(self.dim_rep)

    def forward(self, inputs):
        inRep, _, _ = self.computeInputRep(inputs)  # (B,T,D)

        convRep = [conv(inRep) for conv in self.convs]

        frep = self.introduceLocalEmbedding(convRep, inputs)

        return self.computeProbDist(frep)

    # def clipEmbeddings(self):
    #    if self.args.use_position: self.dist_embedding.weight.data = clipTwoDimentions(self.dist_embedding.weight.data,
    #                                                                                   norm=self.args.norm_lim,
    #                                                                                   device=self.args.device)
    #    if self.args.use_ner: self.ner_embedding.weight.data = clipTwoDimentions(self.ner_embedding.weight.data,
    #                                                                             norm=self.args.norm_lim,
    #                                                                             device=self.args.device)

    #    for conv in self.convs:
    #        for i in range(len(conv.Ws)):
    #            conv.Ws[i].data = clipTwoDimentions(conv.Ws[i].data,
    #                                                norm=self.args.norm_lim,
    #                                                device=self.args.device)


#### Graph CNN ####
class GCNNLayer(nn.Module):

    def __init__(self, input_dim, output_dim, edge_patterns, num_dep_relation, device):
        super(GCNNLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_patterns = edge_patterns
        self.num_dep_relation = num_dep_relation
        self.device = device

        if self.edge_patterns[0]:  # using in links
            self.conv_W_in = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim).to(self.device))
            nn.init.xavier_normal_(self.conv_W_in.data)
            self.conv_b_in = nn.Parameter(torch.zeros(self.num_dep_relation, self.output_dim).to(self.device))

            self.conv_W_gate_in = nn.Parameter(torch.FloatTensor(self.input_dim, 1).uniform_().to(self.device))
            nn.init.uniform_(self.conv_W_gate_in.data)
            self.conv_b_gate_in = nn.Parameter(torch.ones(self.num_dep_relation, 1).to(self.device))

        if self.edge_patterns[1]:  # using out links
            self.conv_W_out = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim).to(self.device))
            nn.init.xavier_normal_(self.conv_W_out.data)
            self.conv_b_out = nn.Parameter(torch.zeros(self.num_dep_relation, self.output_dim).to(self.device))

            self.conv_W_gate_out = nn.Parameter(torch.FloatTensor(self.input_dim, 1).to(self.device))
            nn.init.uniform_(self.conv_W_gate_out.data)
            self.conv_b_gate_out = nn.Parameter(torch.ones(self.num_dep_relation, 1).to(self.device))

        if self.edge_patterns[2]:  # using self-loop links
            self.conv_W_self = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim).to(self.device))
            nn.init.xavier_normal_(self.conv_W_self.data)

            self.conv_W_gate_self = nn.Parameter(torch.FloatTensor(self.input_dim, 1).to(self.device))
            nn.init.uniform_(self.conv_W_gate_self.data)

    def forward(self, rep, inputs):

        batch_size, lent_size, input_dim = rep.size()
        max_degree = 0

        rep_ = rep.view(batch_size * lent_size, self.input_dim)  # (b*l, d)

        potentials, potentials_gate, mask_soft = [], [], []

        if self.edge_patterns[0]:
            # transformation
            input_in = torch.mm(rep_, self.conv_W_in)  # (b*l, do)
            first_in = input_in[inputs['adj_arc_in'][0] * lent_size + inputs['adj_arc_in'][1]]  # (b*l, do)
            second_in = self.conv_b_in[inputs['adj_lab_in'][0]]  # (b*l, do)

            in_ = (first_in + second_in).view(batch_size, lent_size, 1, self.output_dim)
            potentials.append(in_)

            # compute gate weights
            input_in_gate = torch.mm(rep_, self.conv_W_gate_in)  # (b*l, 1)
            first_in_gate = input_in_gate[inputs['adj_arc_in'][0] * lent_size + inputs['adj_arc_in'][1]]  # [b*l, 1]
            second_in_gate = self.conv_b_gate_in[inputs['adj_lab_in'][0]]  # (b*l, 1)

            in_gate = (first_in_gate + second_in_gate).view(batch_size, lent_size, 1)
            potentials_gate.append(in_gate)

            mask_soft.append(inputs['adj_mask_in'])

            max_degree += 1
        if self.edge_patterns[1]:
            # transformation
            input_out = torch.mm(rep_, self.conv_W_out)  # (b*l, do)
            first_out = input_out[inputs['adj_arc_out'][0] * lent_size + inputs['adj_arc_out'][1]]  # (b*l*degree, do)
            second_out = self.conv_b_out[inputs['adj_lab_out'][0]]  # (b*l*degree, do)
            degr = int(first_out.size(0) / batch_size / lent_size)
            max_degree += degr

            out_ = (first_out + second_out).view(batch_size, lent_size, degr, self.output_dim)
            potentials.append(out_)

            # compute gate weights
            input_out_gate = torch.mm(rep_, self.conv_W_gate_out)  # (b*l, 1)
            first_out_gate = input_out_gate[
                inputs['adj_arc_out'][0] * lent_size + inputs['adj_arc_out'][1]]  # [b*l*degree, 1]
            second_out_gate = self.conv_b_gate_out[inputs['adj_lab_out'][0]]  # (b*l*degree, 1)

            out_gate = (first_out_gate + second_out_gate).view(batch_size, lent_size, degr)
            potentials_gate.append(out_gate)

            mask_soft.append(inputs['adj_mask_out'])

        if self.edge_patterns[2]:
            # transformation

            same_ = torch.mm(rep_, self.conv_W_self).view(batch_size, lent_size, 1, self.output_dim)
            potentials.append(same_)

            # compute gate weights
            same_gate = torch.mm(rep_, self.conv_W_gate_self).view(batch_size, lent_size, 1)
            potentials_gate.append(same_gate)

            max_degree += 1

            mask_soft.append(inputs['adj_mask_loop'])

        potentials = torch.cat(potentials, 2)  # b x l x degree x do
        potentials_gate = torch.cat(potentials_gate, 2)  # b x l x degree
        mask_soft = torch.cat(mask_soft, 1)  # (b*l) x degree

        potentials_ = potentials.permute(3, 0, 1, 2)  # do x b x l x degree
        potentials_resh = potentials_.view(self.output_dim, batch_size * lent_size, -1)  # do x b*l x degree

        # calculate the gate
        potentials_gate_ = potentials_gate.view(batch_size * lent_size, -1)  # b*l x degree
        probs_det_ = torch.sigmoid(potentials_gate_) * mask_soft
        probs_det_ = probs_det_.unsqueeze(0)
        mask_soft = mask_soft.unsqueeze(0)

        potentials_masked = potentials_resh * mask_soft * probs_det_  # do x b*l x degree
        potentials_masked_ = potentials_masked.sum(2)  # [do, b * l]
        potentials_masked_ = torch.relu(potentials_masked_).transpose(0, 1)  # b*l x do

        res = potentials_masked_.view(batch_size, lent_size, self.output_dim)
        res = res * inputs['mask_input'].unsqueeze(2)

        return res


class GCNN(BaseNet):

    def __init__(self, arguments):

        super(GCNN, self).__init__(arguments)

        self.lstm = LSTMLayer(self.embedding_input_dim, self.args.rnn_num_hidden_units, self.args.batch_size,
                              self.args.device)

        self.gcnnLayers = []
        dims = [self.lstm.dim_rep] + self.args.gcnn_kernel_numbers
        for i, (input_dim, output_dim) in enumerate(zip(dims, dims[1:])):
            layer = GCNNLayer(input_dim, output_dim, self.args.gcnn_edge_patterns, self.args.num_rel_dep,
                              self.args.device)
            self.add_module('GCNNLayer' + str(i), layer)
            self.gcnnLayers.append(layer)

        self.dim_rep = 2 * output_dim if 'dynamic' in self.args.gcnn_pooling else output_dim
        self.createFcModule(self.dim_rep)

    def forward(self, inputs):

        inRep, lengths, mask = self.computeInputRep(inputs)

        outRep = self.lstm(inRep, lengths)

        for layer in self.gcnnLayers:
            outRep = layer(outRep, inputs)

        rep = eval(self.args.gcnn_pooling)(outRep, inputs['iniPos'], mask, inputs['mner'], self.args.device)

        frep = self.introduceLocalEmbedding([rep], inputs)

        return self.computeProbDist(frep)


#### GRU ####

class GRULayer(nn.Module):
    def __init__(self, embedding_input_dim, rnn_num_hidden_units, batch_size, device):

        super(GRULayer, self).__init__()

        self.embedding_input_dim = embedding_input_dim
        self.rnn_num_hidden_units = rnn_num_hidden_units
        self.batch_size = batch_size
        self.device = device

        self.gru = nn.GRU(self.embedding_input_dim, self.rnn_num_hidden_units, num_layers=1, batch_first=True,
                          bidirectional=True)

        self.dim_rep = 1 * 2 * self.rnn_num_hidden_units

    def initHidden(self):
        h0 = torch.zeros(2 * 1, self.batch_size, self.rnn_num_hidden_units).to(self.device)
        return Variable(h0)

    def forward(self, inRep, lengths):

        initLength = inRep.shape[1]
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        iperm_idx = torch.LongTensor(perm_idx.shape).fill_(0).to(self.device)
        for i, v in enumerate(perm_idx):
            iperm_idx[v.data] = i
        inRep = inRep[perm_idx]

        inRep = pack_padded_sequence(inRep, seq_lengths.data.cpu().numpy(), batch_first=True)

        h0 = self.initHidden()

        outRep, h_n = self.gru(inRep, h0)

        outRep, _ = pad_packed_sequence(outRep, batch_first=True)
        outRep = outRep[iperm_idx]

        if outRep.shape[1] < initLength:
            zeros = Variable(
                torch.FloatTensor(outRep.shape[0], initLength - outRep.shape[1], outRep.shape[2]).fill_(0.).to(
                    self.args.device))
            outRep = torch.cat([outRep, zeros], 1)

        return outRep


class GRU(BaseNet):

    def __init__(self, arguments):
        super(GRU, self).__init__(arguments)

        # self.gru = nn.GRU(self.embedding_input_dim, self.args.rnn_num_hidden_units, num_layers=1, batch_first=True, bidirectional=True)
        self.gru = GRULayer(self.embedding_input_dim, self.args.rnn_num_hidden_units, self.args.batch_size,
                            self.args.device)

        self.dim_rep = 2 * self.gru.dim_rep if 'dynamic' in self.args.rnn_pooling else self.gru.dim_rep
        self.createFcModule(self.dim_rep)

    def forward(self, inputs):
        inRep, lengths, mask = self.computeInputRep(inputs)

        outRep = self.gru(inRep, lengths)

        rnnRep = eval(self.args.rnn_pooling)(outRep, inputs['iniPos'], mask, inputs['mner'], self.args.device)

        frep = self.introduceLocalEmbedding([rnnRep], inputs)

        return self.computeProbDist(frep)


#### LSTM ####

class LSTMLayer(nn.Module):
    def __init__(self, embedding_input_dim, rnn_num_hidden_units, batch_size, device):

        super(LSTMLayer, self).__init__()

        self.embedding_input_dim = embedding_input_dim
        self.rnn_num_hidden_units = rnn_num_hidden_units
        self.batch_size = batch_size
        self.device = device

        self.lstm = nn.LSTM(self.embedding_input_dim, self.rnn_num_hidden_units, num_layers=1, batch_first=True,
                            bidirectional=True)

        self.dim_rep = 1 * 2 * self.rnn_num_hidden_units

    def initHidden(self):
        h0 = torch.zeros(2 * 1, self.batch_size, self.rnn_num_hidden_units).to(self.device)
        c0 = torch.zeros(2 * 1, self.batch_size, self.rnn_num_hidden_units).to(self.device)
        return Variable(h0), Variable(c0)

    def forward(self, inRep, lengths):

        initLength = inRep.shape[1]
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        iperm_idx = torch.LongTensor(perm_idx.shape).fill_(0).to(self.device)
        for i, v in enumerate(perm_idx):
            iperm_idx[v.data] = i
        inRep = inRep[perm_idx]

        inRep = pack_padded_sequence(inRep, seq_lengths.data.cpu().numpy(), batch_first=True)

        h0, c0 = self.initHidden()

        outRep, h_n = self.lstm(inRep, (h0, c0))

        outRep, _ = pad_packed_sequence(outRep, batch_first=True)
        outRep = outRep[iperm_idx]

        if outRep.shape[1] < initLength:
            zeros = Variable(
                torch.FloatTensor(outRep.shape[0], initLength - outRep.shape[1], outRep.shape[2]).fill_(0.).to(
                    self.args.device))
            outRep = torch.cat([outRep, zeros], 1)

        return outRep


class LSTM(BaseNet):

    def __init__(self, arguments):
        super(LSTM, self).__init__(arguments)

        # self.gru = nn.GRU(self.embedding_input_dim, self.args.rnn_num_hidden_units, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm = LSTMLayer(self.embedding_input_dim, self.args.rnn_num_hidden_units, self.args.batch_size,
                              self.args.device)

        self.dim_rep = 2 * self.lstm.dim_rep if 'dynamic' in self.args.rnn_pooling else self.lstm.dim_rep
        self.createFcModule(self.dim_rep)

    def forward(self, inputs):
        inRep, lengths, mask = self.computeInputRep(inputs)

        outRep = self.lstm(inRep, lengths)

        rnnRep = eval(self.args.rnn_pooling)(outRep, inputs['iniPos'], mask, inputs['mner'], self.args.device)

        frep = self.introduceLocalEmbedding([rnnRep], inputs)

        return self.computeProbDist(frep)


#### Pooling Methods ######

def pool_anchor(rep, iniPos, mask, nmask, device):
    ids = iniPos.view(-1, 1)
    ids = ids.expand(ids.size(0), rep.size(2)).unsqueeze(1)

    res = rep.gather(1, ids)
    res = res.squeeze(1)
    return res


def pool_max(rep, iniPos, mask, nmask, device):
    rep = torch.exp(rep) * mask.unsqueeze(2)
    res = torch.log(rep.max(1)[0])
    return res


def pool_dynamic(rep, iniPos, mask, nmask, device):
    rep = torch.exp(rep) * mask.unsqueeze(2)
    left, right = [], []
    batch, lent, dim = rep.size(0), rep.size(1), rep.size(2)
    for i in range(batch):
        r, id, ma = rep[i], iniPos.tolist()[i], mask[i]
        tleft = torch.log(r[0:(id + 1)].max(0)[0].unsqueeze(0))
        left += [tleft]
        if (id + 1) < lent and ma.cpu().numpy()[(id + 1):].sum() > 0:
            tright = torch.log(r[(id + 1):].max(0)[0].unsqueeze(0))
        else:
            tright = Variable(torch.zeros(1, dim).to(device))
        right += [tright]
    left = torch.cat(left, 0)
    right = torch.cat(right, 0)
    res = torch.cat([left, right], 1)

    return res


def pool_entity(rep, iniPos, mask, nmask, device):
    rep = torch.exp(rep) * nmask.unsqueeze(2)
    res = torch.log(rep.max(1)[0])
    return res


def max_along_time(rep, lengths):
    """
    :param rep: [B * T * D]
    :param lengths:  [B]
    :return: [B * D] max_along_time
    """
    rep = rep.permute(1, 0, 2)
    # [T * B * D]
    ls = list(lengths)

    b_seq_max_list = []
    for i, l in enumerate(ls):
        seq_i = rep[:l, i, :]
        seq_i_max, _ = seq_i.max(dim=0)
        seq_i_max = seq_i_max.squeeze()
        b_seq_max_list.append(seq_i_max)

    return torch.stack(b_seq_max_list)
