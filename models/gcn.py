import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import OrderedDict
from models.embedding import Embedding


def retrieveLocalEmbeddings(embed, anchors, window, device):
    zeros = torch.zeros(embed.size(0), window, embed.size(2)).float().to(device)

    padded = torch.cat([zeros, embed, zeros], 1)

    ids = []
    for i in range(2 * window + 1):
        ids += [(anchors + i).long().view(-1, 1)]
    ids = torch.cat(ids, 1)
    ids = ids.unsqueeze(2).expand(ids.size(0), ids.size(1), embed.size(2)).to(device)

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

    def __init__(self, args):
        super(BaseNet, self).__init__()
        self.device = args.device
        self.hidden_size = args.hidden_size
        self.cnn_kernel_sizes = [2, 3, 4, 5]
        self.cnn_kernel_number = 150

        self.embedder = Embedding(args.vectors,
                                  tune_embedding=args.tune_embedding,
                                  device=args.device)
        self.window = args.window
        self.embedding_input_dim = self.embedder.size
        self.conv = nn.Conv1d(self.embedding_input_dim, self.hidden_size, 3, padding=1)
        self.pool = nn.MaxPool1d(args.max_length)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(args.dropout)

    def createFcModule(self, dim_rep):

        if self.window > 0:
            dim_rep += (1 + 2 * self.window) * self.embedder.size

        rep_hids = [dim_rep, self.hidden_size, self.hidden_size]

        ofcs = OrderedDict()
        for i, (ri, ro) in enumerate(zip(rep_hids, rep_hids[1:])):
            ofcs['finalRep_' + str(i)] = nn.Linear(ri, ro)
            # ofcs['finalNL_' + str(i)] = nn.Tanh()
        self.fc = nn.Sequential(ofcs)

    def computeInputRep(self, inputs):
        length = inputs['length'].view(-1)
        mask = inputs['mask'].view(-1)

        inRep = self.embedder(inputs)

        return inRep, length, mask

    def introduceLocalEmbedding(self, frep, inputs):

        assert type(frep) == list

        inWord_embeddings = self.embedder(inputs)
        shape = inWord_embeddings.shape

        # print('| Basenet > introduceLocalEmbedding > inWord_embeddings', tuple(inWord_embeddings.shape))

        if self.window > 0:
            local_rep = retrieveLocalEmbeddings(inWord_embeddings.view(-1, shape[-2], shape[-1]),
                                                inputs['anchor_index'], self.window,
                                                self.device)
            frep += [local_rep]

        frep = frep[0] if len(frep) == 1 else torch.cat(frep, 1)

        return frep


#### CNN ####
class CNN(BaseNet):

    def __init__(self, arguments):
        super(CNN, self).__init__(arguments)

        # if self['useRelDep']: embedding_input_dim += self['numRelDep']

        self.convs = nn.ModuleList([nn.Conv2d(1, self.cnn_kernel_number, (K, self.embedding_input_dim)) for K in
                                    self.cnn_kernel_sizes])

        self.dim_rep = self.cnn_kernel_number * len(self.cnn_kernel_sizes)

        self.createFcModule(self.dim_rep)

    def forward(self, inputs):
        inRep, _, _ = self.computeInputRep(inputs)
        B, N, K, L, D = inRep.shape
        BNK = B * N * K
        # print('| CNN > inRep', tuple(inRep.shape))
        inRep = inRep.view(BNK, L, D).unsqueeze(1)  # (B,1,T,D)
        # print('| CNN > inRep', tuple(inRep.shape))

        convRep = [torch.tanh(conv(inRep)).squeeze(3) for conv in self.convs]  # [(B,Co,T), ...]*len(Ks)
        # print('| CNN > convRep', tuple(convRep[0].shape))

        pooledRep = [F.max_pool1d(cr, cr.size(2)).squeeze(2) for cr in convRep]  # [(B,Co), ...]*len(Ks)
        # print('| CNN > pooledRep', tuple(pooledRep[0].shape))

        frep = self.introduceLocalEmbedding(pooledRep, inputs)
        # print('| CNN > frep', tuple(frep.shape))

        frep = self.fc(self.dropout(frep)).view(B,N,K,-1)
        # print('| CNN > frep', tuple(frep.shape))

        return frep


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

        B, N, K, L, DG = inputs['adj_mask_out'].shape
        BNK = B * N * K
        adj_arc_in = inputs['adj_arc_in'].view(-1, 2).to(self.device)
        adj_lab_in = inputs['adj_lab_in'].view(-1).to(self.device)
        # adj_arc_out = inputs['adj_arc_out'].view(-1, DG, 2).to(self.device)
        # adj_lab_out = inputs['adj_lab_out'].view(-1, DG).to(self.device)
        adj_mask_in = inputs['adj_mask_in'].view(-1, 1).to(self.device)
        adj_mask_out = inputs['adj_mask_out'].view(-1, DG).to(self.device)
        adj_mask_loop = inputs['adj_mask_loop'].view(-1, 1).to(self.device)
        mask = inputs['mask'].view(-1, L).to(self.device)
        max_degree = 0

        # print('-----')
        # print('| GCNLayer > adj_arc_in', tuple(adj_arc_in.shape))
        # print('| GCNLayer > adj_lab_in', tuple(adj_lab_in.shape))
        # print('| GCNLayer > adj_arc_out', tuple(adj_arc_out.shape))
        # print('| GCNLayer > adj_lab_out', tuple(adj_lab_out.shape))
        # print('| GCNLayer > adj_mask_in', tuple(adj_mask_in.shape))
        # print('| GCNLayer > adj_mask_out', tuple(adj_mask_out.shape))
        # print('| GCNLayer > adj_mask_loop', tuple(adj_mask_loop.shape))
        # print('| GCNLayer > mask', tuple(mask.shape))
        # print('-----')

        rep_ = rep.view(BNK * L, self.input_dim)  # (b*l, d)

        potentials, potentials_gate, mask_soft = [], [], []

        if self.edge_patterns[0]:
            # transformation
            input_in = torch.mm(rep_, self.conv_W_in)  # (b*l, do)
            # print('| GCNLayer > input_in', tuple(input_in.shape))
            first_in = input_in[adj_arc_in[:, 0] * L + adj_arc_in[:, 1]]  # (b*l, do)
            # print('| GCNLayer > first_in', tuple(first_in.shape))

            second_in = self.conv_b_in[adj_lab_in]  # (b*l, do)
            # print('| GCNLayer > second_in', tuple(second_in.shape))

            in_ = (first_in + second_in).view(BNK * L, 1, self.output_dim)
            # print('| GCNLayer > in_', tuple(in_.shape))

            potentials.append(in_)

            # compute gate weights
            input_in_gate = torch.mm(rep_, self.conv_W_gate_in)  # (b*l, 1)
            # print('| GCNLayer > input_in_gate', tuple(input_in_gate.shape))

            first_in_gate = input_in_gate[adj_arc_in[:, 0] * L + adj_arc_in[:, 1]]  # [b*l, 1]
            # print('| GCNLayer > first_in_gate', tuple(first_in_gate.shape))

            second_in_gate = self.conv_b_gate_in[adj_lab_in]  # (b*l, 1)
            # print('| GCNLayer > second_in_gate', tuple(second_in_gate.shape))

            in_gate = (first_in_gate + second_in_gate).view(BNK, L, 1)
            potentials_gate.append(in_gate)
            # print('| GCNLayer > in_gate', tuple(in_gate.shape))

            mask_soft.append(adj_mask_in)

            max_degree += 1
        # print('--------')
        # if self.edge_patterns[1]:
        #     # transformation
        #     input_out = torch.mm(rep_, self.conv_W_out)  # (b*l, do)
        #     # print('| GCNLayer > input_out', tuple(input_out.shape))
        #     first_out = input_out[adj_arc_out[:, :, 0] * L + adj_arc_out[:, :, 1]]  # (b*l x degree x do)
        #     # print('| GCNLayer > first_out', tuple(first_out.shape))
        #     second_out = self.conv_b_out[adj_lab_out]  # (b*l x degree x do)
        #     # print('| GCNLayer > second_out', tuple(second_out.shape))
        #
        #     degr = int(first_out.size(0) / BNK / L)
        #     max_degree += degr
        #
        #     out_ = (first_out + second_out).view(BNK * L, DG, self.output_dim)
        #     # print('| GCNLayer > out_', tuple(out_.shape))
        #
        #     potentials.append(out_)
        #
        #     # compute gate weights
        #     input_out_gate = torch.mm(rep_, self.conv_W_gate_out)  # (b*l, 1)
        #     first_out_gate = input_out_gate[
        #         adj_arc_out[:, :, 0] * L + adj_arc_out[:, :, 1]]  # [b*l, degree, 1]
        #     # print('| GCNLayer > first_out_gate', tuple(first_out_gate.shape))
        #
        #     second_out_gate = self.conv_b_gate_out[adj_lab_out]  # (b*l,degree, 1)
        #     # print('| GCNLayer > second_out_gate', tuple(second_out_gate.shape))
        #
        #     out_gate = (first_out_gate + second_out_gate).view(BNK, L, DG)
        #     potentials_gate.append(out_gate)
        # print('| GCNLayer > out_gate', tuple(out_gate.shape))

        # mask_soft.append(adj_mask_out)
        # print('--------')
        if self.edge_patterns[2]:
            # transformation
            same_ = torch.mm(rep_, self.conv_W_self).view(BNK * L, 1, self.output_dim)
            potentials.append(same_)
            # print('| GCNLayer > same_', tuple(same_.shape))

            # compute gate weights
            same_gate = torch.mm(rep_, self.conv_W_gate_self).view(BNK, L, 1)
            # print('| GCNLayer > same_gate', tuple(same_gate.shape))

            potentials_gate.append(same_gate)

            max_degree += 1
            # print('| GCNLayer > adj_mask_loop', tuple(adj_mask_loop.shape))

            mask_soft.append(adj_mask_loop)
        # print('--------')
        potentials = torch.cat(potentials, 1)  # b*l x degree x do
        # print('| GCNLayer > potentials', tuple(potentials.shape))

        potentials_gate = torch.cat(potentials_gate, 2)  # b x l x degree
        mask_soft = torch.cat(mask_soft, 1)  # (b*l) x degree
        # print('| GCNLayer > mask_soft', tuple(mask_soft.shape))

        potentials_ = potentials.permute(2, 0, 1)  # do x b*l x degree
        potentials_resh = potentials_.view(self.output_dim, BNK * L, -1)  # do x b*l x degree

        # calculate the gate
        potentials_gate_ = potentials_gate.view(BNK * L, -1)  # b*l x degree
        # print('| GCNLayer > potentials_gate_', tuple(potentials_gate_.shape))
        # print('| GCNLayer > mask_soft', tuple(mask_soft.shape))

        # # print('| GCNLayer > potentials_gate_ > device', potentials_gate_.device)
        # # print('| GCNLayer > mask_soft > device', mask_soft.device)

        probs_det_ = torch.sigmoid(potentials_gate_) * mask_soft
        probs_det_ = probs_det_.unsqueeze(0)
        mask_soft = mask_soft.unsqueeze(0)

        potentials_masked = potentials_resh * mask_soft * probs_det_  # do x b*l x degree
        potentials_masked_ = potentials_masked.sum(2)  # [do, b * l]
        potentials_masked_ = torch.relu(potentials_masked_).transpose(0, 1)  # b*l x do

        res = potentials_masked_.view(BNK, L, self.output_dim)
        # print('| GCNLayer > res', tuple(res.shape))

        res = res * mask.unsqueeze(2)

        return res


class GCNN(BaseNet):

    def __init__(self, args):

        super(GCNN, self).__init__(args)
        self.gcnn_edge_patterns = args.gcnn_edge_patterns
        self.gcnn_pooling = args.gcnn_pooling
        self.num_rel_dep = args.num_rel_dep

        self.lstm = LSTMLayer(self.embedding_input_dim, self.hidden_size, args.device)

        self.gcnnLayers = []
        dims = [self.lstm.dim_rep, self.hidden_size, self.hidden_size]
        for i, (input_dim, output_dim) in enumerate(zip(dims, dims[1:])):
            layer = GCNNLayer(input_dim, output_dim, self.gcnn_edge_patterns, self.num_rel_dep,
                              self.device)
            self.add_module('GCNNLayer' + str(i), layer)
            self.gcnnLayers.append(layer)

        self.dim_rep = 2 * output_dim if 'dynamic' in self.gcnn_pooling else output_dim
        self.createFcModule(self.dim_rep)

    def forward(self, inputs):

        for k, v in inputs.items():
            v.to(self.device)

        inRep, lengths, mask = self.computeInputRep(inputs)

        B, N, K, L, D = inRep.shape
        # # print('| GCNN > inRep', tuple(inRep.shape))
        # # print('| GCNN > lengths', tuple(lengths.shape))
        # # print('| GCNN > mask', tuple(mask.shape))

        outRep = self.lstm(inRep, lengths)

        for layer in self.gcnnLayers:
            outRep = layer(outRep, inputs)

        rep = eval(self.gcnn_pooling)(outRep, inputs['anchor_index'], mask, inputs['mner'], self.device)

        frep = self.introduceLocalEmbedding([rep], inputs)

        frep = self.dropout(frep).view(B, N, K, -1)
        frep = self.fc(frep)

        # print('| GCNN > frep', tuple(frep.shape))

        return frep


#### GRU ####

class GRULayer(nn.Module):
    def __init__(self, embedding_input_dim, rnn_num_hidden_units, device):

        super(GRULayer, self).__init__()

        self.embedding_input_dim = embedding_input_dim
        self.rnn_num_hidden_units = rnn_num_hidden_units
        self.device = device

        self.gru = nn.GRU(self.embedding_input_dim, self.rnn_num_hidden_units, num_layers=1, batch_first=True,
                          bidirectional=True)

        self.dim_rep = 1 * 2 * self.rnn_num_hidden_units

    def initHidden(self, B):
        h0 = torch.zeros(2 * 1, B, self.rnn_num_hidden_units).to(self.device)
        return h0

    def forward(self, inRep, lengths):

        B, L, D = inRep.shape
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        iperm_idx = torch.LongTensor(perm_idx.shape).fill_(0).to(self.device)
        for i, v in enumerate(perm_idx):
            iperm_idx[v.data] = i
        inRep = inRep[perm_idx]

        inRep = pack_padded_sequence(inRep, seq_lengths.data.cpu().numpy(), batch_first=True)

        h0 = self.initHidden(B)

        outRep, h_n = self.gru(inRep, h0)

        outRep, _ = pad_packed_sequence(outRep, batch_first=True)
        outRep = outRep[iperm_idx]

        if outRep.shape[1] < L:
            zeros = torch.zeros(outRep.shape[0], L - outRep.shape[1], outRep.shape[2]).to(self.device)
            outRep = torch.cat([outRep, zeros], 1)

        return outRep


class GRU(BaseNet):

    def __init__(self, arguments):
        super(GRU, self).__init__(arguments)

        # self.gru = nn.GRU(self.embedding_input_dim, self.rnn_num_hidden_units, num_layers=1, batch_first=True, bidirectional=True)
        self.gru = GRULayer(self.embedding_input_dim, self.rnn_num_hidden_units,
                            self.device)

        self.dim_rep = 2 * self.gru.dim_rep if 'dynamic' in self.rnn_pooling else self.gru.dim_rep
        self.createFcModule(self.dim_rep)

    def forward(self, inputs):
        inRep, lengths, mask = self.computeInputRep(inputs)

        outRep = self.gru(inRep, lengths)

        rnnRep = eval(self.rnn_pooling)(outRep, inputs['iniPos'], mask, inputs['mner'], self.device)

        frep = self.introduceLocalEmbedding([rnnRep], inputs)

        frep = self.dropout(frep)

        return frep


#### LSTM ####

class LSTMLayer(nn.Module):
    def __init__(self, embedding_input_dim, rnn_num_hidden_units, device):

        super(LSTMLayer, self).__init__()

        self.embedding_input_dim = embedding_input_dim
        self.rnn_num_hidden_units = rnn_num_hidden_units
        self.device = device

        self.lstm = nn.LSTM(self.embedding_input_dim, self.rnn_num_hidden_units, num_layers=1, batch_first=True,
                            bidirectional=True)

        self.dim_rep = 1 * 2 * self.rnn_num_hidden_units

    def initHidden(self, B):
        h0 = torch.zeros(2 * 1, B, self.rnn_num_hidden_units).to(self.device)
        c0 = torch.zeros(2 * 1, B, self.rnn_num_hidden_units).to(self.device)
        return h0, c0

    def forward(self, inRep, lengths):
        """

        :param inRep:
        :param lengths:
        :return:
        """
        shape = inRep.shape
        inRep = inRep.view(-1, shape[-2], shape[-1])

        B, L, D = inRep.shape

        initLength = inRep.shape[1]
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        iperm_idx = torch.LongTensor(perm_idx.shape).to(self.device)
        for i, v in enumerate(perm_idx):
            iperm_idx[v.data] = i
        inRep = inRep[perm_idx]

        inRep = pack_padded_sequence(inRep, seq_lengths.data.cpu().numpy(), batch_first=True)

        h0, c0 = self.initHidden(B)

        outRep, h_n = self.lstm(inRep, (h0, c0))

        outRep, _ = pad_packed_sequence(outRep, batch_first=True)
        outRep = outRep[iperm_idx]

        if outRep.shape[1] < initLength:
            zeros = torch.zeros(outRep.shape[0], initLength - outRep.shape[1], outRep.shape[2]).to(
                self.device)
            outRep = torch.cat([outRep, zeros], 1)

        return outRep


class LSTM(BaseNet):

    def __init__(self, arguments):
        super(LSTM, self).__init__(arguments)

        # self.gru = nn.GRU(self.embedding_input_dim, self.rnn_num_hidden_units, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm = LSTMLayer(self.embedding_input_dim, self.rnn_num_hidden_units, self.device)

        self.dim_rep = 2 * self.lstm.dim_rep if 'dynamic' in self.rnn_pooling else self.lstm.dim_rep
        self.createFcModule(self.dim_rep)

    def forward(self, inputs):
        inRep, lengths, mask = self.computeInputRep(inputs)

        outRep = self.lstm(inRep, lengths)

        rnnRep = eval(self.rnn_pooling)(outRep, inputs['iniPos'], mask, inputs['mner'], self.device)

        frep = self.introduceLocalEmbedding([rnnRep], inputs)

        frep = self.dropout(frep)

        return frep


#### Pooling Methods ######

def pool_anchor(rep, iniPos, mask, nmask, device):
    ids = iniPos.view(-1, 1)
    ids = ids.expand(ids.size(0), rep.size(2)).unsqueeze(1).to(device)

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
