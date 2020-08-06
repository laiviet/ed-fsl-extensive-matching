import torch
import torch.nn as nn


def clipTwoDimentions(mat, norm=3.0):
    col_norms = ((mat ** 2).sum(0, keepdim=True)) ** 0.5
    desired_norms = col_norms.clamp(0.0, norm)
    scale = desired_norms / (1e-7 + col_norms)
    res = mat * scale
    res = res.cuda()
    return res


class Embedding(nn.Module):

    def __init__(self, word_vec_mat,
                 max_length=31,
                 pos_embedding_dim=50,
                 norm_lim=3.0,
                 tune_embedding=False,
                 device='cpu'):
        super(Embedding, self).__init__()

        self.device = device
        self.norm_lim = norm_lim
        print('| Embedding > device', device, '; Tune embedding: ', tune_embedding)
        self.max_length = max_length
        self.word_embedding_dim = word_vec_mat.shape[1]
        self.pos_embedding_dim = pos_embedding_dim
        self.size = word_vec_mat.shape[1] + pos_embedding_dim

        # print(type(word_vec_mat))
        word_vec_mat = torch.from_numpy(word_vec_mat).float()
        self.word_embedding = nn.Embedding(word_vec_mat.shape[0], word_vec_mat.shape[1]).to(device)
        self.word_embedding.weight.data.copy_(word_vec_mat)
        self.word_embedding.weight.requires_grad = tune_embedding

        # Position Embedding
        self.dist_embedding = nn.Embedding(100, pos_embedding_dim, padding_idx=0).to(device)

    def create_mask(self, length, max_len=31, dtype=torch.float32):
        """length: B x N x K
            return B x N x K x max_len.
            If max_len is None, then max of length will be used.
            """
        length = length.view(-1)
        mask = torch.arange(max_len, device=length.device,
                            dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
        if dtype is not None:
            mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
        return mask.to(self.device)

    def forward(self, inputs, concat=True):
        word = inputs['indices'].to(self.device)
        dist = inputs['dist'].to(self.device)

        mask = self.create_mask(inputs['length']).view(word.shape)  # B x N x (K|Q) x max_len
        mask = mask.unsqueeze(dim=len(word.shape))

        word = self.word_embedding(word)
        dist = self.dist_embedding(dist)

        x = torch.cat([word, dist], dim=-1)  # B*N*(K|Q) x max_len x size
        # print('| Embedding > x: ', tuple(x.shape))
        x = x * mask
        return x

    def clipEmbedding(self):
        self.dist_embedding.weight.data = clipTwoDimentions(self.pos_embedding.weight.data, norm=self.norm_lim)
