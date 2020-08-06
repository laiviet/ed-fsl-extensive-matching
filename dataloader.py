import collections
import random
import numpy as np
import torch


class ContextFreeTokenizer():

    def __init__(self, mapping):
        self.mapping = mapping
        self.pad = mapping['<pad>']
        self.unk = mapping['<unk>']

    def __call__(self, item):
        tokens = item['token']
        anchor_index = item['anchor_index']

        # Crop
        start_crop = 0
        end_crop = len(tokens)
        if end_crop - anchor_index > 15:
            end_crop = anchor_index + 16

        if anchor_index > 15:
            start_crop = anchor_index - 15
            anchor_index = 15
        crop_tokens = tokens[start_crop:end_crop]

        indices = []
        for token in crop_tokens:
            if token in self.mapping:
                indices.append(self.mapping[token])
            else:
                indices.append(self.unk)

        l = len(crop_tokens)
        # Make position embedding
        dist = [0 for x in range(l)]
        for i in range(anchor_index):
            dist[i] = i - anchor_index + 15
        for i in range(anchor_index + 1, l):
            dist[i] = i - anchor_index + 15

        item['length'] = len(indices)
        item['indices'] = indices + [self.pad for _ in range(31 - len(indices))]
        item['anchor_index'] = anchor_index
        item['dist'] = dist + [self.pad for _ in range(31 - len(dist))]
        return item


def load_mapping_vector_tokenizer(embedding):
    import utils
    assert embedding in ['word2vec', 'glove', 'debug']
    if embedding == 'debug':
        mapping, vectors = utils.load_text_vec(utils.GLOVE50)
    if embedding == 'word2vec':
        mapping, vectors = utils.load_text_vec(utils.WORD2VEC)
    if embedding == 'glove':
        mapping, vectors = utils.load_text_vec(utils.GLOVE)
    tokenizer = ContextFreeTokenizer(mapping)
    return mapping, vectors, tokenizer


def load_ace_dataset(options):
    import utils
    test_type = options.test_type

    data, label2idx = utils.read_ace_data(utils.ACE)
    train = data['train']
    valid = data['dev']
    test = data['test']
    other = data['other']

    data = train + valid + test

    # Filter test from train:
    train = [x for x in data if not x['label'].startswith(test_type)]
    rest = [x for x in data if x['label'].startswith(test_type)]
    valid = []
    test = []

    for label, idx in label2idx.items():
        samples = [x for x in train if x['target'] == idx]
        if len(samples) > 0:
            for x in range(30 // (len(samples))):
                train += samples

    counter = collections.Counter()
    counter.update([x['target'] for x in rest])
    accepted_target_classes = [k for k, v in counter.items() if v > 20]
    print(accepted_target_classes)

    for t in accepted_target_classes:
        samples = [x for x in rest if x['target'] == t]
        valid += samples[:len(samples) // 2]
        test += samples[len(samples) // 2:]

    l = len(other) // 3
    train_other = other[:l]
    valid_other = other[l:2 * l]
    test_other = other[2 * l:]

    return train, valid, test, train_other, valid_other, test_other


def load_tac_dataset(options):
    import utils
    test_type = options.test_type
    data, label2idx = utils.read_tac_from_pickle()

    other = data['other']
    train = data['train']
    test = data['test']

    _data = train + test

    # Filter test from train:
    train = [x for x in _data if not x['label'].startswith(test_type)]
    rest = [x for x in _data if x['label'].startswith(test_type)]
    valid = []
    test = []

    for label, idx in label2idx.items():
        samples = [x for x in train if x['target'] == idx]
        if len(samples) > 0:
            for x in range(30 // (len(samples))):
                train += samples

    counter = collections.Counter()
    counter.update([x['target'] for x in rest])
    accepted_target_classes = [k for k, v in counter.items() if v > 20]
    # print(accepted_target_classes)

    for t in accepted_target_classes:
        samples = [x for x in rest if x['target'] == t]
        valid += samples[:len(samples) // 2]
        test += samples[len(samples) // 2:]
    # utils.print_label_distribution(train, valid, test)

    l = len(other) // 3
    train_other = other[:l]
    valid_other = other[l:2 * l]
    test_other = other[2 * l:]

    return train, valid, test, train_other, valid_other, test_other


class Fewshot(object):

    def __init__(self, data, tokenizer, N=5, K=5, Q=4, O=0, other=None, noise=0.0):
        self.positive_length = len(data)
        self.negative_length = len(data)
        self.max_length = 31
        self.tokenizer = tokenizer
        self.data = data
        self.other = other
        self.noise = noise

        self.N = N
        self.K = K
        self.Q = Q
        self.O = O

        self.data_word2vec = [None for _ in range(self.positive_length)]
        self.data_dist = [None for _ in range(self.positive_length)]
        self.data_length = [None for _ in range(self.positive_length)]
        self.data_anchor_index = [None for _ in range(self.positive_length)]
        self.data_mask = [None for _ in range(self.positive_length)]

        self.other_word2vec = [None for _ in range(self.negative_length)]
        self.other_dist = [None for _ in range(self.negative_length)]
        self.other_length = [None for _ in range(self.negative_length)]
        self.other_anchor_index = [None for _ in range(self.negative_length)]
        self.other_mask = [None for _ in range(self.negative_length)]

        self.data_cached = set()
        self.other_cached = set()

        self.event2indices = {'other': [x for x in range(self.negative_length)]}
        targets = {x['target'] for x in data}
        for t in targets:
            indices = [idx for idx, x in enumerate(data) if x['target'] == t]
            self.event2indices[t] = indices

    def __len__(self):
        return 10000000

    def get_positive(self, indices):
        word2vec, dist, length, anchor_index, mask = [], [], [], [], []
        for idx in indices:
            if idx in self.data_cached:
                word2vec.append(self.data_word2vec[idx])
                dist.append(self.data_dist[idx])
                length.append(self.data_length[idx])
                anchor_index.append(self.data_anchor_index[idx])
                mask.append(self.data_mask[idx])
            else:
                item = self.tokenizer(self.data[idx])
                word2vec.append(item['indices'])
                self.data_word2vec[idx] = item['indices']

                dist.append(item['dist'])
                length.append(item['length'])
                anchor_index.append(item['anchor_index'])
                l = item['length']
                m = [1.0 for _ in range(l)] + [0.0 for _ in range(31 - l)]
                mask.append(m)

                self.data_dist[idx] = item['dist']
                self.data_length[idx] = item['length']
                self.data_anchor_index[idx] = item['anchor_index']
                self.data_mask[idx] = m
                self.data_cached.add(idx)
        return word2vec, dist, length, anchor_index, mask

    def get_negative(self):
        scope = self.event2indices['other']
        indices = random.sample(scope, self.O)
        word2vec, dist, length, anchor_index, mask = [], [], [], [], []
        for idx in indices:
            if idx in self.other_cached:
                word2vec.append(self.other_word2vec[idx])
                dist.append(self.other_dist[idx])
                length.append(self.other_length[idx])
                anchor_index.append(self.other_anchor_index[idx])
                mask.append(self.other_mask[idx])
            else:
                item = self.tokenizer(self.other[idx])
                word2vec.append(item['indices'])
                self.other_word2vec[idx] = item['indices']

                dist.append(item['dist'])
                length.append(item['length'])
                anchor_index.append(item['anchor_index'])
                l = item['length']
                m = [1.0 for _ in range(l)] + [0.0 for _ in range(31 - l)]
                mask.append(m)
                self.other_dist[idx] = item['dist']
                self.other_length[idx] = item['length']
                self.other_anchor_index[idx] = item['anchor_index']
                self.other_mask[idx] = m
                self.other_cached.add(idx)
        negative = {'word2vec': word2vec, 'dist': dist, 'length': length, 'anchor_index': anchor_index, 'mask': mask}

        return negative

    def __getitem__(self, item):
        N, K, Q = self.N, self.K, self.Q
        target_classes = random.sample(self.event2indices.keys(), N)
        noise_classes = []
        for class_name in self.event2indices.keys():
            if not (class_name in target_classes):
                noise_classes.append(class_name)
        support_set = {'word2vec': [], 'dist': [], 'length': [], 'anchor_index': [], 'mask': []}
        query_set = {'word2vec': [], 'dist': [], 'length': [], 'anchor_index': [], 'mask': []}
        query_label = []

        for i, class_name in enumerate(target_classes):  # N way
            scope = self.event2indices[class_name]
            indices = random.sample(scope, K + Q)
            word2vec, dist, length, anchor_index, mask = self.get_positive(indices)
            support_word2vec, query_word2vec = word2vec[:K], word2vec[K:]
            support_dist, query_dist = dist[:K], dist[K:]
            support_length, query_length = length[:K], length[K:]
            support_anchor_index, query_anchor_index = anchor_index[:K], anchor_index[K:]
            support_mask, query_mask = mask[:K], mask[K:]

            if self.noise > 0.0:
                for j in range(K):
                    prob = np.random.rand()
                    if prob < self.noise:
                        noise_class_name = noise_classes[np.random.randint(0, len(noise_classes))]
                        scope = self.event2indices[noise_class_name]
                        indices = random.sample(scope, 1)
                        word2vec, dist, length, anchor_index, mask = self.get_positive(indices)
                        support_word2vec[j] = word2vec[0]
                        support_dist[j] = dist[0]
                        support_length[j] = length[0]
                        support_anchor_index[j] = anchor_index[0]
                        support_mask[j] = mask[0]
            support_set['word2vec'].append(support_word2vec)
            support_set['dist'].append(support_dist)
            support_set['length'].append(support_length)
            support_set['anchor_index'].append(support_anchor_index)
            support_set['mask'].append(support_mask)

            query_set['word2vec'].append(query_word2vec)
            query_set['dist'].append(query_dist)
            query_set['length'].append(query_length)
            query_set['anchor_index'].append(query_anchor_index)
            query_set['mask'].append(query_mask)

            query_label += [i] * Q

        negative = self.get_negative() if self.O > 0 else None

        return support_set, query_set, negative, query_label


def fewshot_fn(items):
    support = {'word2vec': [], 'dist': [], 'length': [], 'anchor_index': [], 'mask': []}
    query = {'word2vec': [], 'dist': [], 'length': [], 'anchor_index': [], 'mask': []}
    label = []
    for current_support, current_query, negative, current_label in items:
        support['word2vec'].append(current_support['word2vec'])
        support['dist'].append(current_support['dist'])
        support['length'].append(current_support['length'])
        support['anchor_index'].append(current_support['anchor_index'])
        support['mask'].append(current_support['mask'])

        query['word2vec'].append(current_query['word2vec'])
        query['dist'].append(current_query['dist'])
        query['length'].append(current_query['length'])
        query['anchor_index'].append(current_query['anchor_index'])
        query['mask'].append(current_query['mask'])
        label.append(current_label)

    support_ts = {'word2vec': torch.Tensor(support['word2vec']).long(),
                  'dist': torch.Tensor(support['dist']).long(),
                  'length': torch.Tensor(support['length']).long(),
                  'mask': torch.Tensor(support['mask']).float(),
                  'anchor_index': torch.Tensor(support['anchor_index']).long()}

    query_ts = {'word2vec': torch.Tensor(query['word2vec']).long(),
                'dist': torch.Tensor(query['dist']).long(),
                'length': torch.Tensor(query['length']).long(),
                'mask': torch.Tensor(query['mask']).float(),
                'anchor_index': torch.Tensor(query['anchor_index']).long()}

    label_ts = torch.Tensor(label).long()

    return support_ts, query_ts, None, label_ts


def fewshot_negative_fn(items):
    support = {'word2vec': [], 'dist': [], 'length': [], 'anchor_index': [], 'mask': []}
    query = {'word2vec': [], 'dist': [], 'length': [], 'anchor_index': [], 'mask': []}
    negative = {'word2vec': [], 'dist': [], 'length': [], 'anchor_index': [], 'mask': []}
    label = []
    for s, q, o, l in items:
        support['word2vec'].append(s['word2vec'])
        support['dist'].append(s['dist'])
        support['length'].append(s['length'])
        support['anchor_index'].append(s['anchor_index'])
        support['mask'].append(s['mask'])

        query['word2vec'].append(q['word2vec'])
        query['dist'].append(q['dist'])
        query['length'].append(q['length'])
        query['anchor_index'].append(q['anchor_index'])
        query['mask'].append(q['mask'])

        negative['word2vec'].append(o['word2vec'])
        negative['dist'].append(o['dist'])
        negative['length'].append(o['length'])
        negative['anchor_index'].append(o['anchor_index'])
        negative['mask'].append(o['mask'])

        label.append(l)

    support_ts = {'word2vec': torch.Tensor(support['word2vec']).long(),
                  'dist': torch.Tensor(support['dist']).long(),
                  'length': torch.Tensor(support['length']).long(),
                  'mask': torch.Tensor(support['mask']).float(),
                  'anchor_index': torch.Tensor(support['anchor_index']).long()}

    query_ts = {'word2vec': torch.Tensor(query['word2vec']).long(),
                'dist': torch.Tensor(query['dist']).long(),
                'length': torch.Tensor(query['length']).long(),
                'mask': torch.Tensor(query['mask']).float(),
                'anchor_index': torch.Tensor(query['anchor_index']).long()}

    negative_ts = {'word2vec': torch.Tensor(negative['word2vec']).long(),
                   'dist': torch.Tensor(negative['dist']).long(),
                   'length': torch.Tensor(negative['length']).long(),
                   'mask': torch.Tensor(negative['mask']).float(),
                   'anchor_index': torch.Tensor(negative['anchor_index']).long()}

    label_ts = torch.Tensor(label).long()

    return support_ts, query_ts, negative_ts, label_ts
