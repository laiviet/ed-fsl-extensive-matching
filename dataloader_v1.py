import collections
import random
import numpy as np
import torch


def merge(d1, d2):
    for id, sample in d1.items():
        sample.update(d2[id])
    return d1


def load_ace_dataset(options):
    import utils
    test_type = options.test_type

    word_data = utils.read_pickle('files/{}/word.proc'.format(options.dataset))
    label2index = utils.read_pickle('files/{}/label2index.proc'.format(options.dataset))

    if options.encoder == 'gcn':
        matrix_data = utils.read_pickle('files/{}/matrix.proc'.format(options.dataset))
        word_data = merge(matrix_data, word_data)

    print(word_data['nw/timex2norm/AFP_ENG_20030327.0022-29'].keys())
    data = [x for id, x in word_data.items() if x['label'] != 'Other']
    other = [x for id, x in word_data.items() if x['label'] == 'Other']

    # Filter test from train:
    train = [x for x in data if not x['label'].startswith(test_type)]
    rest = [x for x in data if x['label'].startswith(test_type)]
    valid = []
    test = []

    # For train
    for label, idx in label2index.items():
        samples = [x for x in train if x['label'] == label]
        if len(samples) > 0:
            for x in range(30 // (len(samples))):
                train += samples

    # For dev and test
    counter = collections.Counter()
    counter.update([x['label'] for x in rest])
    accepted_target_classes = [k for k, v in counter.items() if v > 20]

    for t in accepted_target_classes:
        samples = [x for x in rest if x['label'] == t]
        valid += samples[:len(samples) // 2]
        test += samples[len(samples) // 2:]

    # For other
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
    utils.print_label_distribution(train, valid, test)

    l = len(other) // 3
    train_other = other[:l]
    valid_other = other[l:2 * l]
    test_other = other[2 * l:]

    return train, valid, test, train_other, valid_other, test_other


DEFAULT_FEATURES = ('indices',
                    'dist',
                    'length',
                    'mask',
                    'anchor_index')
GCN_FEATURES = ('indices',
                'dist',
                'length',
                'mask',
                'anchor_index',
                'adj_arc_in',
                'adj_lab_in',
                # 'adj_arc_out',
                # 'adj_lab_out',
                'adj_mask_in',
                'adj_mask_out',
                'adj_mask_loop',
                'mner'
                )


class Fewshot(object):

    def __init__(self, positive_data, negative_data, features=DEFAULT_FEATURES, N=5, K=5, Q=4, O=0, noise=0.0):
        self.features = features
        self.positive_length = len(positive_data)
        self.negative_length = len(negative_data)
        self.max_length = 31
        self.positive_data = positive_data
        self.negative_data = negative_data
        self.noise = noise

        self.N = N
        self.K = K
        self.Q = Q
        self.O = O
        self.event2indices = {'other': [x for x in range(self.negative_length)]}
        self.positive_class = {x['label'] for x in positive_data}
        for t in self.positive_class:
            indices = [idx for idx, x in enumerate(positive_data) if x['label'] == t]
            self.event2indices[t] = indices
        print('Positive_data: ', len(positive_data))
        print('negative_data: ', len(negative_data))

    def __len__(self):
        return 10000000

    def pack(self, items):
        data = {}
        for k in self.features:
            data[k] = [x[k] for x in items]
        return data

    def get_positive(self, scope):
        """

        :param scope: indices of sample of the same class
        :return:
        """
        indices = random.sample(scope, self.K + self.Q)
        # print(indices)

        items = [self.positive_data[i] for i in indices]
        support = self.pack(items[:self.K])
        query = self.pack(items[self.K:])

        # print(query['dist'])
        # exit(0)
        return support, query

    def get_negative(self):
        O, K = self.O, self.K
        scope = self.event2indices['other']
        indices = random.sample(scope, O * K)

        data = []
        for i in range(O):
            _indices = indices[i * K:(i + 1) * K]
            items = [self.negative_data[j] for j in _indices]
            _data = self.pack(items)
            data.append(_data)

        return data

    def __getitem__(self, idx):
        N, K, Q = self.N, self.K, self.Q
        target_classes = random.sample(self.positive_class, N)
        noise_classes = []
        for class_name in self.event2indices.keys():
            if not (class_name in target_classes):
                noise_classes.append(class_name)

        support_set = []
        query_set = []
        query_label = []

        for i, class_name in enumerate(target_classes):  # N way
            scope = self.event2indices[class_name]
            support, query = self.get_positive(scope)
            support_set.append(support)
            query_set.append(query)

            query_label.append([i] * Q)

        other_set = self.get_negative()

        return self.pack(support_set), self.pack(query_set), self.pack(other_set), query_label


FEATURE_TYPES = {
    'target': torch.LongTensor,
    'indices': torch.LongTensor,
    'dist': torch.LongTensor,
    'length': torch.LongTensor,
    'mask': torch.FloatTensor,
    'anchor_index': torch.LongTensor,
    'adj_arc_in': torch.LongTensor,
    'adj_arc_out': torch.LongTensor,
    'adj_lab_in': torch.LongTensor,
    'adj_lab_out': torch.LongTensor,
    'adj_mask_in': torch.FloatTensor,
    'adj_mask_out': torch.FloatTensor,
    'adj_mask_loop': torch.FloatTensor,
    'adj_rel_fet': torch.FloatTensor,
    'mner': torch.LongTensor

}


def fewshot_negative_fn(items):
    feature_names = items[0][0].keys()
    positive = {k: [] for k in feature_names}
    query = {k: [] for k in feature_names}
    negative = {k: [] for k in feature_names}
    label = []
    for s, q, o, l in items:
        for k in feature_names:
            positive[k].append(s[k])
            query[k].append(q[k])
            negative[k].append(o[k])
            # print(len(s[k]), len(q[k]), len(o[k]))
        label.append(l)
    positive_ts = {k: FEATURE_TYPES[k](positive[k]) for k in feature_names}
    query_ts = {k: FEATURE_TYPES[k](query[k]) for k in feature_names}
    negative_ts = {k: FEATURE_TYPES[k](negative[k]) for k in feature_names}

    label_ts = torch.LongTensor(label)

    return positive_ts, query_ts, negative_ts, label_ts
