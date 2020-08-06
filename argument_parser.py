import argparse


def str_list(text):
    return tuple(text.split(','))


def int_list(text):
    return [int(x) for x in text.split(',')]


def parse_int_list(input_):
    if input_ == None:
        return []
    return list(map(int, input_.split(',')))


def parse_float_list(input_):
    if input_ == None:
        return []
    return list(map(float, input_.split(',')))


def one_or_list(parser):
    def parse_one_or_list(input_):
        output = parser(input_)
        if len(output) == 1:
            return output[0]
        else:
            return output

    return parse_one_or_list


def proto_parser():
    parser = argparse.ArgumentParser()
    # Training setting
    model_choices = ['proto', 'protohatt', 'matching', 'relation']
    encoder_choices = ['cnn', 'gru', 'lstm', 'trans', 'gcn']
    parser.add_argument('--model', default='proto', choices=model_choices)
    parser.add_argument('--encoder', default='gcn', choices=encoder_choices)
    parser.add_argument('-b', '--batch_size', default=4, type=int)
    parser.add_argument('--optimizer', default='sgd', type=str, choices=['adam', 'sgd', 'adadelta'])
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--lr_step_size', default=500, type=int)
    parser.add_argument('--gpu', default='0')
    # Few-shot settings
    parser.add_argument('--dataset', default='ace', choices=['ace', 'tac', 'debug'])
    parser.add_argument('--save', default='checkpoints', type=str)
    parser.add_argument('--seed', default=1234, type=int)

    parser.add_argument('-t', '--train_way', default=20, type=int)
    parser.add_argument('-n', '--way', default=5, type=int)
    parser.add_argument('-k', '--shot', default=5, type=int)
    parser.add_argument('-q', '--query', default=4, type=int)
    parser.add_argument('--noise', default=0.0, type=float)

    # Embedding
    parser.add_argument('--test_type', default='Life,Movement,Personnel,Transaction', type=str_list)
    parser.add_argument('--embedding', default='glove', type=str_list)
    parser.add_argument('--tune_embedding', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--progress', default=False, action='store_true')
    parser.add_argument('--max_length', default=31, type=int)

    # CNN params
    parser.add_argument('--window', default=2, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)


    # CNN, NCNN parameters
    parser.add_argument('--cnn_kernel_sizes', default=[2, 3, 4, 5], type=parse_int_list)
    parser.add_argument('--cnn_kernel_number', default=150, type=int)

    # RNN parameters (i.e, GRU, LSTM)
    parser.add_argument('--rnn_num_hidden_units', default=300, type=int)
    parser.add_argument('--rnn_pooling', default='pool_anchor',
                        choices=['pool_anchor', 'pool_max', 'pool_dynamic', 'pool_entity'])


    # GCN params

    parser.add_argument('--num_rel_dep', default=50, type=int)
    parser.add_argument('--gcnn_kernel_numbers', default=[300, 300], type=parse_int_list)
    parser.add_argument('--gcnn_edge_patterns', default=[1, 0, 1], type=parse_int_list)
    parser.add_argument('--gcnn_pooling', default='pool_anchor',
                        choices=['pool_anchor', 'pool_max', 'pool_dynamic', 'pool_entity'])

    # Transformer model
    parser.add_argument('--trans_n', default=2, type=int)

    # Auxilarity params
    loss_functions = ['euclidean', 'cosine', 'learnable','kl']
    parser.add_argument('--betaf', default='euclidean', type=str, choices=loss_functions)
    parser.add_argument('--gammaf', default='cosine', type=str, choices=loss_functions)
    parser.add_argument('--alpha', default=0.0, type=float)
    parser.add_argument('--beta', default=0.0, type=float)
    parser.add_argument('--gamma', default=0.0, type=float)

    return parser
