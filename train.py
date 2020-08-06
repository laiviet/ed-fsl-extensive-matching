from argument_parser import *
from dataloader_v1 import *
from framework import FewShotREFramework
from models.proto import *
from models.matching import *
from models.relation_network import *
import random
from torch.utils.data import DataLoader
import datetime
import utils

if __name__ == '__main__':

    args = proto_parser().parse_args()
    # print(args)

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    B = args.batch_size  # default = 4
    TN = args.train_way  # default = 20
    N = args.way
    O = 1  # defautl = 5
    K = args.shot  # default = 5
    Q = args.query  # default = 5

    current_time = str(datetime.datetime.now().time())
    args.log_dir = 'logs/{}-{}-way-{}-shot-{}'.format(args.model, args.way, args.shot, current_time)

    if args.noise > 0.0:
        print('Noise level: ', args.noise)
    print('-' * 80)

    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("Model: {}".format(args.model))
    print('alpha: ', args.alpha)
    print('beta: ', args.beta)
    print('gamma: ', args.gamma)
    print('-' * 80)
    for k, v in args.__dict__.items():
        print(k, ': ', v)
    print('-' * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    train_rev, valid_rev, test_rev, train_other, valid_other, test_other = load_ace_dataset(args)
    # else:
    #     train_rev, valid_rev, test_rev, train_other, valid_other, test_other = load_tac_dataset(args)

    if args.encoder == 'gcn':
        feature_set = GCN_FEATURES
    else:
        feature_set = DEFAULT_FEATURES

    print(feature_set)

    train_dataset = Fewshot(train_rev, train_other, feature_set, TN, K, Q, O, noise=args.noise)
    val_dataset = Fewshot(valid_rev, valid_other, feature_set, N, K, Q, O, noise=args.noise)
    test_dataset = Fewshot(test_rev, test_other, feature_set, N, K, Q, O, noise=args.noise)

    args.vectors = utils.read_pickle('files/{}/W.proc'.format(args.dataset))
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=B,
                                   num_workers=4,
                                   collate_fn=fewshot_negative_fn)
    val_data_loader = DataLoader(val_dataset,
                                 batch_size=B,
                                 num_workers=4,
                                 collate_fn=fewshot_negative_fn)
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=B,
                                  num_workers=4,
                                  collate_fn=fewshot_negative_fn)

    if 'protohatt' == args.model:
        model = ProtoHATT(args)
    elif 'relation' == args.model:
        model = RelationNetwork(args)
    elif 'matching' == args.model:
        model = Matching(args)
    elif 'proto' == args.model:
        model = Proto(args)
    else:
        print('Cannot recognize model, exit')
        exit(0)
    model = model.to(device)
    framework = FewShotREFramework(model,
                                   train_data_loader,
                                   val_data_loader,
                                   test_data_loader,
                                   args)
    if args.debug:
        framework.train(train_iter=3050, val_step=200, val_iter=200, test_iter=200)
    else:
        framework.train(train_iter=5050, val_step=1000, test_iter=1000)
