# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
import torch
import random
import argparse
import os
import warnings
warnings.filterwarnings("ignore")
from utils import process
from utils import aug
from modules.GIN import GIN
from net.IMCSNet import IMCSNet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--data', type=str, default='cornell')
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--eval_every', type=int, default=10)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--sample_size', type=int, default=2000)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--sparse', type=str_to_bool, default=True)

parser.add_argument('--input_dim', type=int, default=1703)
parser.add_argument('--gnn_dim', type=int, default=512)
parser.add_argument('--proj_dim', type=int, default=512)
parser.add_argument('--proj_hid', type=int, default=4096)
parser.add_argument('--pred_dim', type=int, default=512)
parser.add_argument('--pred_hid', type=int, default=4096)
parser.add_argument('--momentum', type=float, default=0.8)
parser.add_argument('--beta', type=float, default=0.95)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--gamma', type=float, default=0.05)
parser.add_argument('--drop_edge', type=float, default=0.25)
parser.add_argument('--drop_feat1', type=float, default=0.15)
parser.add_argument('--drop_feat2', type=float, default=0.15)
parser.add_argument('--GIN', type=bool, default=True)
parser.add_argument('--SIMGCL_aug_feature_dropout', type=bool, default=True)
args = parser.parse_args()
torch.set_num_threads(4)


def evaluation(args, adj, diff, feat, gnn, idx_train, idx_test, sparse):
    clf = LogisticRegression(random_state=0, max_iter=2000)
    if not args.GIN:
        model = GIN(input_size, gnn_output_size)  # 1-layer
    else:
        model = GIN(input_size, gnn_output_size)  # 1-layer
    model.load_state_dict(gnn.state_dict())
    with torch.no_grad():
        embeds1 = model(feat, adj, sparse)
        embeds2 = model(feat, diff, sparse)
        if not args.GIN:
            train_embs = embeds1[0, idx_train] + embeds2[0, idx_train]
            test_embs = embeds1[0, idx_test] + embeds2[0, idx_test]
        else:
            train_embs = embeds1[idx_train] + embeds2[idx_train]
            test_embs = embeds1[idx_test] + embeds2[idx_test]

        train_labels = torch.argmax(labels[0, idx_train], dim=1)
        test_labels = torch.argmax(labels[0, idx_test], dim=1)
    clf.fit(train_embs, train_labels)
    pred_test_labels = clf.predict(test_embs)
    #print(pred_test_labels)
    #print(test_embs)

    return accuracy_score(test_labels, pred_test_labels)


def train_and_evaluate(features, dict):
    lr, weight_decay, beta, momentum, alpha, gamma, drop_edge, drop_feat1, drop_feat2, tau= dict['learning_rate'], dict['weight_decay'], dict['beta'],dict['momentum'],dict['alpha'],dict['gamma'],dict['drop_edge'],dict['drop_feat1'],dict['drop_feat2'],dict['tau']


    # Initiate models
    if not args.GIN:
        model = GIN(input_size, gnn_output_size)
    else:
        model = GIN(input_size, gnn_output_size)       # fxme here
    imcsn = IMCSNet(gnn=model,
                  feat_size=input_size,
                  projection_size=projection_size,
                  projection_hidden_size=projection_hidden_size,
                  prediction_size=prediction_size,
                  prediction_hidden_size=prediction_hidden_size,
                  moving_average_decay=momentum, beta=beta, alpha=alpha).to(device)

    opt = torch.optim.Adam(imcsn.parameters(), lr=lr, weight_decay=weight_decay)

    results = []

    # Training
    best = 0
    patience_count = 0
    for epoch in range(epochs):
        for _ in range(batch_size):
            idx = np.random.randint(0, adj.shape[-1] - sample_size + 1)
            ba = adj[idx: idx + sample_size, idx: idx + sample_size]
            bd = diff[idx: idx + sample_size, idx: idx + sample_size]
            bd = sp.csr_matrix(np.matrix(bd))
            features = features.squeeze(0)
            bf = features[idx: idx + sample_size]

            ori_adj=ba
            aug_adj1 = aug.aug_random_edge(ba, drop_percent=drop_edge_rate_1)
            aug_adj2 = bd

            ori_features=bf

            ori_adj=process.normalize_adj(ori_adj+sp.eye(ori_adj.shape[0]))

            aug_adj1 = process.normalize_adj(aug_adj1 + sp.eye(aug_adj1.shape[0]))
            aug_adj2 = process.normalize_adj(aug_adj2 + sp.eye(aug_adj2.shape[0]))

            ori_features=ori_features.to(device)

            if sparse:
                ori_adj=process.sparse_mx_to_torch_sparse_tensor(ori_adj).to(device)
                adj_1 = process.sparse_mx_to_torch_sparse_tensor(aug_adj1).to(device)
                adj_2 = process.sparse_mx_to_torch_sparse_tensor(aug_adj2).to(device)
            else:
                ori_adj = (ori_adj + sp.eye(ori_adj.shape[0])).todense()
                aug_adj1 = (aug_adj1 + sp.eye(aug_adj1.shape[0])).todense()
                aug_adj2 = (aug_adj2 + sp.eye(aug_adj2.shape[0])).todense()

                ori_adj=torch.FloatTensor(ori_adj[np.newaxis]).to(device)
                adj_1 = torch.FloatTensor(aug_adj1[np.newaxis]).to(device)
                adj_2 = torch.FloatTensor(aug_adj2[np.newaxis]).to(device)

            if not args.SIMGCL_aug_feature_dropout:
                aug_features1 = aug.aug_feature_dropout(bf, drop_percent=drop_feature_rate_1)
                aug_features2 = aug.aug_feature_dropout(bf, drop_percent=drop_feature_rate_2)
            else:
                bf = aug.SIMGCL_aug_feature_dropout(bf, ori_adj)
                aug_features1 = aug.aug_feature_dropout(bf, drop_percent=drop_feature_rate_1)
                aug_features2 = aug.aug_feature_dropout(bf, drop_percent=drop_feature_rate_2)

            aug_features1 = aug_features1.to(device)
            aug_features2 = aug_features2.to(device)

            opt.zero_grad()
            loss = imcsn(ori_adj, adj_1, adj_2, ori_features, aug_features1, aug_features2, sparse)
            loss.backward()
            opt.step()
            imcsn.update_ma()

        if epoch % eval_every_epoch == 0:
            acc = evaluation(args, eval_adj, eval_diff, features, model, idx_train, idx_test, sparse)
            if acc > best:
                best = acc
                patience_count = 0
            else:
                patience_count += 1
            results.append(acc)
            print('\t epoch {:03d} | loss {:.5f} | clf test acc {:.5f}'.format(epoch, loss.item(), acc))
            if patience_count >= patience:
                print('Early Stopping.')
                break
            
    result_over_runs.append(max(results))
    print('\t best acc {:.5f}'.format(max(results)))

    validation_metric = max(results)
    return validation_metric

if __name__ == '__main__':

    param_grid = {
        #'learning_rate': [1e-5,2e-5,3e-5,4e-5,5e-5,6e-5,7e-5,8e-5,9e-5,1e-4,2e-4,3e-4,4e-4,5e-4,6e-4,7e-4,8e-4,9e-4],
        #'weight_decay': [0, 1e-6,2e-6,3e-6,4e-6,5e-6,6e-6,7e-6,8e-6,9e-6,1e-7,3e-7,5e-7,7e-7,9e-7],
        #'beta': [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0,45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
        #'momentum':[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0,45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
        #'alpha':[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0,45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
        #'gamma':[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0,45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
        #'drop_edge':[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0,45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
        #'drop_feat1':[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0,45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
        #'drop_feat2':[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0,45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
        'learning_rate': [1e-2],
        'weight_decay': [0.0],
        'beta': [0.9],
        'momentum': [0.2],
        'alpha': [0.15],
        'gamma': [0.05],
        'drop_edge': [0.4],
        'drop_feat1': [0.35],
        'drop_feat2': [0.35],
        'tau': [0.7]

    }
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    n_runs = args.runs
    eval_every_epoch = args.eval_every

    dataset = args.data
    input_size = args.input_dim

    gnn_output_size = args.gnn_dim
    projection_size = args.proj_dim
    projection_hidden_size = args.proj_hid
    prediction_size = args.pred_dim
    prediction_hidden_size = args.pred_hid
    momentum = args.momentum
    beta = args.beta
    alpha = args.alpha
    gamma = args.gamma

    drop_edge_rate_1 = args.drop_edge
    drop_feature_rate_1 = args.drop_feat1
    drop_feature_rate_2 = args.drop_feat2

    epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay
    sample_size = args.sample_size
    batch_size = args.batch_size
    patience = args.patience

    sparse = args.sparse

    # Loading dataset
    adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if os.path.exists('data/diff_{}_{}.npy'.format(dataset, gamma)):
        diff = np.load('data/diff_{}_{}.npy'.format(dataset, gamma), allow_pickle=True)
    else:
        diff = aug.gdc(adj, gamma=gamma, eps=0.0001)
        np.save('data/diff_{}_{}'.format(dataset, gamma), diff)

    features, _ = process.preprocess_features(features)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = labels.shape[1]

    features = torch.FloatTensor(features[np.newaxis])
    labels = torch.FloatTensor(labels[np.newaxis])

    norm_adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    norm_diff = sp.csr_matrix(diff)
    if sparse:
        eval_adj = process.sparse_mx_to_torch_sparse_tensor(norm_adj)
        eval_diff = process.sparse_mx_to_torch_sparse_tensor(norm_diff)
    else:
        eval_adj = (norm_adj + sp.eye(norm_adj.shape[0])).todense()
        eval_diff = (norm_diff + sp.eye(norm_diff.shape[0])).todense()
        eval_adj = torch.FloatTensor(eval_adj[np.newaxis])
        eval_diff = torch.FloatTensor(eval_diff[np.newaxis])

    result_over_runs = []

    # 3. Grid Search
    best_performance = float('-inf')
    best_params = None
    for params in ParameterGrid(param_grid):
        performance = train_and_evaluate(features, params)
        # 4.best
        if performance > best_performance:
            best_performance = performance
            best_params = params
    print('best_params:{} '.format(best_params))
    print('best acc:{}'.format(best_performance))