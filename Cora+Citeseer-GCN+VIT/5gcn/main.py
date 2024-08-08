import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN
from utils1 import load_data1
import utils

seed = 42
epochs = 200
lr = 0.001
weight_decay = 5e-4
hidden = 128
dropout = 0.5

np.random.seed(seed)
torch.manual_seed(seed)
is_cuda = torch.cuda.is_available()
if is_cuda:
    torch.cuda.manual_seed(seed)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output, feature_matrix = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    model.eval()
    output, _ = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return feature_matrix


def test():
    model.eval()
    output, _ = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


if __name__ == '__main__':
    t_total = time.time()

    # adj, features, tmp_labels, tmp_train, tmp_val, tmp_test = load_data1('citeseer')
    # labels=[]
    # for i in range(tmp_labels.shape[0]):
    #     for j in range(6):
    #         if(tmp_labels[i][j]==1): break
    #     labels.append(j)
    # # labels = torch.LongTensor(np.where(labels)[1])
    # labels = torch.LongTensor(labels)
    # print(labels.shape)
    # adj = utils.sparse_mx_to_torch_sparse_tensor(adj)
    # features = torch.FloatTensor(np.array(features.todense()))
    #
    # idx_train = []
    # idx_val=[]
    # idx_test=[]
    # for i in range(3327):
    #     if tmp_train[i]==True:
    #         idx_train.append(i)
    #     if tmp_val[i]==True:
    #         idx_val.append(i)
    #     if tmp_test[i]==True:
    #         idx_test.append(i)
    # idx_train=torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)
    # print(tmp_train.shape)
    # print(adj.shape, features.shape, labels.shape)
    # print(idx_train.shape,idx_val.shape,idx_test.shape)

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    print(adj.shape, features.shape, labels.shape)
    print(idx_train.shape, idx_val.shape, idx_test.shape)

    model = GCN(nfeat=features.shape[1],
                nhid=hidden,
                nclass=labels.max().item() + 1,
                dropout=dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)
    if is_cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    matrix = []
    for epoch in range(epochs):
        matrix = train(epoch)
    print(matrix.shape)

    layer_diffs = []
    for layer in range(matrix.shape[1]):  # 针对每一层
        layer_features = matrix[:, layer, :]  # 获取当前层的所有节点特征
        diffs = np.linalg.norm(np.diff(layer_features, axis=0), ord=2, axis=1)  # 计算当前层每个节点的特征差异
        mean_diff = np.mean(diffs)  # 计算当前层所有节点特征差异的平均值
        layer_diffs.append(mean_diff)  # 将平均差异值添加到列表中

    print("每一层节点间特征的差异程度：", layer_diffs)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    test()
