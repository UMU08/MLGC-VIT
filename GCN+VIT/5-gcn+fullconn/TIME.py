import torch
import torch.nn as nn
import numpy as np
import config
from dataset import create_graph
from model import JointlyTrainModel
from AutoWeight import AutomaticWeightedLoss
import os
from sklearn.utils import shuffle
import time

path = 'D:/syj/BiDGCN/GMSS/GNN_NPY_DATASETS/SEED/data_dependent/de_LDS_s'

batch_size = config.batch_size
epochs = config.epochs
lr = config.lr
weight_decay = config.weight_decay
device = config.device

DATASETS = ['SEED', 'SEED_IV']
DATASET = path.strip().split('/')[-3]
assert DATASET in DATASETS
DEPENDENT = path.strip().split('/')[-2]
if DEPENDENT == 'data_independent':
    DATASET = DATASET + '_' + DEPENDENT


def writeEachEpoch(people, epoch, batchsize, lr, temperature, acc):
    import model
    log = []
    log.append(f'{DATASET}\t{people}\t{temperature}\t'
               f'{batchsize}\t{epoch}\t{lr}\t{model.drop_rate}\t{acc:.4f}\n')
    with open(
            f'D:/demo/gcn+transformer/5-gcn+全连接/record/{DATASET}_All_log1.txt',
            'a') as f:
        f.writelines(log)


def updatelog(people, epoch, acc):
    log = []
    log.append(f'{DATASET}\t{people}\t{epoch}\t{lr}\t{batch_size}\t{acc:.4f}\n')
    with open(f'D:/demo/gcn+transformer/5-gcn+全连接/record/{DATASET}_UPDATE_LOG1.txt', 'a') as f:
        f.writelines(log)


def test(net, test_data, test_label, people, highest_acc, epoch):
    criterion = nn.CrossEntropyLoss().to(device)

    gloader = create_graph(test_data, test_label, shuffle=True, batch_size=batch_size, drop_last=True)
    net.testmode = True
    net.eval()
    epoch_loss = 0.0
    correct_pred = 0
    for ind, data in enumerate(gloader):
        data = data.to(device)
        out = net(data)
        y = data.y
        _, pre = torch.max(out, dim=1)

        correct_pred += sum([1 for a, b in zip(pre, y) if a == b])
        loss = criterion(out, y)

        epoch_loss += float(loss.item())

    ACC = correct_pred / ((ind + 1) * batch_size)
    if ACC > highest_acc:
        updatelog(people, epoch, ACC)
        highest_acc = ACC
        ck = {}
        ck['epoch'] = epoch
        ck['model'] = net.state_dict()
        ck['ACC'] = ACC

        torch.save(ck, f'{DATASET}_jointly_checkpoint/checkpoint_{people}.pkl')

    net.train()
    net.testmode = False
    return highest_acc, ACC


def train(train_data, train_label, test_data, test_label, people):
    highest_acc = 0.0

    if not os.path.exists(f'{DATASET}_jointly_checkpoint'):
        os.makedirs(f'{DATASET}_jointly_checkpoint')

    if os.path.exists(f'{DATASET}_jointly_checkpoint/checkpoint_{people}.pkl'):
        check = torch.load(f'{DATASET}_jointly_checkpoint/checkpoint_{people}.pkl')
        highest_acc = check['ACC']

    HC = None
    if 'SEED_IV' in DATASET:
        HC = 4
    else:
        HC = 3
    assert HC is not None

    awl = AutomaticWeightedLoss(2)
    net = JointlyTrainModel(5, 32, batch_size, testmode=False, HC=HC)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True,
                                                           threshold=0.0001,
                                                           threshold_mode='rel', cooldown=1, min_lr=0, eps=1e-8)

    gloader = create_graph(train_data, train_label, shuffle=False, batch_size=batch_size)
    for epoch in range(epochs):

        epoch_loss = 0.0
        epoch_loss3 = 0.0
        correct_pred3 = 0

        for ind, gdata in enumerate(gloader):
            gdata = gdata.to(device)
            x3 = net(gdata)
            y3 = gdata.y
            _, pred3 = torch.max(x3, dim=1)
            correct_pred3 += sum([1 for a, b in zip(pred3, y3) if a == b])
            loss3 = criterion(x3, y3)
            loss = loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            epoch_loss3 += float(loss3.item())

        highest_acc, current_acc = test(net, test_data, test_label, people, highest_acc, epoch)
        writeEachEpoch(people, epoch, batch_size, lr, 0.25, current_acc)

        scheduler.step(epoch_loss)

        denominator = (ind + 1) * batch_size
        if epoch % 5 == 0:
            print()
            print(f'-----highest_acc {highest_acc:.4f} current_acc {current_acc:.4f}-----')
            print('Dataset: ', DATASET)
            print(f'batch {batch_size}, lr {lr}')
            print()

        print(f'Epoch [{epoch}/{epochs}] \n'
              f'Loss gLoss[{epoch_loss3 / (ind + 1):.4f}] \n'
              f'ACC@1 gACC[{correct_pred3 / denominator:.4f}] \n')


def runs(people):
    print(f'load object {people}\'s data.....')
    train_data = np.load(path + '/de_LDS_{}'.format(people) + '/train_dataset_{}.npy'.format(people))
    train_label = np.load(path + '/de_LDS_{}'.format(people) + '/train_labelset_{}.npy'.format(people))
    test_data = np.load(path + '/de_LDS_{}'.format(people) + '/test_dataset_{}.npy'.format(people))
    test_label = np.load(path + '/de_LDS_{}'.format(people) + '/test_labelset_{}.npy'.format(people))
    #
    # data = np.concatenate((train_data,test_data))
    # label = np.concatenate((train_label, test_label))
    # print(data.shape,label.shape)
    #
    # data, label = shuffle(data, label, random_state=1337)
    #
    # print('loaded!')
    # train_data = data[:2010]
    # test_data = data[2010:]
    # train_label = label[:2010]
    # test_label = label[2010:]
    train(train_data, train_label, test_data, test_label, people)


if __name__ == '__main__':
    for i in range(1):
        start = time.time()
        runs(i + 1)
        end = time.time()
        print("循环运行时间:%.2f秒" % (end - start))
