#from common.config import global_args
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = str(global_args.gpu_id)


import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os.path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import argparse

import torchvision
from torchvision import datasets, transforms

import os.path
from collections import OrderedDict
from common.buffer import Buffer
from polar_express import polar_express_
import random
import pdb
import argparse,time
import math
from copy import deepcopy
import time

from common.utils import create_log_dir

def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))

class AlexNet(nn.Module):
    def __init__(self, taskcla, args=None, device=None):
        super(AlexNet, self).__init__()
        self.act = OrderedDict()
        self.map = []
        self.ksize = []
        self.in_channel = []
        self.map.append(32)
        self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.ksize.append(4)
        self.in_channel.append(3)
        self.map.append(s)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.ksize.append(3)
        self.in_channel.append(64)
        self.map.append(s)
        self.conv3 = nn.Conv2d(128, 256, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(256, track_running_stats=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.ksize.append(2)
        self.in_channel.append(128)
        self.map.append(256 * self.smid * self.smid)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = nn.Linear(256 * self.smid * self.smid, 2048, bias=False)
        self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.fc2 = nn.Linear(2048, 2048, bias=False)
        self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.map.extend([2048])

        self.taskcla = taskcla
        self.fc3 = torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.fc3.append(torch.nn.Linear(2048, n, bias=False))

        self.args = args
        self.buffer = Buffer(args.buffer_size, device)

    def forward(self, x):
        bsz = deepcopy(x.size(0))
        self.act['conv1'] = x
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

        self.act['conv2'] = x
        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

        self.act['conv3'] = x
        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

        x = x.view(bsz, -1)
        self.act['fc1'] = x
        x = self.fc1(x)
        x = self.drop2(self.relu(self.bn4(x)))

        self.act['fc2'] = x
        x = self.fc2(x)
        x = self.drop2(self.relu(self.bn5(x)))

        y = []
        for t, i in self.taskcla:
            y.append(self.fc3[t](x))

        return y


def get_model(model):
    return deepcopy(model.state_dict())


def set_model_(model, state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer.param_groups:
        if epoch == 1:
            param_group['lr'] = args.lr
        else:
            param_group['lr'] /= args.lr_factor

def train(args, model, device, x, y, optimizer, criterion, task_id, epoch):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    x=x.to(device)
    y=y.to(device)
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r):
            b=r[i:i+args.batch_size_train]
        else:
            b=r[i:]
        data_x = x[b]
        data_y = y[b]
        inputs, labels = data_x.to(device), data_y.to(device)

        data_task_label = (torch.ones(len(b))*task_id).long()
        task_ids = data_task_label.to(device)

        output = model(inputs)
        indexs = torch.LongTensor(np.arange(inputs.size(0))).to(device)
        pred = torch.stack(output).transpose(0, 1)[indexs, task_ids, :]
        loss = criterion(pred, labels)

        # ER
        if not model.buffer.is_empty():
            buf_inputs, buf_labels, buf_task_labels = model.buffer.get_data(args.batch_size_train)
            output = model(buf_inputs)
            buf_indexs = torch.LongTensor(np.arange(buf_inputs.size(0))).to(device)
            buf_pred = torch.stack(output).transpose(0, 1)[buf_indexs, buf_task_labels, :]
            loss += criterion(buf_pred, buf_labels)

        optimizer.zero_grad()
        loss.backward()

        if args.use_papo == 'True':
            if epoch % args.interval == 0:
                for k, (m, params) in enumerate(model.named_parameters()):
                    if len(params.size()) == 4 and 'weight' in m:
                        sz = params.grad.data.size(0)
                        flat_grad = params.grad.data.view(sz, -1)

                        papo_grad = polar_express_(args, flat_grad)
                        final_grad = flat_grad + args.Lambda * papo_grad
                        params.grad.data = final_grad.view(params.size())

        optimizer.step()

    # model.buffer.add_data(examples=data_x, labels=data_y, task_labels=data_task_label)

def test(args, model, device, x, y, criterion, task_id):
    model.eval()
    total_loss = 0
    total_num = 0
    correct = 0
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        # Loop batches
        for i in range(0,len(r),args.batch_size_test):
            if i+args.batch_size_test<=len(r): b=r[i:i+args.batch_size_test]
            else: b=r[i:]
            data = x[b]
            data, target = data.to(device), y[b].to(device)
            output = model(data)
            loss = criterion(output[task_id], target)
            pred = output[task_id].argmax(dim=1, keepdim=True)

            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num  += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc

def update_Memory(args, model, x, y, device, task_id, max_update):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    x = x.to(device)
    y = y.to(device)
    num = 0
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data_x = x[b]
        data_y = y[b]
        data_task_label = (torch.ones(len(b))*task_id).long()

        model.buffer.add_data(examples=data_x, labels=data_y, task_labels=data_task_label)

        num += args.batch_size_train
        '''if num > max_update: #max_update:, 2000
            break'''

def main(args, str_time):
    tstart=time.time()
    ## Device Setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ## Load CIFAR100 DATASET
    from dataloader import cifar100 as cf100
    data, taskcla, inputsize = cf100.get(seed=args.seed, pc_valid=args.pc_valid)

    acc_matrix = np.zeros((10, 10))
    criterion = torch.nn.CrossEntropyLoss()

    model = AlexNet(taskcla, args, device).to(device)

    task_id = 0
    task_list = []

    for k, ncla in taskcla:
        lr = args.lr
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

        log.info('*' * 100)
        log.info('Task {:2d} ({:s})'.format(k, data[k]['name']))
        log.info('*' * 100)

        xtrain = data[k]['train']['x']
        ytrain = data[k]['train']['y']
        xvalid = data[k]['valid']['x']
        yvalid = data[k]['valid']['y']
        xtest = data[k]['test']['x']
        ytest = data[k]['test']['y']
        task_list.append(k)

        best_loss = np.inf
        best_model = get_model(model)

        for epoch in range(1, args.n_epochs + 1):
            # Train
            clock0 = time.time()
            train(args, model, device, xtrain, ytrain, optimizer, criterion, k, epoch)
            clock1 = time.time()
            tr_loss, tr_acc = test(args, model, device, xtrain, ytrain, criterion, k)
            log.info('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,
                                                                                                tr_loss,
                                                                                                tr_acc,
                                                                                                1000 * (
                                                                                                            clock1 - clock0)))
            # Validate
            valid_loss, valid_acc = test(args, model, device, xvalid, yvalid, criterion, k)
            log.info(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc))
            # Adapt lr
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = get_model(model)
                patience = args.lr_patience
            else:
                patience -= 1
                if patience <= 0:
                    lr /= args.lr_factor
                    log.info(' lr={:.1e}'.format(lr))
                    if lr < args.lr_min:
                        break
                    patience = args.lr_patience
                    adjust_learning_rate(optimizer, epoch, args)

        set_model_(model, best_model)
        max_update = int(args.buffer_size / 20)
        if max_update != 0:
            update_Memory(args, model, xtrain, ytrain, device, task_id, max_update)

        # Test
        log.info('-' * 40)
        test_loss, test_acc = test(args, model, device, xtest, ytest, criterion, k)
        log.info('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss, test_acc))

        # save model
        if not os.path.exists(args.savename + '/' + str_time):
            os.makedirs(args.savename + '/' + str_time)
        torch.save(model.state_dict(),
                   args.savename + '/' + str_time + '/model_random_' + str(args.seed) + '_task_' + str(
                       task_id) + '.pkl')

        # save accuracy
        jj = 0
        for ii in np.array(task_list)[0:task_id + 1]:
            xtest = data[ii]['test']['x']
            ytest = data[ii]['test']['y']
            _, acc_matrix[task_id, jj] = test(args, model, device, xtest, ytest, criterion, ii)
            jj += 1
        log.info('Accuracies =')
        for i_a in range(task_id + 1):
            # log.info('\t')
            acc_ = ''
            for j_a in range(acc_matrix.shape[1]):
                acc_ += '{:5.1f}% '.format(acc_matrix[i_a, j_a])
            log.info(acc_)
        # update task id
        task_id += 1
    log.info('-' * 50)
    # Simulation Results
    log.info('Task Order : {}'.format(np.array(task_list)))
    log.info('Average Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean()))
    bwt = np.mean((acc_matrix[-1] - np.diag(acc_matrix))[:-1])
    log.info('Backward transfer: {:5.2f}%'.format(bwt))
    log.info('[Elapsed time = {:.1f} ms]'.format((time.time() - tstart) * 1000))
    log.info('-' * 50)

    array = acc_matrix
    df_cm = pd.DataFrame(array, index=[i for i in ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]],
                         columns=[i for i in ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]])
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})
    plt.savefig(args.savename + '/' + str_time + '/' + str(args.seed) + '.pdf')
    plt.show()

    return acc_matrix[-1].mean(), bwt

if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='CIFAR-100 with ER')
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=100, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed')
    parser.add_argument('--pc_valid', default=0.05, type=float,
                        help='fraction of training data used for validation')
    parser.add_argument('--buffer_size', type=int, default=2000,
                        help='number of buffer_size (default: 2000)')
    #PAPO parameters
    parser.add_argument('--use_papo', type=str, default='True',
                        help='True or False')
    parser.add_argument('--iteration_step', type=int, default=5,
                        help='iteration step of polar express')
    parser.add_argument('--Lambda', default=0.5, type=float,
                        help='coefficient lambda')
    parser.add_argument('--interval', type=int, default=5,
                        help='application interval of PAPO')
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--lr_min', type=float, default=1e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')
    parser.add_argument('--savename', type=str, default='./log/CIFAR100/ER/',
                        help='save path')

    args = parser.parse_args()

    str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    log = create_log_dir(args.savename, 'log_{}.txt'.format(str_time_))

    log.info('=' * 100)
    log.info('Arguments =')
    log.info(str(args))
    log.info('=' * 100)

    accs, bwts = [], []
    for seed_ in [1, 2, 3, 4, 5]:
        try:
            args.seed = seed_
            str_time = str_time_

            log.info('=' * 100)
            log.info('Arguments =')
            log.info(str(args))
            log.info('=' * 100)

            acc, bwt = main(args, str_time_)
            accs.append(acc)
            bwts.append(bwt)
        except:
            print("seed " + str(seed_) + "Error!!")

    log.info('Accuracy: ' + str(accs))
    log.info('Backward transfer: ' + str(bwts))
    log.info('Final Avg Accuracy: {:5.2f}%, std:{:5.2f}'.format(np.mean(accs), np.std(accs)))
    log.info('Final Avg Backward transfer: {:5.2f}%, std:{:5.2f}'.format(np.mean(bwts), np.std(bwts)))