import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import traceback

import os
import os.path
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import random
import pdb
import argparse,time
import math
from copy import deepcopy
import time
from polar_express import polar_express_

## Define MLP model
class MLPNet(nn.Module):
    def __init__(self, n_hidden=100, n_outputs=10, args=None, device=None):
        super(MLPNet, self).__init__()
        self.act=OrderedDict()
        self.lin1 = nn.Linear(784,n_hidden,bias=False)
        self.lin2 = nn.Linear(n_hidden,n_hidden, bias=False)
        self.fc1  = nn.Linear(n_hidden, n_outputs, bias=False)

        self.args = args

    def forward(self, x):
        self.act['Lin1']=x
        x = self.lin1(x)        
        x = F.relu(x)
        self.act['Lin2']=x
        x = self.lin2(x)        
        x = F.relu(x)
        self.act['fc1']=x
        x = self.fc1(x)
        return x 

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def train (args, model, device, x, y, optimizer,criterion, taskid, omega, model_old, epoch, sv_history):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    x = x.to(device)
    y = y.to(device)
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data_x = x[b].view(-1, 28*28)
        data_y = y[b]
        inputs, labels = data_x.to(device), data_y.to(device)
        optimizer.zero_grad()

        output = model(inputs)
        loss = criterion(output, labels)

        if taskid > 0:
            loss_reg = 0
            for (name, param), (_, param_old) in zip(model.named_parameters(), model_old.named_parameters()):
                loss_reg += torch.sum(omega[name] * (param_old - param).pow(2)) / 2
            loss = loss + args.lamb * loss_reg

        loss.backward()
        if taskid == 0:
            Lambda = args.Lambda0
        else:
            Lambda = args.Lambda
        for k, (m,params) in enumerate(model.named_parameters()):
            if args.use_papo == 'True':
                if epoch % args.interval == 0:
                    sz =  params.grad.data.size(0)
                    flat_grad = params.grad.data.view(sz, -1)
                    papo_grad = polar_express_(args, flat_grad)
                    final_grad = flat_grad + Lambda * papo_grad
                    params.grad.data = final_grad.view(params.size())
        optimizer.step()
    return

def test (args, model, device, x, y, criterion):
    model.eval()
    total_loss = 0
    total_num = 0 
    correct = 0
    r=np.arange(x.size(0))
    x = x.to(device)
    y = y.to(device)
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    with torch.no_grad():
        # Loop batches
        for i in range(0,len(r),args.batch_size_test):
            if i+args.batch_size_test<=len(r): b=r[i:i+args.batch_size_test]
            else: b=r[i:]
            data = x[b].view(-1,28*28)
            data, target = data.to(device), y[b].to(device)
            output = model(data)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True) 
            
            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num  += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc


def main(args):
    tstart=time.time()
    ## Device Setting 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ## Load PMNIST DATASET
    from dataloader import pmnist as pmd
    data,taskcla,inputsize=pmd.get(seed=args.seed, pc_valid=args.pc_valid)

    model = MLPNet(args.n_hidden, args.n_outputs, args, device).to(device)
    log.info('Model parameters ---')
    for k_t, (m, param) in enumerate(model.named_parameters()):
        log.info('%d %s %s', k_t, m, param.shape)
    log.info('-' * 40)

    acc_matrix = np.zeros((10, 10))
    criterion = torch.nn.CrossEntropyLoss()

    task_id = 0
    task_list = []
    sv_history = {}

    omega = {}
    model_old = deepcopy(model)

    for n, p in model.named_parameters():
        omega[n] = 0

    for k,ncla in taskcla:
        log.info('*'*100)
        log.info('Task {:2d} ({:s})'.format(k,data[k]['name']))
        log.info('*'*100)
        xtrain=data[k]['train']['x']
        ytrain=data[k]['train']['y']
        xvalid=data[k]['valid']['x']
        yvalid=data[k]['valid']['y']
        xtest =data[k]['test']['x']
        ytest =data[k]['test']['y']
        task_list.append(k)

        lr = args.lr
        log.info ('-'*40)
        log.info ('Task ID :{} | Learning Rate : {}'.format(task_id, lr))
        log.info ('-'*40)

        optimizer = optim.SGD(model.parameters(), lr=args.lr)

        for epoch in range(1, args.n_epochs+1):
            # Train
            clock0=time.time()
            train(args, model, device, xtrain, ytrain, optimizer, criterion, task_id, omega, model_old, epoch, sv_history)
            clock1=time.time()
            tr_loss,tr_acc = test(args, model, device, xtrain, ytrain,  criterion)
            log.info('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch, tr_loss,tr_acc, 1000*(clock1-clock0)))
            # Validate
            valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion)
            log.info(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc))
            log.info('')

        # Test
        log.info ('-'*40)
        test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion)
        log.info('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))

        # save model
        if not os.path.exists(args.savename + '/' + str_time):
            os.makedirs(args.savename + '/' + str_time)
        torch.save(model.state_dict(),
                   args.savename + '/' + str_time + '/model_random_' + str(args.seed) + '_task_' + str(
                       task_id) + '.pkl')

        # save accuracy 
        jj = 0 
        for ii in np.array(task_list)[0:task_id+1]:
            xtest =data[ii]['test']['x']
            ytest =data[ii]['test']['y'] 
            _, acc_matrix[task_id,jj] = test(args, model, device, xtest, ytest,criterion) 
            jj +=1
        log.info('Accuracies =')
        for i_a in range(task_id + 1):
            # log.info('\t')
            acc_ = ''
            for j_a in range(acc_matrix.shape[1]):
                acc_ += '{:5.1f}% '.format(acc_matrix[i_a, j_a])
            log.info(acc_)

        # ------------------ Synaptic intelligence !! ---------------------------
        model_old = deepcopy(model)
        freeze_model(model_old)

        omega = omega_update(omega, model, xtrain, device)

        # update task id
        task_id += 1

    log.info('-'*50)
    # Simulation Results 
    log.info ('Task Order : {}'.format(np.array(task_list)))
    log.info ('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean())) 
    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1]) 
    log.info ('Backward transfer: {:5.2f}%'.format(bwt))
    log.info('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000))
    log.info('-'*50)
    # Plots
    array = acc_matrix
    df_cm = pd.DataFrame(array, index = [i for i in ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10"]],
                      columns = [i for i in ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10"]])
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})

    plt.tight_layout()
    plt.savefig(args.savename + '/' + str_time + '/' + str(args.seed) + '.pdf')
    plt.show()
    # plt.savefig()
    return acc_matrix[-1].mean(), bwt

def omega_update(omega, model, x, device):
    sbatch = 20

    # Compute
    model = deepcopy(model)
    model.train()

    r = np.arange(x.size(0))
    x = x.to(device)
    np.random.shuffle(r)
    r = torch.LongTensor(r).to(device)

    for i in range(0,len(r),args.batch_size_train):
        if i+sbatch<=len(r): b=r[i:i+sbatch]
        else: b=r[i:]
        data_x = x[b].view(-1,28*28)
        inputs = data_x.to(device)

        # Forward and backward
        model.zero_grad()
        outputs = model.forward(inputs)

        # Sum of L2 norm of output scores
        loss = torch.sum(outputs.norm(2, dim=-1))

        loss.backward()

        # Get gradients
        for n, p in model.named_parameters():
            if p.grad is not None:
                omega[n] += p.grad.data.abs() / x.size(0)

    return omega

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='Sequential PMNIST with ER')
    parser.add_argument('--batch_size_train', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=5, metavar='N',
                        help='number of training epochs/task (default: 5)')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 2)')
    parser.add_argument('--pc_valid', default=0.1,type=float,
                        help='fraction of training data used for validation')
    # PAPO parameters
    parser.add_argument('--use_papo', type=str, default='True',
                        help='True or False')
    parser.add_argument('--iteration_step', type=int, default=5,
                        help='iteration step of polar express')
    parser.add_argument('--Lambda0', default=10, type=float,
                        help='coefficient lambda0 (default: 10)')
    parser.add_argument('--Lambda', default=0.001, type=float,
                        help='coefficient lambda (default: 0.001)')
    parser.add_argument('--interval', type=int, default=5, metavar='N',
                        help='application interval of PAPO (default: 5)')
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr_min', type=float, default=1e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')
    # Architecture
    parser.add_argument('--n_hidden', type=int, default=100, metavar='NH',
                        help='number of hidden units in MLP (default: 100)')
    parser.add_argument('--n_outputs', type=int, default=10, metavar='NO',
                        help='number of output units in MLP (default: 10)')
    parser.add_argument('--n_tasks', type=int, default=10, metavar='NT',
                        help='number of tasks (default: 10)')

    parser.add_argument('--savename', type=str, default='./log/PMNIST/MAS/',
                        help='save path')

    parser.add_argument('--lamb', type=float, default=1, metavar='lamb', help='MAS_lamb (default: 1.0)')


    args = parser.parse_args()
    str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    log = create_log_dir(args.savename, 'log_{}.txt'.format(str_time_))

    accs, bwts = [], []
    for seed_ in [1, 2, 3, 4, 5]:
        try:
            args.seed = seed_
            str_time = str_time_

            log.info('='*100)
            log.info('Arguments =')
            log.info(str(args))
            log.info('='*100)

            acc, bwt = main(args)
            accs.append(acc)
            bwts.append(bwt)
        except:
            print("seed " + str(seed_) + "Error!!")
            traceback.print_exc()

    log.info('Accuracy: '+str(accs))
    log.info('Backward transfer: '+str(bwts))
    log.info ('Final Avg Accuracy: {:5.2f}%, std:{:5.2f}'.format(np.mean(accs), np.std(accs)))
    log.info ('Final Avg Backward transfer: {:5.2f}%, std:{:5.2f}'.format(np.mean(bwts), np.std(bwts)))


