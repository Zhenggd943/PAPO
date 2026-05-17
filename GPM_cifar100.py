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
from common.utils import create_log_dir
from polar_express import polar_express_

## Define AlexNet model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class AlexNet(nn.Module):
    def __init__(self,taskcla):
        super(AlexNet, self).__init__()
        self.act=OrderedDict()
        self.map =[]
        self.ksize=[]
        self.in_channel =[]
        self.map.append(32)
        self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        s=compute_conv_output_size(32,4)
        s=s//2
        self.ksize.append(4)
        self.in_channel.append(3)
        self.map.append(s)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False)
        s=compute_conv_output_size(s,3)
        s=s//2
        self.ksize.append(3)
        self.in_channel.append(64)
        self.map.append(s)
        self.conv3 = nn.Conv2d(128, 256, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(256, track_running_stats=False)
        s=compute_conv_output_size(s,2)
        s=s//2
        self.smid=s
        self.ksize.append(2)
        self.in_channel.append(128)
        self.map.append(256*self.smid*self.smid)
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)

        self.fc1 = nn.Linear(256*self.smid*self.smid,2048, bias=False)
        self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.fc2 = nn.Linear(2048,2048, bias=False)
        self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.map.extend([2048])
        
        self.taskcla = taskcla
        self.fc3=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.fc3.append(torch.nn.Linear(2048,n,bias=False))
        
    def forward(self, x):
        bsz = deepcopy(x.size(0))
        self.act['conv1']=x
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

        self.act['conv2']=x
        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

        self.act['conv3']=x
        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

        x=x.view(bsz,-1)
        self.act['fc1']=x
        x = self.fc1(x)
        x = self.drop2(self.relu(self.bn4(x)))

        self.act['fc2']=x        
        x = self.fc2(x)
        x = self.drop2(self.relu(self.bn5(x)))
        y=[]
        for t,i in self.taskcla:
            y.append(self.fc3[t](x))
            
        return y

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer.param_groups:
        if (epoch ==1):
            param_group['lr']=args.lr
        else:
            param_group['lr'] /= args.lr_factor

def train(args, model, device, x,y, optimizer,criterion, task_id, epoch):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    x = x.to(device)
    y = y.to(device)
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r):
            b=r[i:i+args.batch_size_train]
        else:
            b=r[i:]
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()        
        output = model(data)
        loss = criterion(output[task_id], target)        
        loss.backward()
        if args.use_papo == 'True':
            for k, (m, params) in enumerate(model.named_parameters()):
                if k<15 and len(params.size())!=1:
                    sz = params.grad.data.size(0)
                    flat_grad = params.grad.data.view(sz, -1)

                    if epoch % args.interval == 0 and len(params.size()) == 4 and 'weight' in m:
                        papo_grad = polar_express_(args, flat_grad)
                        final_grad = flat_grad + args.Lambda0 * papo_grad
                        params.grad.data = final_grad.view(params.size())

        optimizer.step()

def train_projected(args,model,device,x,y,optimizer,criterion,feature_mat,task_id,epoch, Lambda):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    x = x.to(device)
    y = y.to(device)
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r):
            b=r[i:i+args.batch_size_train]
        else:
            b=r[i:]
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()        
        output = model(data)
        loss = criterion(output[task_id], target)         
        loss.backward()
        # Gradient Projections

        kk = 0 
        for k, (m,params) in enumerate(model.named_parameters()):
            if k < 15 and len(params.size()) != 1:
                sz = params.grad.data.size(0)
                flat_grad = params.grad.data.view(sz, -1)
                if args.use_papo == 'True':
                    if epoch % args.interval == 0:
                        papo_grad = polar_express_(args, flat_grad)
                        final_grad = flat_grad - torch.mm(flat_grad, feature_mat[kk]) + args.Lambda * papo_grad
                    else:
                        final_grad = flat_grad - torch.mm(flat_grad, feature_mat[kk])
                else:
                    final_grad = flat_grad - torch.mm(flat_grad, feature_mat[kk])
                params.grad.data = final_grad.view(params.size())
                kk += 1
            elif (k < 15 and len(params.size()) == 1) and task_id != 0:
                params.grad.data.fill_(0)
        optimizer.step()

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
            if i+args.batch_size_test<=len(r):
                b=r[i:i+args.batch_size_test]
            else:
                b=r[i:]
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

def get_representation_matrix (net, device, x, y=None): 
    # Collect activations by forward pass
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    x = x.to(device)
    b=r[0:125] # Take 125 random samples
    example_data = x[b]
    example_data = example_data.to(device)
    example_out  = net(example_data)
    
    batch_list=[2*12,100,100,125,125] 
    mat_list=[]
    act_key=list(net.act.keys())
    for i in range(len(net.map)):
        bsz=batch_list[i]
        k=0
        if i<3:
            ksz= net.ksize[i]
            s=compute_conv_output_size(net.map[i],net.ksize[i])
            mat = np.zeros((net.ksize[i]*net.ksize[i]*net.in_channel[i],s*s*bsz))
            act = net.act[act_key[i]].detach().cpu().numpy()
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:,k]=act[kk,:,ii:ksz+ii,jj:ksz+jj].reshape(-1) 
                        k +=1
            mat_list.append(mat)
        else:
            act = net.act[act_key[i]].detach().cpu().numpy()
            activation = act[0:bsz].transpose()
            mat_list.append(activation)

    log.info('-'*30)
    log.info('Representation Matrix')
    log.info('-'*30)
    for i in range(len(mat_list)):
        log.info('Layer {} : {}'.format(i+1,mat_list[i].shape))
    log.info('-'*30)
    return mat_list    


def update_GPM (model, mat_list, threshold, feature_list=[],):
    log.info('Threshold: %s', threshold)
    if not feature_list:
        # After First Task 
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U,S,Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria (Eq-5)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1  
            feature_list.append(U[:,0:r])
    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1**2).sum()
            # Projected Representation (Eq-8)
            act_hat = activation - np.dot(np.dot(feature_list[i],feature_list[i].transpose()),activation)
            U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
            # criteria (Eq-9)
            sval_hat = (S**2).sum()
            sval_ratio = (S**2)/sval_total               
            accumulated_sval = (sval_total-sval_hat)/sval_total
            
            r = 0
            for ii in range (sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                log.info('Skip Updating GPM for layer: {}'.format(i+1))
                continue
            # update GPM
            Ui=np.hstack((feature_list[i],U[:,0:r]))  
            if Ui.shape[1] > Ui.shape[0] :
                feature_list[i]=Ui[:,0:Ui.shape[0]]
            else:
                feature_list[i]=Ui
    
    log.info('-'*40)
    log.info('Gradient Constraints Summary')
    log.info('-'*40)
    for i in range(len(feature_list)):
        log.info('Layer {} : {}/{}'.format(i+1,feature_list[i].shape[1], feature_list[i].shape[0]))
    log.info('-'*40)
    return feature_list  


def main(args, str_time):
    tstart=time.time()
    ## Device Setting 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ## Load CIFAR100 DATASET
    from dataloader import cifar100 as cf100
    data,taskcla,inputsize=cf100.get(seed=args.seed, pc_valid=args.pc_valid)

    acc_matrix=np.zeros((10,10))
    criterion = torch.nn.CrossEntropyLoss()

    task_id = 0
    task_list = []
    for k,ncla in taskcla:
        threshold = np.array([0.97] * 5) + task_id*np.array([0.003] * 5)

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
        best_loss=np.inf
        log.info('-'*40)
        log.info('Task ID :{} | Learning Rate : {}'.format(task_id, lr))
        log.info('-'*40)
        
        if task_id==0:
            model = AlexNet(taskcla).to(device)
            log.info('Model parameters ---')
            for k_t, (m, param) in enumerate(model.named_parameters()):
                log.info('%d %s %s', k_t,m,param.shape)
            log.info('-'*40)

            best_model=get_model(model)
            feature_list =[]
            optimizer = optim.SGD(model.parameters(), lr=lr)

            for epoch in range(1, args.n_epochs+1):
                # Train
                clock0=time.time()
                train(args, model, device, xtrain, ytrain, optimizer, criterion, k, epoch)
                clock1=time.time()
                tr_loss,tr_acc = test(args, model, device, xtrain, ytrain,  criterion, k)
                log.info('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,
                                                                                                    tr_loss,
                                                                                                    tr_acc,
                                                                                                    1000 * (clock1 - clock0)))
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion, k)
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
            set_model_(model,best_model)
            # Test
            log.info('-'*40)
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion, k)
            log.info('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))
            # Memory Update  
            mat_list = get_representation_matrix (model, device, xtrain, ytrain)
            feature_list = update_GPM (model, mat_list, threshold, feature_list)
            Lambda = args.Lambda
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
            feature_mat = []
            # Projection Matrix Precomputation
            for i in range(len(model.act)):
                Uf=torch.Tensor(np.dot(feature_list[i],feature_list[i].transpose())).to(device)
                log.info('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape))
                feature_mat.append(Uf)
            log.info('-' * 40)
            for epoch in range(1, args.n_epochs+1):
                # Train 
                clock0=time.time()
                train_projected(args, model,device,xtrain, ytrain,optimizer,criterion,feature_mat,k,epoch, Lambda)
                clock1=time.time()
                tr_loss, tr_acc = test(args, model, device, xtrain, ytrain,criterion,k)
                log.info('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,
                                                                                                    tr_loss,
                                                                                                    tr_acc,
                                                                                                    1000 * (clock1 - clock0)))
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
            set_model_(model,best_model)
            # Test
            log.info('-' * 40)
            test_loss, test_acc = test(args, model, device, xtest, ytest, criterion, k)
            log.info('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss, test_acc))
            # Memory Update 
            mat_list = get_representation_matrix(model, device, xtrain, ytrain)
            feature_list = update_GPM (model, mat_list, threshold, feature_list)
            Lambda /= args.Lambda_factor

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

    # Plots
    array = acc_matrix
    df_cm = pd.DataFrame(array, index = [i for i in ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10"]],
                      columns = [i for i in ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10"]])
    sn.set(font_scale=1.4) 
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})
    plt.show()

    return acc_matrix[-1].mean(), bwt


if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='Sequential CIFAR-100 with GPM')
    parser.add_argument('--algo_name', type=str, default='GPM',
                        help='True or False')
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=100, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--pc_valid',default=0.05,type=float,
                        help='fraction of training data used for validation')
    # PAPO parameters
    parser.add_argument('--use_papo', type=str, default='True',
                        help='True or False')
    parser.add_argument('--iteration_step', type=int, default=5,
                        help='iteration step of polar express')
    parser.add_argument('--Lambda0', type=float, default=5,
                        help='coefficient lambda')
    parser.add_argument('--Lambda', default=0.005, type=float,
                        help='coefficient lambda (default: 0.005)')
    parser.add_argument('--Lambda_factor', default=2, type=float,
                        help='lambda decay factor (default: 2)')
    parser.add_argument('--interval', type=int, default=2, metavar='N',
                        help='application interval of PAPO')
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr_min', type=float, default=1e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')
    parser.add_argument('--savename', type=str, default='./log/CIFAR100/GPM',
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
            traceback.print_exc()

    log.info('Accuracy: ' + str(accs))
    log.info('Backward transfer: ' + str(bwts))
    log.info('Final Avg Accuracy: {:5.2f}%, std:{:5.2f}'.format(np.mean(accs), np.std(accs)))
    log.info('Final Avg Backward transfer: {:5.2f}%, std:{:5.2f}'.format(np.mean(bwts), np.std(bwts)))