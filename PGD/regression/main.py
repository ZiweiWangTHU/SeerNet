import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import argparse
from models import AlexNet, Q_AlexNet, AlexNet1
import torch.quantization
import torch.nn.utils.prune as prune
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sys
import datetime
import copy
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import time
import torch.utils.data as data
from utils.quantize_utils import quantize_model
import random
import models
import numpy as np

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

batch = 128
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

acc_table = []
q_policy = []
p_policy = []


lr = 4e-3
Epoch = 200
finetune_epoch = 300 
device = torch.device("cuda")

def train(trainloader, model, criterion, optimizer, epoch, use_cuda, size = None):
    # switch to train mode
    model.train()
    if use_cuda:
        model = model.to(device)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        #print(inputs)
        # compute output
        
        outputs = model(inputs)
        #print(outputs.shape)
        if size is not None:
            targets = targets.reshape(size, 1)

        loss = criterion(outputs, targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda, save_pth = None, save = False, best_acc = 0.0):
    #global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    #bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data, inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    if save:
        if top1.avg > best_acc:
           torch.save(model.state_dict(), save_pth) 

    return (losses.avg, top1.avg)

def pretrain(model, lr, Epoch, trainloader, testloader, criterion, save_pth):
    StartTime = datetime.datetime.now()
    best_acc = 0.0
    model_params = []
    for name, params in model.named_parameters():
        if 'act_alpha' in name:
            model_params += [{'params': [params], 'lr': 1e-1, 'weight_decay': 1e-4}]
        elif 'wgt_alpha' in name:
            model_params += [{'params': [params], 'lr': 2e-2, 'weight_decay': 1e-4}]
        else:
            model_params += [{'params': [params]}] 
    optimizer = torch.optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=3e-5)
    for epoch in range(Epoch):
        for param_group in optimizer.param_groups:
            if epoch in [150, 225]:
                param_group['lr'] *= 0.1
        print('\nEpoch: [%d | %d]' % (epoch + 1, Epoch))
        S_T = datetime.datetime.now() 
        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda = True)
        test_loss, test_acc = test(testloader, model, criterion, epoch, save_pth = save_pth, use_cuda = True, save = True, best_acc = best_acc)
        best_acc = max(test_acc, best_acc)
        print('the accuracy is %.03f, the best accuracy is %.03f'%(test_acc, best_acc))
        E_T = datetime.datetime.now()
    
    print('Best acc:')
    print(best_acc)
    EndTime = datetime.datetime.now()
    print('time consumed: {}'.format(EndTime - StartTime))
def finetune(model_pth, sample, finetune_epoch, trainloader, testloader, criterion, lr):
    global acc_table
    num = 0
    l = len(sample)
    for s in sample:
        print("samples: [%d | %d ]"%(num, l), file = sys.stdout)
        num = num + 1
        s1 = s[18:36]
        s2 = s[0:18]
        s1 = [8] * 18
        s2 = [0.11573754, 0.6413544, 0.62782437, 0.67589295, 0.32582024, 0.3806936, 0.5084326, 0.15915419, 0.5718948, 0.38279435, 0.44264346, 0.39072034, 0.13285162, 0.05415095, 0.02470551, 0.46598032, 0.05738132, 0.6798185 ] 
        s1 = [3, 2, 4, 3, 4, 3, 2, 5, 4, 3, 3, 3, 2, 3, 2, 3, 3, 2]
        model_1 = models.QResNet(block=models.QBasicBlock, num_blocks=[3,3,3], w_bit = s1, a_bit = s1)
        model_1.load_state_dict(model_pth, strict=False)
        model_1 = model_1.to(device)

        optimizer = torch.optim.Adam(model_1.parameters(), lr = lr)

        print("finetune after pruning ...", file = sys.stdout)

        test_loss, test_acc = test(testloader, model_1, criterion, 1, use_cuda = True) 
        print('the accuracy is %.03f'%(test_acc), file = sys.stdout)        
        
        for i in range(0,3):
            prune.ln_structured(model_1.layer1[i].conv1, name = 'weight', amount = s2[2*i], dim = 0, n = 2)
            prune.ln_structured(model_1.layer1[i].conv2, name = 'weight', amount = s2[2*i+1], dim = 0, n = 2)
        for i in range(3,6):
            prune.ln_structured(model_1.layer2[i-3].conv1, name = 'weight', amount = s2[2*i], dim = 0, n = 2)
            prune.ln_structured(model_1.layer2[i-3].conv2, name = 'weight', amount = s2[2*i+1], dim = 0, n = 2)
        for i in range(6,9):
            prune.ln_structured(model_1.layer3[i-6].conv1, name = 'weight', amount = s2[2*i], dim = 0, n = 2)
            prune.ln_structured(model_1.layer3[i-6].conv2, name = 'weight', amount = s2[2*i+1], dim = 0, n = 2)
        
        print('policy:{}'.format(s2, s1))
        
        test_loss, test_acc = test(testloader, model_1, criterion, 1, use_cuda = True) 
        print('the accuracy is %.03f'%(test_acc), file = sys.stdout) 
        best_acc = 0.0
        f_e = 0
        while f_e < finetune_epoch:
            for param_group in optimizer.param_groups:
                if f_e in [150, 225]:
                    param_group['lr'] *= 0.1
                #print('lr:{}'.format(param_group['lr']))
            print('\nEpoch: [%d | %d]' % (f_e + 1, finetune_epoch), file = sys.stdout)
            train_loss, train_acc = train(trainloader, model_1, criterion, optimizer, f_e, use_cuda = True)
            test_loss, test_acc = test(testloader, model_1, criterion, f_e, use_cuda = True, save = False, best_acc = best_acc)
            best_acc = max(best_acc, test_acc)
            print('the accuracy is %.03f, the best accuracy is %.03f'%(test_acc, best_acc), file = sys.stdout) 
            f_e = f_e + 1
            
        acc_table.append(float(best_acc))
        print(acc_table, file = sys.stdout)
def main():
    StartTime = datetime.datetime.now()
    name = str(StartTime) + '.log'
    sys.stdout = Logger(name, sys.stdout)
    print("--------StartTime: {}--------".format(StartTime), file = sys.stdout)
    
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

    train_dataset = tv.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=5)

    test_dataset = tv.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch, shuffle=False, num_workers=5)
'
    
    prune_table = [0.25, 0.5, 0.75]
    quantization_table = [2, 3, 4,6]
    policy = []

    sample = []
    
    sample = np.load("search_result_100.npy")
    sample = sample.tolist()

    criterion = nn.CrossEntropyLoss().cuda()

    ch = torch.load("res20_3.pkl")

    finetune(ch, sample, finetune_epoch, trainloader, testloader, criterion, lr)

    print('sample:{}'.format(sample))
    print('acc_table:{}'.format(acc_table))
    np.save("search_acc_100.npy", acc_table)
    X_train,X_test, y_train, y_test =\
    train_test_split(sample, acc_table, test_size=0.1, random_state=1)

     
    #MLP
    regr = MLPRegressor(random_state=1,max_iter=10000).fit(X_train, y_train)
    pred=regr.predict(X_test)
    score=regr.score(X_test,y_test)
    print('pred:', file = sys.stdout)
    print(list(pred), file = sys.stdout)
    print('actual:', file = sys.stdout)
    print(y_test, file = sys.stdout)
    print(mean_squared_error(y_test,pred), file = sys.stdout)
    print(score, file = sys.stdout)
    
    EndTime = datetime.datetime.now()
    print("--------EndTime: {}--------".format(EndTime), file = sys.stdout)
    
def adjust_learning_rate(optimizer, epoch):
    global lr
    if epoch in [150, 225]:
        lr *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__ == "__main__":
    main()