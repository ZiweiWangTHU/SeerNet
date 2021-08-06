
import os
import time
import math
import random
import shutil
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.models as models
import models as customized_models
import random
import logging
import numpy as np
import torch.nn.utils.prune as prune


from lib.utils.utils import Logger, AverageMeter, accuracy
from lib.utils.data_utils import get_dataset

from lib.utils.quantize_utils import  QConv2d, QLinear, calibrate
from math import ceil
from tensorboardX import SummaryWriter


# Models
device = torch.device("cuda")
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='data/imagenet', type=str)
parser.add_argument('--data_name', default='imagenet', type=str)
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--warmup_epoch', default=0, type=int, metavar='N',
                    help='manual warmup epoch number (useful on restarts)')
parser.add_argument('--train_batch_per_gpu', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test_batch', default=512, type=int, metavar='N',
                    help='test batchsize (default: 512)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_type', default='cos', type=str,
                    help='lr scheduler (exp/cos/step3/fixed)')
parser.add_argument('--schedule', type=int, nargs='+', default=[31, 61, 91],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', action='store_true',
                    help='use pretrained model')

# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50', choices=model_names,
                    help='model architecture:' + ' | '.join(model_names) + ' (default: resnet18)')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pin_memory', default=2, type=int,
                        help='pin_memory of Dataloader')
# Device options
parser.add_argument('--gpu_id', default='0,1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
lr_current = state['lr']
args.batch_size=args.train_batch_per_gpu* ceil(len(args.gpu_id) / 2)
print('batch size:',args.batch_size)

# Use CUDA
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
print('gpu:',args.gpu_id)
use_cuda = torch.cuda.is_available()


# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)


best_acc = 0  # best test accuracy



def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()


    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    start=time.time()


    for batch_idx, (inputs, targets) in enumerate(train_loader):


        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient
        optimizer.zero_grad()

        loss.backward()
        # do SGD step
        optimizer.step()

        if batch_idx % 50 == 0:
            print_logger.info("The train loss of epoch{}-batch-{}:{},top1 acc:{},top5 acc:{}".format(epoch,
                                                                       batch_idx, losses.avg,top1.avg,top5.avg))

    print_logger.info("The overall train loss of epoch{}:{},top1 acc:{},top5 acc:{},used_time:{}".format(epoch,
                                                                                              losses.avg,
                                                                                             top1.avg, top5.avg,time.time()-start))
    return losses.avg, top1.avg


def test(val_loader, model, criterion, epoch, use_cuda):



    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        # switch to evaluate mode
        model.eval()

        start = time.time()
        for batch_idx, (inputs, targets) in enumerate(val_loader):


            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))


    print_logger.info('Epoch:{}, test loss:{}, top1 acc:{}, top5 acc:{}, used time:{}'.format(epoch, losses.avg,top1.avg,top5.avg,time.time()-start))
    return losses.avg, top1.avg


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar',epoch=0):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global lr_current

    if epoch < args.warmup_epoch:
        lr_current = state['lr']*args.gamma
    elif args.lr_type == 'cos':
        # cos
        lr_current = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
    elif args.lr_type == 'exp':
        step = 1
        decay = args.gamma
        lr_current = args.lr * (decay ** (epoch // step))
    elif epoch in args.schedule:
        lr_current *= args.gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_current


def finetune(quant_strategy,exp_name='finetune',prune_policy=None):


    lr_current = args.lr
    best_acc=0
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch


    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)
    sub_checkpoint=os.path.join(args.checkpoint,str(exp_name))
    if not os.path.isdir(sub_checkpoint):
        os.makedirs(sub_checkpoint)
    run_name=os.path.join(args.checkpoint,'visualization')
    if not os.path.isdir(run_name):
        os.makedirs(run_name)
    writer = SummaryWriter(log_dir=run_name)


    train_loader, val_loader, n_class = get_dataset(dataset_name=args.data_name, batch_size=args.batch_size,
                                                    n_worker=args.workers, data_root=args.data,pin_memory=args.pin_memory)
    model = models.__dict__[args.arch](pretrained=args.pretrained,num_classes=1000)

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)#momentum




    quantizable_idx = []
    for i, m in enumerate(model.modules()):
        if type(m) in [QConv2d]:
            quantizable_idx.append(i)
    quantizable_idx=quantizable_idx[1:]

    print('quantizing:', (quantizable_idx))
    print((quant_strategy))
    quantize_layer_bit_dict = {n: b for n, b in zip(quantizable_idx, quant_strategy)}

    for i, layer in enumerate(model.modules()):
        if i not in quantizable_idx:
            continue
        else:

            layer.w_bit = quantize_layer_bit_dict[i][0]
            layer.a_bit = quantize_layer_bit_dict[i][1]
    model = model.cuda()
    model = calibrate(model, train_loader)


    model = torch.nn.DataParallel(model,device_ids=range(ceil(len(args.gpu_id)/2)))


    # Resume
    title = 'ImageNet-' + args.arch

    prune_model(model, prune_policy)



    if args.resume:

        print('==> Resuming from checkpoint..')


        checkpoint = torch.load(os.path.join(args.resume))
        best_acc = checkpoint['best_acc']
        print('resuming best acc:',best_acc)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'],strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])


        if os.path.isfile(os.path.join(sub_checkpoint, 'log.txt')):
            logger = Logger(os.path.join(sub_checkpoint, 'log.txt'), title=title, resume=True)
        else:
            logger = Logger(os.path.join(sub_checkpoint, 'log.txt'), title=title)
            logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])




    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return



    for epoch in range(start_epoch, args.epochs):


        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr_current))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        writer.add_scalar('train_loss'+'_'+str(exp_name),train_loss,epoch)
        writer.add_scalar('train_acc'+'_'+str(exp_name),train_acc,epoch)
        test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda)
        writer.add_scalar('test_loss'+'_'+str(exp_name),test_loss,epoch)
        writer.add_scalar('test_acc'+'_'+str(exp_name),test_acc,epoch)

        # append logger file
        logger.append([lr_current, train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        writer.add_scalar('best_acc' + '_' + str(exp_name), best_acc, epoch)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=sub_checkpoint, epoch=epoch)

    logger.close()
    writer.close()

    print('Best acc:')
    print(best_acc)
    return best_acc

def prune_model(model,prune_policy):
    print('=====> Pruning')
    prunable_idx = []
    for i, m in enumerate(model.modules()):
        if type(m) in [QConv2d]:
            prunable_idx.append(i)
    prunable_idx=prunable_idx[1:]
    print((prunable_idx))
    print(prune_policy)
    prune_layer_dict = {key: value for key, value in zip(prunable_idx, prune_policy)}
    for i,layer in enumerate(model.modules()):
        if i in prunable_idx:
            prune.ln_structured(layer, name='weight', amount=prune_layer_dict[i], dim=0, n=2)
    return model





if __name__ == '__main__':

    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    print_logger = logging.getLogger()
    print_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.basename(args.checkpoint) + '_log' + '.txt')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    print_logger.addHandler(fh)
    print_logger.addHandler(ch)

    if  'resnet18' in args.arch:
        prune_strategy=[0.13979352, 0.27091193, 0.42088735, 0.61789185, 0.11904202, 0,
                        0.05577452, 0.02558663, 0.09508231, 0.    , 0.     , 0.17391534,
                        0.20223643, 0.41929013, 0.16388999, 0.11159942, 0.63201606, 0.,
                        0.19483185]
        quant_vector=[4, 4, 2, 2, 7, 5, 3, 4, 6, 2, 4, 8, 3, 5, 7, 7, 5, 2, 5, 4, 5, 2, 2, 2, 6, 2, 7, 4, 6, 4, 2, 5, 8, 2, 4, 3, 3, 5]
    elif 'mobilenetv2' in args.arch:
        quant_vector=[ 8, 7, 6, 6, 4, 6, 5, 6, 4, 7, 6, 6, 8, 4, 4, 6, 5, 6, 8, 7, 6, 6, 4, 6, 8, 3, 5, 6, 4, 7, 8, 4, 5, 8,
         4, 6, 6, 6, 4, 6, 4, 8, 6, 4, 6, 8, 4, 6, 7, 5, 6, 8, 5, 6, 7, 4, 6, 8, 5, 8, 6, 4, 7, 7, 6, 8, 6, 4, 6, 7,
         6, 6, 6, 5, 7, 7, 5, 8, 8, 7, 4, 6, 8, 8, 4, 6, 5, 8, 6, 8, 6, 7, 5, 8, 6, 8, 8, 8, 5, 8, 6, 8, 8, 8]

        prune_strategy = [0,0,0.10005943, 0.191646, 0.11976264, 0.23125488, 0.1562162, 0.21855335,
            0.10542893, 0.1368392, 0.16366164, 0.21477097, 0.28086156, 0.18381168,
            0.14281463, 0.21056205, 0.13750421, 0.15259006, 0.1425915, 0.217176,
            0.20183505, 0.23165666, 0.2662687, 0.19464038, 0.22918665, 0.18618044,
            0.22193481, 0.2440036, 0.21972626, 0.20435818, 0.16830897, 0.22948292,
            0.12220504, 0.24659497, 0.13098075, 0.17524926, 0.1551066, 0.15673437,
            0.21098225, 0.21464992, 0.19426939, 0.21860462, 0.27278265, 0.22996318,
            0.1394073, 0.13579283, 0.19481003, 0.14152212, 0.11255494, 0.06533574,0,0]
    else:
        raise NotImplementedError

    quant_strategy=[]
    for i in range(0,len(quant_vector),2):
        quant_strategy.append([quant_vector[i], quant_vector[i + 1]])

    best_acc = finetune(quant_strategy, exp_name='finetune' + args.arch,
                        prune_policy=prune_strategy)

    print_logger.info([quant_strategy, prune_strategy, best_acc])



