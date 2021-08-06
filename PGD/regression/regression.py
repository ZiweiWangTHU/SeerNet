import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from main import test
from main import train
from MLP import MLP
from cf_loss import cf_loss
import random
import argparse

prune_table = [0.25, 0.5, 0.75]
quantization_table = [0.4, 0.6, 0.8, 1]
policy = []
policy = [(a, b, c, d, j, k, l, m) for a in prune_table for b in prune_table for c in prune_table  for d in prune_table for j in quantization_table[1:] for k in quantization_table  for l in quantization_table  for m in quantization_table[1:]]


def normalize(l):
    for i in l:
        i[18:36] = i[18:36] / 5

    return l
def Hamming(a1, a2):
    a1 = np.array(list(a1))
    a2 = np.array(list(a2))
    return sum(a1 != a2)

def gen_important_sample(distance, x, prune_table, quantization_table, dim):
    c_factual_list = []
    index = []
    for i in range(dim):
        index.append(prune_table.index(x[i]))
    for i in range(dim):
        index.append(quantization_table.index(x[i + dim]))

    cf_indexes = []
    cf_samples = []
    for i in range(len(index)):
        index1 = index[:]

        if index[i] != 0:
            index1[i] = index[i] - 1
            cf_indexes.append(index1)

    for i in cf_indexes:
        sample = []
        for j in range(dim):
            sample.append(prune_table[i[j]])
        for j in range(dim):
            sample.append(quantization_table[i[j + dim]])

        cf_samples.append(tuple(sample))
    uni_cf_list = []
    
    for i in cf_samples:
        if i not in uni_cf_list:# and i not in X_test:
            uni_cf_list.append(i)
    return uni_cf_list

def train_and_eval(trainloader, testloader, lr, optimizer, net, y_test, X_train, cf, Epoch, dim):
    #counterfactual weight matrix
    weights = torch.rand(len(X_train), 1).cuda()
    #new loss: origin loss times weight mat.
    loss_func = cf_loss(weights)
    best_err = 100.0
    for w in range(1):
        flag = False
        weights = torch.randn(len(X_train), 1)
        for epoch in range(Epoch):
            
            #adjust learning rate
            if epoch in [3000,6000] and flag == False:
                lr *= 0.1
                flag = True
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            train(trainloader, net, loss_func, optimizer, epoch, use_cuda=True, size = len(X_train))
            #net = net.to(device)
            
            if epoch % 100 == 99:
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    # measure data loading time
                    inputs, targets = inputs.cuda(), targets.cuda()
                    pred = net(inputs).cpu()
)
                    err = mean_squared_error(y_test, pred.detach().numpy())

                    if err < best_err:
                        best_err = err
                        torch.save(net.state_dict(), "mlp_cf.pkl") 
                    print("%d : "%(epoch))
                    print("my MLP: {}, best err: {}".format(err, best_err))
        #update the weight mat.
        if cf:
            for i in range(len(X_train)):
                flag = False
                cf_samples = gen_important_sample(1, X_train[i], prune_table, quantization_table, dim)
                cf_samples = torch.tensor(cf_samples, dtype = torch.float32) 
                cf_samples = cf_samples.cuda()
                pred = net(cf_samples)                
                weights[i] = torch.sqrt(((pred - y_train[i])**2)).mean()
            
            std = weights.std()
            weights = weights / std
            weights = weights.cuda()
            loss_func = cf_loss(weights)
        else:
            loss_func = nn.MSELoss()
    return best_err

device = torch.device("cuda")
def parse_args():
    parser = argparse.ArgumentParser(description='mlp regression')
    parser.add_argument('--sample_path', default='results/resnet18_sample.npy')
    parser.add_argument('--acc_path', default='results/resnet18_acc.npy')
    parser.add_argument('--lr', default=0.2)
    parser.add_argument('--batch', default=400)
    parser.add_argument('--epoch', default=5000) 
    parser.add_argument('--dim', default=18)
    
def main():
        
    rand_sample = np.load(args.sample_path)
    rand_acc = np.load(args.acc_path)

    rand_sample = normalize(rand_sample)
    rand_acc = rand_acc / 100
    acc_table = acc_table / 100
    rand_sample = rand_sample.tolist()
    rand_acc = rand_acc.tolist()
    sample = sample.tolist()
    acc_table = acc_table.tolist()

    mse = []


    #repeat to get different samples
    for a in range(args.rpt_num):

        X_train,X_test, y_train, y_test =\
        train_test_split(rand_sample, rand_acc, test_size=0.1, random_state=None)
        print("%d | %d : "%(a, rpt_num))
        y_train1 = []
        X_train1 = []
        y_train2 = []
        #random sample
        X_train2 = random.sample(X_train, 190)
        for i in X_train2:
            y_train2.append(y_train[X_train.index(i)])
        X_train2 = rand_sample
        y_train2 = rand_acc

        mse1 = mean_squared_error(y_test,pred) 

        X_train1 = random.sample(sample, 50)
        for i in X_train1:
            y_train1.append(acc_table[X_train1.index(i)])

        testset = Data.TensorDataset(torch.tensor(X_train2, dtype = torch.float32), torch.tensor(y_train2, dtype = torch.float32))
        testloader = Data.DataLoader(dataset=testset, batch_size=args.batch, shuffle=False)

        trainset1 = Data.TensorDataset(torch.tensor(X_train2, dtype = torch.float32), torch.tensor(y_train2, dtype = torch.float32))
        trainloader1 = Data.DataLoader(dataset=trainset1, batch_size=args.batch, shuffle=True)
        
        #repeat 10 times to get the mean
        for i in range(10):
            net = MLP()
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
            mse = train_and_eval(trainloader1, testloader, args.lr, optimizer, net4, y_train2, X_train2, cf = True, Epoch = args.epoch, dim = args.dim)
            print("%d | 10 "%(i))
            mse.append(mse)

        print("mse: {}".format(mse))
        print("mean: %f"%(np.mean(mse)))

        

if __name__ == "__main__":
    main()

