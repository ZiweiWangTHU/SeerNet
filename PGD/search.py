import random
import numpy as np
import regression
import torch
from MLP import MLP
from sklearn.model_selection import train_test_split
import torch.utils.data as Data

def random_gen(prune_table, quantization_table, dim, num = 50):
    group = []
    for i in range(num):
        s = random.choices(prune_table,k=dim)
        s += random.choices(quantization_table,k=dim)
        group.append(tuple(s))
    return group

def select(candidates, fitness, select_num):
    sort = [x for _,x in sorted(zip(fitness,candidates), reverse = True)]
    return sort[:select_num]


def fit(candidates, net, prune_table, quantization_table):
    fitness = []
    candidates1 = torch.tensor(candidates, dtype = torch.float32)
    candidates1 = candidates1.cuda()
    preds = net(candidates1)
    j = 0
    for i in candidates:
        cf_samples = regression.gen_important_sample(1, i, prune_table, quantization_table)
        cf_samples = torch.tensor(cf_samples, dtype = torch.float32) 
        cf_samples = cf_samples.cuda()
        pred = net(cf_samples)
        fit_val = ((pred - preds[j])**2).mean()
        j = j + 1
        fitness.append(fit_val)
    return fitness

def get_mutation(quantization_table, prune_table, dim, group, m_prob = 0.1, mutation_num = 10):
    k = len(group)
    mutation = []
    ids = np.random.choice(k, int(k * m_prob))
    i = 0
    while len(mutation) < mutation_num:
        for i in ids:
            group[i] = list(group[i])
            mutated_dim_num = np.random.choice(2*dim)
            mutated_dim_ids = np.random.choice(2*dim, mutated_dim_num + 1, replace = False)
            for dim_ids in mutated_dim_ids:
                if dim_ids < dim:
                    to_be_selected = np.random.choice(prune_table)
                    while group[i][dim_ids] == to_be_selected:
                        to_be_selected = np.random.choice(prune_table) 
                    group[i][dim_ids] = to_be_selected 
                else:
                    to_be_selected = np.random.choice(quantization_table)
                    while group[i][dim_ids] == to_be_selected:
                        to_be_selected = np.random.choice(quantization_table) 
                    group[i][dim_ids] = to_be_selected 
            mutation.append(tuple(group[i]))
    return mutation

def get_crossover(group, candidates, c_prob = 0.3, crossover_num = 10):
    max_iter = 5
    i = 0
    k = len(group)
    crossover = []
    while len(crossover) < crossover_num and i < max_iter :
        i = i + 1
        id1, id2 = np.random.choice(k, 2, replace=False)
        p1 = list(group[id1])
        p2 = list(group[id2])

        ids = np.random.choice(2*dim, int(c_prob*2*dim), replace = False)
        for j in ids:
            p1[j], p2[j] = p2[j], p1[j]
        p1 = tuple(p1)
        p2 = tuple(p2)

        if p1 not in candidates:
            crossover.append(p1)
        if p2 not in candidates:
            crossover.append(p2)    
    return crossover

def search(net, X_train, dim):
    prune_table = [0.25,0.5,0.75]
    quantization_table = [0.4, 0.6, 0.8, 1]
    select_num = 400
    population_num = 800
    mutation_num = 125
    m_prob = 0.1
    c_prob = 0.3
    crossover_num = 475
    random_num = population_num - mutation_num - crossover_num
    max_iters = 2000
    x = 0
    
    candidates = []
    most_fited = []
    print('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_iters = {}'.format(population_num, select_num, mutation_num, crossover_num, random_num, max_iters))

    candidates = random_gen(prune_table, quantization_table, population_num)
    
    for i in range(max_iters):
        print("%d | %d: "%(i, max_iters))
        fitness = fit(candidates, net, prune_table, quantization_table)
        most_fited = select(candidates, fitness, select_num = select_num)        
        mutation = get_mutation(quantization_table, prune_table, dim, most_fited, m_prob, mutation_num)
        crossover = get_crossover(most_fited, candidates, c_prob, crossover_num)

        random_num = population_num - len(mutation) -len(crossover)
        rand = random_gen(prune_table, quantization_table, dim, random_num)
        
        candidates = []
        candidates.extend(mutation)
        candidates.extend(crossover)
        candidates.extend(rand)


    return most_fited

def parse_args():
    parser = argparse.ArgumentParser(description='evolution search')
    parser.add_argument('--sample_path', default='results/res20/res20_sample_result_400.npy')
    parser.add_argument('--acc_path', default='results/res20/res20_acc_result_400.npy')
    parser.add_argument('--lr', default=0.2)
    parser.add_argument('--batch', default=400)
    parser.add_argument('--epoch', default=1000) 
    parser.add_argument('save_path', default = 'search_result.npy')
    parser.add_argument('--dim', default=18)  

def main():
    prune_table = [0.25,0.5,0.75]
    quantization_table = [0.4, 0.6, 0.8, 1]
    sample = np.load(args.sample_path)
    acc_table = np.load(args.acc_path)
    acc_table = acc_table / 100
    X_train,X_test, y_train, y_test =\
    train_test_split(sample, acc_table, test_size=0.2)
    y_train2 = []
    
    X_train = X_train.tolist()
    y_train = y_train.tolist()

    trainset = Data.TensorDataset(torch.tensor(X_train, dtype = torch.float32), torch.tensor(y_train, dtype = torch.float32))
    trainloader = Data.DataLoader(dataset=trainset, batch_size=args.batch, shuffle=True)
    testset = Data.TensorDataset(torch.tensor(X_test, dtype = torch.float32), torch.tensor(y_test, dtype = torch.float32))
    testloader = Data.DataLoader(dataset=testset, batch_size=args.batch, shuffle=False)
    net = MLP()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    result = search(net, X_train, args.dim)
    print(torch.mean(a))
    print("active: ")
    b = fit(result, net, prune_table, quantization_table)
    b.sort(reverse = True)
    b = torch.tensor(b, dtype=torch.float32)
    print(torch.mean(b))
    np.save(args.save_path, result)
if __name__ == "__main__":
    main()