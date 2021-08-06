import numpy as np
import torch.nn as nn
import torch
from regression.MLP import MLP
import time
import os
import random
import argparse
from bops_measure import get_bops

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


prune_table = [0.25, 0.5, 0.75]
quant = [2, 4, 6, 8]
min_bit = np.min(quant)
max_bit = np.max(quant)
class max_loss(nn.Module):
    def __init__(self):
        super(max_loss, self).__init__()

    def forward(self, acc):
        result = torch.log(acc)
        return result.mean()

def pgd(model, x, loss_fn, num_steps, step_size, step_norm, eps, eps_norm,arch='qresnet18',layer_nums=19,
                               clamp=(0, 1),bops_threshold=None):

    x_adv = x.clone().detach().requires_grad_(True).to(x.device)
    model.train()
    result=x_adv



    for i in range(num_steps):

        _x_adv = x_adv.clone().detach().requires_grad_(True)

        prediction = model(_x_adv)
        print('predicted accuracy',prediction.cpu().detach().numpy())
        loss = loss_fn(prediction)
        loss.backward()



        with torch.no_grad():


            if step_norm == 'inf':

                gradients = _x_adv.grad.sign() * step_size


            else:

                gradients = _x_adv.grad * step_size / _x_adv.grad.view(_x_adv.shape[0], -1) \
                    .norm(step_norm, dim=-1) \

            x_adv += gradients


        if eps_norm == 'inf':

            x_adv = x_adv.clamp(*clamp)

        else:
            delta = x_adv - x


            mask = delta.view(delta.shape[0], -1).norm(eps_norm, dim=1) <= eps

            scaling_factor = delta.view(delta.shape[0], -1).norm(eps_norm, dim=1)
            scaling_factor[mask] = eps

            delta *= eps / scaling_factor.view(-1, 1, 1, 1)

            x_adv = x + delta


        x_adv_arr = x_adv[0].cpu().detach().numpy()

        bops = get_bops(arch,x_adv_arr[:layer_nums], np.around( max_bit* x_adv_arr[layer_nums:] + min_bit))
        print(bops)


        if bops >= bops_threshold:
            print('====> Above threshold')

            return result

        print('bops', bops)

        result=x_adv.clamp(*clamp)



    return result

def run_pgd(net,step_size,arch='qresnet18',layer_nums=19,bops_threshold=30):



    prune_policy = random.choices(prune_table,  k=layer_nums)
    quant_policy = random.choices(quant,  k=layer_nums*2)

    quant_policy_norm = [(item - min_bit) / (max_bit-min_bit) for item in quant_policy]
    x = torch.tensor([tuple(prune_policy + quant_policy_norm)]).float()


    x = x.cuda()
    loss_fn = max_loss()

    x_adv = pgd(net, x,  loss_fn,
                                       num_steps=400, step_size=step_size,
                                       eps=1, eps_norm='inf',arch=arch,layer_nums=layer_nums,
                                       step_norm=2,bops_threshold=bops_threshold)

    print('step size:',step_size)
    print('initial strategy',prune_policy,quant_policy)
    print('initial bops:',get_bops(arch,prune_policy,quant_policy))
    pred=net(x)
    print('initial accuracy',pred)
    x_adv_arr=x_adv[0].cpu().detach().numpy()
    print('final strategy',x_adv_arr[:layer_nums],np.around((max_bit-min_bit)*x_adv_arr[layer_nums:]+min_bit))
    bops=get_bops(arch,x_adv_arr[:layer_nums],np.around(max_bit*x_adv_arr[layer_nums:]+min_bit))
    print('final bops:',bops)
    x_adv_tensor = torch.tensor([(x_adv_arr)]).cuda()
    pred = net(x_adv_tensor)
    print('final accuracy:', pred)

parser = argparse.ArgumentParser(description='PGD search')
parser.add_argument('--arch', default='qresnet18', type=str)
parser.add_argument('--layer_nums', default=19, type=int)
parser.add_argument('--step_size', default=0.005, type=float)
parser.add_argument('--max_bops', default=30, type=float)
parser.add_argument('--hidden_size', default=15, type=int)
parser.add_argument('--pretrained_weight', default='mlp_res18.pkl', type=str)
args = parser.parse_args()
input_dim=3*args.layer_nums
net = MLP(input_dim=input_dim,hidden_size=args.hidden_size).cuda()
# net.load_state_dict(torch.load(args.pretrained_weight))

run_pgd(net,step_size=args.step_size,arch=args.arch,layer_nums=args.layer_nums,bops_threshold=args.max_bops)
