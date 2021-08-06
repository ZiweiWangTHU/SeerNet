from torchvision.models import alexnet,resnet18
from torchvision import models

from thop_count import profile
import torch
import torch.nn as nn
import numpy as np

import time
import math
import sys
sys.path.append("..")
import torch.nn.functional as F


import torch.nn.utils.prune as prune
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.sys.path.insert(0, os.path.abspath("../.."))
import models as customized_models
from lib.utils.quantize_utils import QConv2d, QLinear
from lib.utils.quant_layer import QuantConv2d

default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

def count_qlinear(m, x, y):

    total_mul = m.in_features
    num_elements = y.numel()
    total_ops = total_mul * num_elements*m.w_bit*m.a_bit

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def count_qconv2d(m, x, y):

    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
    total_ops = y.nelement() * (m.in_channels * kernel_ops)* m.w_bit * m.a_bit
    m.total_ops += torch.DoubleTensor([int(total_ops)])


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


def get_bops(arch, prune_num, bit_vector, verbose=False):

    quant_strategy=[]
    for i in range(0,len(bit_vector),2):
        quant_strategy.append([bit_vector[i],bit_vector[i+1]])



    model = models.__dict__[arch](pretrained=False, num_classes=1000)
    quantizable_idx=[]
    for i, m in enumerate(model.modules()):
        if type(m) in [QConv2d]:
            quantizable_idx.append(i)
    quantizable_idx=quantizable_idx[1:]#quantization of the first layer is fixed to 8bit
    assert len(quant_strategy) == len(quantizable_idx)


    quantize_layer_bit_dict = {n: b for n, b in zip(quantizable_idx, quant_strategy)}

    for i, layer in enumerate(model.modules()):
        if i not in quantizable_idx:
            continue
        else:

            layer.w_bit = quantize_layer_bit_dict[i][0]
            layer.a_bit = quantize_layer_bit_dict[i][1]

    prune_layer_dict = {key: value for key, value in zip(quantizable_idx, prune_num)}
    for i, m in enumerate(model.modules()):
        if i in quantizable_idx:
            prune.ln_structured(m, name='weight', amount=prune_layer_dict[i], dim=0, n=2)

    input = torch.randn(1, 3, 224, 224)
    macs,_ = profile(model, inputs=(input, ),
                        custom_ops={QConv2d:count_qconv2d},verbose=verbose)
    ops_list=[]
    for i,m in enumerate(model.modules()):

        if i in quantizable_idx:

            ind=quantizable_idx.index(i)

            if ind==0:
                m_ops = m.total_ops.numpy() * (1 - prune_layer_dict[i])
            else:
                layer_ind=quantizable_idx[ind-1]

                m_ops = m.total_ops.numpy() * (1 - prune_layer_dict[i]) * (1 - prune_layer_dict[layer_ind])

        else:
            m_ops=m.total_ops.numpy()


        ops_list.append(m_ops)
    result=np.sum(ops_list)
    return result/(1000**3)
