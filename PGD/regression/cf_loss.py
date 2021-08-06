import torch.nn as nn
import torch

class cf_loss(nn.Module):
    def __init__(self, cf_weights):
        super(cf_loss, self).__init__()
        self.cf_weights = nn.Parameter(cf_weights)
    def forward(self, pred, truth):
        result = torch.mul(self.cf_weights, ((pred-truth)**2))
        return result.mean()
