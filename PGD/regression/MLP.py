import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,input_dim,hidden_size):
        super(MLP, self).__init__()

        self.feature = nn.Sequential(
            
            nn.Linear(input_dim, hidden_size),
            nn.ELU(),
            # nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            # nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 1)
            
        )
    def forward(self, x):
        x = self.feature(x)
        return x