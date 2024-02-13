import torch
from torch import nn

class Pi(nn.Module):
    def __init__(self, input_size, num_actions):
        super(Pi, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.mean = nn.Linear(64, num_actions)
        # Fixed exploration for more stable learning
        self.log_std = nn.Parameter(torch.zeros(num_actions))

    def forward(self, x):
        x = self.network(x)
        mean = self.mean(x)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

class V(nn.Module):
    def __init__(self, input_size):
        super(V, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)
