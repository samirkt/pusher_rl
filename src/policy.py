import torch
from torch import nn

class PPO:
    def __init__(self):
        ...

    # Define a simple placeholder policy function
    def sample(self, observation, env):
        # Run observation thru pi
        ...

        return env.action_space.sample()

    def update(self):
        ...

class Pi(nn.Module):
    def __init__(self, input_size, num_actions):
        super(Pi, self).__init__()

        self.network = nn.Sequential(
           nn.Linear(input_size, 64),
           nn.ReLU(),
           nn.Linear(64, num_actions),
           nn.Softmax(dim=-1),
        )
    
    def forward(self, x):
        return self.network(x)

class V(nn.Module):
    def __init__(self, input_size):
        super(V, self).__init__()

        self.network = nn.Sequential(
           nn.Linear(input_size, 64),
           nn.ReLU(),
           nn.Linear(64, 1),
        )
    
    def forward(self, x):
        return self.network(x)