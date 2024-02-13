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

    def ppo_update(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, log_probs, returns, advantages, clip_param=0.2):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        # Policy loss
        new_log_probs = policy_net(states).log_prob(actions)
        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        values = value_net(states)
        value_loss = (returns - values).pow(2).mean()

        # Update policy
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        # Update value
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

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