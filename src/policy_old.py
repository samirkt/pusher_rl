import numpy as np
import torch
from torch import nn
import torch.optim as optim

class PPO:
    def __init__(self, input_size, num_actions):
        self.pi = Pi(input_size=input_size, num_actions=num_actions)
        self.v = V(input_size=input_size)

        self.pi_opt = optim.Adam(self.pi.parameters(), lr=0.001)
        self.v_opt = optim.Adam(self.v.parameters(), lr=0.001)

    # Define a simple placeholder policy function
    def sample(self, observation, env):
        return env.action_space.sample()
    
    def policy(self, input):
        return self.pi(input)
    
    def value(self, input):
        return self.v(input)

    # TODO: log_probs_old? Won't this be the same as log_probs?
    def update(self, states, actions, log_probs_old, returns, advantages, clip_param=0.2, ppo_epochs=4, batch_size=64):
        total_batch_size = states.size(0)
        for _ in range(ppo_epochs):
            # Generating random mini-batch indices
            for _ in range(total_batch_size // batch_size):
                rand_ids = np.random.randint(0, total_batch_size, batch_size)
                states_batch, actions_batch = states[rand_ids], actions[rand_ids]
                log_probs_old_batch, returns_batch, advantages_batch = log_probs_old[rand_ids], returns[rand_ids], advantages[rand_ids]
                
                # Compute log probabilities of new actions
                means, std_devs = self.pi(states_batch)
                dists = torch.distributions.Normal(means, std_devs)
                log_probs = dists.log_prob(actions_batch).sum(axis=-1)
                
                # Compute ratio (pi_theta / pi_theta_old)
                ratios = torch.exp(log_probs - log_probs_old_batch)
                
                # Compute PPO objective (clipped)
                surr1 = ratios * advantages_batch
                surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Update policy network
                self.pi_opt.zero_grad()
                policy_loss.backward()
                self.pi_opt.step()
                
                # Compute value loss and update value network
                values = self.v(states_batch).squeeze()
                value_loss = (returns_batch - values).pow(2).mean()
                
                self.v_opt.zero_grad()
                value_loss.backward()
                self.v_opt.step()


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
        self.std = nn.Linear(64, num_actions)
        # Fixed exploration
        #self.log_std = nn.Parameter(torch.zeros(1, num_actions))
    
    def forward(self, x):
        x = self.network(x)
        mean = self.mean(x)
        std = torch.exp(self.std(x))
        # Fixed exploration
        #std = torch.exp(self.log_std)
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