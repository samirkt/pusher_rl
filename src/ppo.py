import torch
import torch.optim as optim
from networks import Pi, V
from utils import process_trajectory
import numpy as np

class PPO:
    def __init__(self, config):
        self.config = config
        self.pi = Pi(config.STATE_SPACE_SIZE, config.ACTION_SPACE_SIZE)
        self.v = V(config.STATE_SPACE_SIZE)
        self.pi_opt = optim.Adam(self.pi.parameters(), lr=config.LEARNING_RATE_PI)
        self.v_opt = optim.Adam(self.v.parameters(), lr=config.LEARNING_RATE_V)

    def update(self, trajectories):
        states, actions, old_log_probs, returns, advantages = process_trajectory(trajectories, self.config.GAMMA, self.config.LAMBDA_GAE)

        total_batch_size = states.size(0)
        for _ in range(self.config.PPO_EPOCHS):
            for _ in range(total_batch_size // self.config.BATCH_SIZE):
                rand_ids = torch.randperm(total_batch_size)[:self.config.BATCH_SIZE]
                states_batch, actions_batch, old_log_probs_batch = states[rand_ids], actions[rand_ids], old_log_probs[rand_ids]
                returns_batch, advantages_batch = returns[rand_ids], advantages[rand_ids]

                # Policy loss
                means, stds = self.pi(states_batch)
                dists = torch.distributions.Normal(means, stds)
                log_probs = dists.log_prob(actions_batch).sum(axis=-1)
                ratios = torch.exp(log_probs - old_log_probs_batch)

                surr1 = ratios * advantages_batch
                surr2 = torch.clamp(ratios, 1.0 - self.config.CLIP_PARAM, 1.0 + self.config.CLIP_PARAM) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                self.pi_opt.zero_grad()
                policy_loss.backward()
                self.pi_opt.step()

                # Value loss
                values = self.v(states_batch).squeeze(-1)
                value_loss = (returns_batch - values).pow(2).mean()

                self.v_opt.zero_grad()
                value_loss.backward()
                self.v_opt.step()
