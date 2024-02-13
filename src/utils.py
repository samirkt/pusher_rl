import torch

def process_trajectory(trajectory, gamma=0.99, lambda_gae=0.95):
    rewards = torch.tensor(trajectory.rewards, dtype=torch.float32)
    values = torch.cat(trajectory.values)
    next_values = torch.cat(trajectory.next_values)
    dones = torch.tensor(trajectory.dones, dtype=torch.float32)
    states = torch.cat(trajectory.states)
    actions = torch.stack(trajectory.actions)
    log_probs = torch.stack(trajectory.log_probs)

    deltas = rewards + gamma * next_values.squeeze() * (1 - dones) - values.squeeze()
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    gae = 0
    for t in reversed(range(len(rewards))):
        gae = deltas[t] + gamma * lambda_gae * (1 - dones[t]) * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]

    return states, actions, log_probs, returns, advantages
