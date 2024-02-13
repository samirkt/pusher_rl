import gymnasium as gym
import torch

class Trajectory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.next_values = []
        self.dones = []

    def add(self, state, action, reward, value, log_prob, next_value, done):
        self.states.append(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.next_values.append(next_value)
        self.dones.append(done)

class Environment:
    def __init__(self, name):
        self.env = gym.make(name, render_mode="human")

    def run_episode(self, policy):
        state = self.env.reset()
        done = False
        trajectory = Trajectory()

        while not done:
            # TODO: abstract action sampling (and state to tensor conversion?) --> into ppo.py
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            mean, std_dev = policy.policy(state_tensor)
            dist = torch.distributions.Normal(mean, std_dev)
            action = dist.sample()
            # TODO: is this the right place to compute log_prob? --> into utils.py
            log_prob = dist.log_prob(action).sum()

            value = policy.value(state_tensor)

            next_state, reward, terminated, truncated, _ = self.env.step(action.numpy())
            done = terminated or truncated
            # TODO: abstract next value calculation --> into utils.py
            next_value = policy.value(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)) if not done else torch.tensor([[0.0]])

            # TODO: cleanup trajectory calss info, should contain minimum raw data
            trajectory.add(
                state=state,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                next_value=next_value,
                done=done
            )

            state = next_state

        return trajectory

    def close(self):
        self.env.close()
