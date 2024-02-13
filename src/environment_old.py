import gymnasium as gym
import torch


class Trajectory:
    def __init__(self):
        self.traj = {
            "state": [],
            "action": [],
            "reward": [],
            "value": [],
            "log_prob": [],
            "next_value": [],
            "done": [],
        }
    
    def add(self, state, action, reward, value, log_prob, next_value, done):
        self.traj["state"].append(state)
        self.traj["action"].append(action)
        self.traj["reward"].append(reward)
        self.traj["value"].append(value)
        self.traj["log_prob"].append(log_prob)
        self.traj["next_value"].append(next_value)
        self.traj["done"].append(done)

    def process(self, gamma=0.99, lambda_gae=0.95):
        states = self.traj["state"]
        actions = self.traj["action"]
        log_probs = self.traj["log_prob"]
        rewards = self.traj["reward"]
        values = self.traj["value"]
        next_values = self.traj["next_value"]
        dones = self.traj["done"]

        # Convert lists to tensors
        rewards = torch.tensor(rewards, dtype=torch.float32)
        values = torch.cat(values)
        next_values = torch.cat(next_values)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # Calculate returns and advantages
        # TODO: what is this doing? Generalized advantage estimation paper...
        deltas = rewards + gamma * next_values.squeeze() * (1 - dones) - values.squeeze()
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = 0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + gamma * lambda_gae * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return torch.stack(states), torch.cat(actions), torch.cat(log_probs), returns, advantages


class Environment:
    def __init__(self, name):
        self.env = gym.make(name, render_mode="human")

    def run_episode(self, policy):
        """Runs a single episode using the specified policy function."""
        state = self.env.reset()
        done = False
        traject = Trajectory()

        while not done:
            state_tensor = torch.FloatTensor(state[0]).unsqueeze(0)
            mean, std_dev = policy.policy(state_tensor)
            dist = torch.distributions.Normal(mean, std_dev)
            action = dist.sample()[0]
            log_prob = dist.log_prob(action)

            value = policy.value(state_tensor)

            # Step the environment forward and get new information
            next_state, reward, terminated, truncated, info = self.env.step(action)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            done = terminated or truncated
            next_value = policy.value(next_state_tensor) if not done else torch.tensor([[0.0]])

            traject.add(
                state=state,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                next_value=next_value,
                done=done
            )

            state = next_state

        return traject
    
    def close(self):
        self.env.close()
