import gymnasium as gym

class Environment:
    def __init__(self, name):
        self.env = gym.make(name, render_mode="human")

    def run_episode(self, policy):
        """Runs a single episode using the specified policy function."""
        observation = self.env.reset()
        truncated, terminated = False, False
        traject = []

        while not (truncated or terminated):
            # Select an action based on the current observation
            action = policy.sample(observation, self.env)

            # Step the environment forward and get new information
            observation, reward, terminated, truncated, info = self.env.step(action)
            traject.append([observation, reward, terminated, truncated, info, action])

        return traject
    
    def close(self):
        self.env.close()
