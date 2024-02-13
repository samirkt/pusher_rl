import gymnasium as gym
import numpy as np

from src import config, environment, memory, policy

def main():
    # Initialize the Pusher environment
    env = environment.Environment(name=config.NAME)
    ppo = policy.PPO()
    
    for ep in range(config.NUM_EPS):
        traject = env.run_episode(ppo)

        # Update policy network
        ...

    env.close()

if __name__ == "__main__":
    main()
