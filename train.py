from src.ppo import PPO
from src.environment import Environment
from src.config import Config

def main():
    config = Config()
    env = Environment(name=config.ENV_NAME)
    ppo = PPO(config)

    # Record episode stats
    ...

    # Record episode video
    ...
    
    for ep in range(config.NUM_EPS):
        trajectory = env.run_episode(ppo)
        ppo.update(trajectory)

    env.close()

if __name__ == "__main__":
    main()
