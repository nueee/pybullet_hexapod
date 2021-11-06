import gym
import numpy as np
import hexapod
import time
from stable_baselines3 import PPO


def main():
    env = gym.make("Hexapod-v0")
    model = PPO(
        "MlpPolicy",
        env=env,
        verbose=2,
    )

    model.learn(int(1e3))

    obs = env.reset()
    while True:
        action, _ = model.predict(obs.astype(np.float32))
        obs, _, done, _ = env.step(action)
        env.render()
        if done:
            obs = env.reset()
            time.sleep(1/60)


if __name__ == '__main__':
    main()
