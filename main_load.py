import gym
import numpy as np
import hexapod
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EveryNTimesteps, CheckpointCallback


def main():
    env = gym.make("Hexapod-v0")
    model = PPO.load("./save_hexy_test/hexy_test_model_2000_steps.zip")

    obs = env.reset()
    env.render()
    while True:
        action, _ = model.predict(obs.astype(np.float32))
        obs, _, done, _ = env.step(action)
        #env.render()
        '''
        if done:
            obs = env.reset()
            time.sleep(1/60)
        '''

if __name__ == '__main__':
    main()



