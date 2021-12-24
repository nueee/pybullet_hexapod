import gym
import numpy as np
import hexapod
import time
from stable_baselines3 import PPO


rendering = gym.make("HexapodRender-v0")
model = PPO.load("./save_model_1127/A/hexapod_model_1127A_1000000_steps.zip")

# start rendering the current model.
obs = rendering.reset()
for i in range(1000):
    action, _ = model.predict(obs.astype(np.float32))
    obs, _, done, _ = rendering.step(action)
    rendering.render()
    if done:
        obs = rendering.reset()




