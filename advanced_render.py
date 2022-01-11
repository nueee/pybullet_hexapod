import numpy as np
import gym
import hexapod
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import time


render_size = 500

env = gym.make("HexapodHexy-v2")
model = PPO.load(path='./hexy_0911_walk_a_5000000_steps.zip', env=env)
obs = env.reset()

# start rendering the current model.
frame = plt.imshow(np.zeros((render_size, render_size, 4)))

for i in range(10000):
    action, _ = model.predict(obs.astype(np.float32))
    obs, _, done, _ = env.step(action)

    frame.set_data(env.render(render_size=render_size))
    plt.draw()
    plt.pause(1e-5)
    if done:
        print("hexapod died")
        # obs = env.reset(load=True, fixed=False)
