import numpy as np
import gym
import hexapod
from stable_baselines3 import PPO
import matplotlib.pyplot as plt


render_size = 500

env = gym.make("Hexapod-v1")
model = PPO.load(path='./save_model_1229/X/hexapod_model_1229X_2200000_steps.zip', env=env)

# start rendering the current model.
obs = env.reset()
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
