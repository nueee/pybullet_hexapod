import numpy as np
import gym
import hexapod
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import time


render_size = 500

env = gym.make("Hexapod-v2")
model = PPO.load(path='./save_model/save_model_1229/restrictedJoint_branch4/hexapod_model_1229restrictedJoint_branch4_6000012_steps.zip', env=env)

# start rendering the current model.
obs = env.reset()
frame = plt.imshow(np.zeros((render_size, render_size, 4)))
plt.draw()
plt.pause(1e-5)
time.sleep(5)
for i in range(10000):
    action, _ = model.predict(obs.astype(np.float32))
    obs, _, done, _ = env.step(action)
    frame.set_data(env.render(render_size=render_size))
    plt.draw()
    plt.pause(1e-5)
    if done:
        print("hexapod died")
        # obs = env.reset(load=True, fixed=False)
