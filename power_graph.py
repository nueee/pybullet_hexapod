import numpy as np
import gym
import hexapod
from stable_baselines3 import PPO
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="load model, and show how much power is consumed.")
parser.add_argument('--model', required=True, help='zipped model path to measure.')
parser.add_argument('--duration', required=True, help='how many timesteps to record.')
parser.add_argument('--env', required=False, default='Hexapod-v0', help='gym env id. Hexapod-v0 by default.')
args = parser.parse_args()

env = gym.make(str(args.env))
model = PPO.load(path=str(args.model), env=env)

total_tau = []

obs = env.reset()
for i in range(int(args.duration)):
    action, _ = model.predict(obs.astype(np.float32))
    obs, _, done, info = env.step(action)
    if done:
        obs = env.reset()
    total_tau.append(np.sum(np.abs(info['torques'])))

time = list(map(lambda t: t*env.dt, range(int(args.duration))))
plt.plot(time, total_tau)
plt.show()
