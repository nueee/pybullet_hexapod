import numpy as np
import gym
import hexapod
from stable_baselines3 import PPO
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="load model, and show how much power is consumed.")
parser.add_argument('--model1', required=True, help='zipped model path to measure.')
parser.add_argument('--model2', required=True, help='zipped model path to measure.')
parser.add_argument('--duration', required=True, help='how many timesteps to record.')
parser.add_argument('--env', required=False, default='Hexapod-v2', help='gym env id. Hexapod-v0 by default.')
args = parser.parse_args()

env = gym.make(str(args.env))
model1 = PPO.load(path=str(args.model1), env=env)
model2 = PPO.load(path=str(args.model2), env=env)

total_tau1 = []
total_tau2 = []
# torque_sampling_rate = 10 # # of torque samples per intended timestep
obs1 = env.reset()
obs2 = env.reset()
for i in range(int(args.duration)):
    # if i%torque_sampling_rate==0:
    action1, _ = model1.predict(obs1.astype(np.float32))
    action2, _ = model1.predict(obs2.astype(np.float32))
    obs1, _, done, info1 = env.step(action1)
    if done:
        print("failed")
    obs2, _, done, info2 = env.step(action2)
    # else:
    #     obs,done,info = env.non_action_step()
    if done:
        print("failed")
    total_tau1.append(np.sum(np.abs(info1['torques'])))
    total_tau2.append(np.sum(np.abs(info2['torques'])))

time = list(map(lambda t: t*env.dt, range(int(args.duration))))
print(sum(total_tau1))
print(sum(total_tau2))
plt.plot(time, total_tau1, time, total_tau2)
plt.show()
