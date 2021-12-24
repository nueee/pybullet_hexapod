import pandas as pd
from pandas import DataFrame
import gym
import numpy as np
import hexapod
import time
from stable_baselines3 import PPO

# overall var
iterations = 1000
n = 2
alpha = [0] * n
da = 0.01
bf_lower_limit = [0.001, 1]
bf_upper_limit = [0.9, 1.5]
# deploy reference data generation from csv file or etc
real_data = []
for i in range(20):
	temp = []
	for j in range(18):
		temp.append(j)
	real_data.append(temp)





def analog_to_digital(x):
    r = (x + 2.62) * 1023 / 5.24
    return np.floor(r + 0.5)


def write_list_to_csv(x):
    data = {"sequence": x}
    data_df = DataFrame(data)
    print(data_df)
    data_df.to_csv('command.csv')


rendering = gym.make("HexapodRender-v0")
model = PPO.load("./save_model_1223/A/hexapod_model_1223A_100000_steps.zip")


def E(a):  # temp linear function

    N = len(a)
    ret = 0
    for i in range(N):
        ret += a[i]
    return ret


def GradE(a):  # get the estimated value of gradient
    N = len(a)
    ret = [0] * N
    for i in range(N):
        a[i] += da
        ret[i] = E(a)
        a[i] -= da
        ret[i] = (ret[i] - E(a)) / da
    return ret


def get_brute_force_alpha():
    samples = 100
    nowalpha = [0] * n
    for i in range(samples):
        for j in range(n):
            nowalpha[j] = bf_lower_limit[j] + np.random.random() * (bf_upper_limit[j] - bf_lower_limit[j])


for i in range(iterations):

# start rendering the current model.
obs = rendering.reset()
fullActionData = []
i = 0
SAMPLE_RATE = 5
DESIRED_TIMESTEPS = 20
while True:
    i += 1
    action, _ = model.predict(obs.astype(np.float32))
    print(action)
    # time.sleep(0.2)
    if i % SAMPLE_RATE == 0:
        fullActionData.append(analog_to_digital(action.astype(np.int32)))
    obs, _, done, _ = rendering.step(action)
    if done:
        i = 0
        fullActionData = []
        obs = rendering.reset()
    if i > SAMPLE_RATE * DESIRED_TIMESTEPS:
        break
    print("i" + str(i))
print("action 20 data")
print(len(fullActionData))

write_list_to_csv(fullActionData)