import pandas as pd
from pandas import DataFrame
import gym
import numpy as np
import sys
sys.path.append('../')
import hexapod
import time
from stable_baselines3 import PPO

# overall var
model_path = "../save_model_1224/A/hexapod_model_1224A_50000_steps.zip"
iterations = 1000
n = 2
alpha = [0] * n
da = 0.01
bf_lower_limit = [0.001, 1]
bf_upper_limit = [0.9, 1.5]
total_timesteps = 20
# deploy reference data generation from csv file or etc
real_data = []
for i in range(total_timesteps):
	temp = []
	for j in range(18):
		temp.append(j)
	real_data.append(temp)

# get command values
commandValues = [[707, 707, 707, 707, 707, 512, 316, 316, 316, 316, 707, 316, 512, 316, 316, 316, 512, 512,], [512, 512, 316, 512, 316, 316, 512, 707, 512, 512, 512, 707, 707, 512, 707, 316, 707, 512,], [707, 512, 707, 512, 707, 512, 707, 512, 316, 512, 316, 707, 316, 512, 512, 707, 316, 512,], [512, 707, 707, 707, 512, 512, 316, 316, 316, 316, 316, 512, 707, 316, 316, 316, 707, 512,], [707, 316, 316, 512, 316, 316, 707, 707, 707, 512, 512, 707, 707, 707, 512, 316, 707, 512,], [512, 707, 707, 512, 707, 512, 707, 512, 316, 707, 316, 707, 316, 512, 512, 707, 316, 512,], [707, 707, 512, 707, 316, 707, 316, 316, 512, 316, 512, 316, 707, 316, 512, 316, 707, 512,], [512, 316, 512, 316, 316, 316, 707, 707, 512, 512, 707, 707, 707, 707, 512, 707, 512, 512,], [512, 707, 707, 707, 707, 707, 512, 512, 316, 707, 316, 512, 316, 316, 512, 707, 316, 512,], [316, 512, 316, 512, 316, 316, 316, 512, 512, 512, 707, 316, 707, 512, 512, 316, 707, 707,], [316, 316, 512, 316, 512, 316, 707, 707, 512, 707, 707, 707, 316, 512, 512, 707, 316, 512,], [316, 707, 707, 707, 707, 512, 512, 316, 316, 707, 316, 316, 316, 316, 707, 707, 316, 707,], [316, 512, 316, 707, 316, 316, 316, 512, 512, 316, 707, 316, 707, 316, 512, 316, 707, 707,], [512, 512, 707, 316, 707, 316, 707, 707, 512, 707, 512, 707, 316, 512, 512, 707, 316, 512,], [707, 707, 707, 707, 512, 512, 316, 316, 316, 316, 512, 316, 512, 316, 316, 316, 512, 512,], [512, 512, 316, 512, 316, 316, 512, 707, 512, 512, 707, 707, 707, 512, 512, 316, 707, 512,], [707, 512, 707, 512, 707, 707, 707, 512, 316, 512, 316, 707, 316, 512, 512, 707, 316, 512,], [707, 707, 316, 707, 316, 707, 316, 316, 316, 316, 316, 316, 707, 316, 512, 316, 707, 512,], [512, 316, 512, 316, 316, 316, 707, 707, 512, 512, 512, 707, 512, 707, 512, 707, 316, 512,], [512, 707, 707, 707, 707, 512, 512, 316, 316, 512, 512, 316, 316, 316, 512, 512, 316, 707,]]


print(commandValues[15][5])

def analog_to_digital(x):
    r = (x + 2.62) * 1023 / 5.24
    return np.floor(r + 0.5)


def write_list_to_csv(x):
    data = {"sequence": x}
    data_df = DataFrame(data)
    print(data_df)
    data_df.to_csv('command.csv')


rendering = gym.make("HexapodRender-v0")
model = PPO.load(model_path)



def E(a):  # temp linear function
    N = len(a)
    ret = 0
    rendering = gym.make("HexapodRender-v0")
    model = PPO.load(model_path)
    obs = rendering.reset(load=True)
    rendering.render()
    for i in range(total_timesteps):
        print("timestep E calc" + str(i))
        action, _ = model.predict(commandValues[i])
        obs, _, done, _ = rendering.step(action)
        print(obs)
        ret += np.rms(real_data[i] - obs)

        # print(action)
        # print(obs)
        # print(analog_to_digital(action))

    return ret/N


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
    bestval = 0
    bestalpha = [0] * n
    for i in range(samples):
        for j in range(n):
            nowalpha[j] = bf_lower_limit[j] + np.random.random() * (bf_upper_limit[j] - bf_lower_limit[j])
            Eval = E(a)
            if i == 0:
                bestval = Eval
                bestalpha = nowalpha
            else:
                if bestval > Eval:
                    bestval = Eval
                    bestalpha = nowalpha

    return bestval, bestalpha


print(E([0.5,0.5]))


# things to keep in mind
'''

initial values must be equal 


'''