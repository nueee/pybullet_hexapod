import pandas as pd
from pandas import DataFrame
import gym
import numpy as np
import sys
sys.path.append('../')
import hexapod
import time
from stable_baselines3 import PPO
import configparser

# overall var
model_path = "../save_model_1225/A/hexapod_model_1225A_500000_steps.zip"
iterations = 1000
n = 2
alpha = [0] * n # joint_damping, force
da = 0.01
lr = 0.001
bf_lower_limit = [-0.7, -0.3]
bf_upper_limit = [0.19, 0.5]
total_timesteps = 20000
# get command values
commandValues = [[707, 707, 707, 707, 707, 512, 316, 316, 316, 316, 707, 316, 512, 316, 316, 316, 512, 512,], [512, 512, 316, 512, 316, 316, 512, 707, 512, 512, 512, 707, 707, 512, 707, 316, 707, 512,], [707, 512, 707, 512, 707, 512, 707, 512, 316, 512, 316, 707, 316, 512, 512, 707, 316, 512,], [512, 707, 707, 707, 512, 512, 316, 316, 316, 316, 316, 512, 707, 316, 316, 316, 707, 512,], [707, 316, 316, 512, 316, 316, 707, 707, 707, 512, 512, 707, 707, 707, 512, 316, 707, 512,], [512, 707, 707, 512, 707, 512, 707, 512, 316, 707, 316, 707, 316, 512, 512, 707, 316, 512,], [707, 707, 512, 707, 316, 707, 316, 316, 512, 316, 512, 316, 707, 316, 512, 316, 707, 512,], [512, 316, 512, 316, 316, 316, 707, 707, 512, 512, 707, 707, 707, 707, 512, 707, 512, 512,], [512, 707, 707, 707, 707, 707, 512, 512, 316, 707, 316, 512, 316, 316, 512, 707, 316, 512,], [316, 512, 316, 512, 316, 316, 316, 512, 512, 512, 707, 316, 707, 512, 512, 316, 707, 707,], [316, 316, 512, 316, 512, 316, 707, 707, 512, 707, 707, 707, 316, 512, 512, 707, 316, 512,], [316, 707, 707, 707, 707, 512, 512, 316, 316, 707, 316, 316, 316, 316, 707, 707, 316, 707,], [316, 512, 316, 707, 316, 316, 316, 512, 512, 316, 707, 316, 707, 316, 512, 316, 707, 707,], [512, 512, 707, 316, 707, 316, 707, 707, 512, 707, 512, 707, 316, 512, 512, 707, 316, 512,], [707, 707, 707, 707, 512, 512, 316, 316, 316, 316, 512, 316, 512, 316, 316, 316, 512, 512,], [512, 512, 316, 512, 316, 316, 512, 707, 512, 512, 707, 707, 707, 512, 512, 316, 707, 512,], [707, 512, 707, 512, 707, 707, 707, 512, 316, 512, 316, 707, 316, 512, 512, 707, 316, 512,], [707, 707, 316, 707, 316, 707, 316, 316, 316, 316, 316, 316, 707, 316, 512, 316, 707, 512,], [512, 316, 512, 316, 316, 316, 707, 707, 512, 512, 512, 707, 512, 707, 512, 707, 316, 512,], [512, 707, 707, 707, 707, 512, 512, 316, 316, 512, 512, 316, 316, 316, 512, 512, 316, 707,]]
# deploy reference data generation from csv file or etc
SCALE = 2
for i in range(20):
    for j in range(18):
        commandValues[i][j] = np.floor((commandValues[i][j]-512)*SCALE+512)
        if commandValues[i][j]>=1024:
            commandValues[i][j] = 1023
        if commandValues[i][j]<0:
            commandValues[i][j] = 0
commandValues = commandValues*1000
real_data = []
for i in range(total_timesteps):
	temp = []
	for j in range(18):
		temp.append(commandValues[i][j])
	real_data.append(temp)


print(commandValues[15][5])

def write_alpha(a):
    edit = configparser.ConfigParser()
    edit.read("../configfile.ini")
    alpha = edit["alpha"]
    alpha["joint_damping"] = str(a[0])
    print("changed to "+str(a[0]))
    alpha["force"] = str(a[1])
    with open('../configfile.ini', 'w') as configfile:
        edit.write(configfile)


def analog_to_digital(x):
    r = (x + 2.62) * 1023 / 5.24
    return np.floor(r + 0.5)

def digital_to_analog(x):
	return (x*5.24/1023)-2.62


def write_list_to_csv(x):
    data = {"sequence": x}
    data_df = DataFrame(data)
    print(data_df)
    data_df.to_csv('command.csv')


#rendering = gym.make("HexapodRender-v0")
#model = PPO.load(model_path)



def E(a):  # temp linear function
    N = len(a)
    write_alpha(a) # upd alpha
    ret = 0
    rendering = gym.make("HexapodRender-v0")
    obs = rendering.reset(load=True,fixed=True)
    #rendering.render()
    for i in range(total_timesteps):
        #print("obs")
        #print(obs)
        #print("timestep E calc" + str(i))
        #print(commandValues[i])
        obs, _, done, _ = rendering.step(digital_to_analog(np.array(commandValues[i])))
        #print(obs)
        ret += np.sqrt(np.mean((real_data[i] - analog_to_digital(obs[:18]))**2))
        #print("expected")
        #print(real_data[i])
        #print("real")
        #print(analog_to_digital(obs[:18]))

        # print(action)
        # print(obs)
        # print(analog_to_digital(action))
    rendering.close()
    print("value is" + str(ret/total_timesteps))
    return ret/total_timesteps


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
        print("testing on "+str(i+1) + "th alpha")
        for j in range(n):
            nowalpha[j] = bf_lower_limit[j] + np.random.random() * (bf_upper_limit[j] - bf_lower_limit[j])
        Eval = E(nowalpha)
        if i == 0:
            bestval = Eval
            bestalpha = nowalpha
        else:
            if bestval > Eval:
                bestval = Eval
                bestalpha = nowalpha

    return bestval, bestalpha


def get_GD_alpha(a0):
    nowalpha = np.array(a0) # init

    for i in range(5):
        nowalpha = nowalpha - lr*np.array(GradE(nowalpha))
    return nowalpha,E(nowalpha)


write_alpha([0,0])

print(E([0.2,0.1]))

#print(get_brute_force_alpha())

#print(get_GD_alpha())

# things to keep in mind or test
'''

initial values must be equal 

sim 2 real is not feasible 


'''