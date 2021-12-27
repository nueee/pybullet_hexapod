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
import random
import math

# overall var
model_path = "../save_model_1226/power/hexapod_model_1226power_500000_steps.zip"
iterations = 1000
n = 2
alpha = [0] * n # joint_damping, force
da = 0.01
lr = 0.001
sample_rate = 10
bf_lower_limit = [-0.7, -0.3]
bf_upper_limit = [0.19, 0.5]
total_timesteps = 2000
# get command values
commandValues = [[707, 707, 707, 707, 707, 512, 316, 316, 316, 316, 707, 316, 512, 316, 316, 316, 512, 512,], [512, 512, 316, 512, 316, 316, 512, 707, 512, 512, 512, 707, 707, 512, 707, 316, 707, 512,], [707, 512, 707, 512, 707, 512, 707, 512, 316, 512, 316, 707, 316, 512, 512, 707, 316, 512,], [512, 707, 707, 707, 512, 512, 316, 316, 316, 316, 316, 512, 707, 316, 316, 316, 707, 512,], [707, 316, 316, 512, 316, 316, 707, 707, 707, 512, 512, 707, 707, 707, 512, 316, 707, 512,], [512, 707, 707, 512, 707, 512, 707, 512, 316, 707, 316, 707, 316, 512, 512, 707, 316, 512,], [707, 707, 512, 707, 316, 707, 316, 316, 512, 316, 512, 316, 707, 316, 512, 316, 707, 512,], [512, 316, 512, 316, 316, 316, 707, 707, 512, 512, 707, 707, 707, 707, 512, 707, 512, 512,], [512, 707, 707, 707, 707, 707, 512, 512, 316, 707, 316, 512, 316, 316, 512, 707, 316, 512,], [316, 512, 316, 512, 316, 316, 316, 512, 512, 512, 707, 316, 707, 512, 512, 316, 707, 707,], [316, 316, 512, 316, 512, 316, 707, 707, 512, 707, 707, 707, 316, 512, 512, 707, 316, 512,], [316, 707, 707, 707, 707, 512, 512, 316, 316, 707, 316, 316, 316, 316, 707, 707, 316, 707,], [316, 512, 316, 707, 316, 316, 316, 512, 512, 316, 707, 316, 707, 316, 512, 316, 707, 707,], [512, 512, 707, 316, 707, 316, 707, 707, 512, 707, 512, 707, 316, 512, 512, 707, 316, 512,], [707, 707, 707, 707, 512, 512, 316, 316, 316, 316, 512, 316, 512, 316, 316, 316, 512, 512,], [512, 512, 316, 512, 316, 316, 512, 707, 512, 512, 707, 707, 707, 512, 512, 316, 707, 512,], [707, 512, 707, 512, 707, 707, 707, 512, 316, 512, 316, 707, 316, 512, 512, 707, 316, 512,], [707, 707, 316, 707, 316, 707, 316, 316, 316, 316, 316, 316, 707, 316, 512, 316, 707, 512,], [512, 316, 512, 316, 316, 316, 707, 707, 512, 512, 512, 707, 512, 707, 512, 707, 316, 512,], [512, 707, 707, 707, 707, 512, 512, 316, 316, 512, 512, 316, 316, 316, 512, 512, 316, 707,]]
# deploy reference data generation from csv file or etc
SCALE = 1.5
for i in range(20):
    for j in range(18):
        commandValues[i][j] = np.floor((commandValues[i][j]-512)*SCALE+512)
        if commandValues[i][j]>=1024:
            commandValues[i][j] = 1023
        if commandValues[i][j]<0:
            commandValues[i][j] = 0
commandValues = commandValues*1000
real_data = []
for i in range(total_timesteps*sample_rate):
	temp = []
	for j in range(18):
		temp.append(commandValues[i//sample_rate][j])
	real_data.append(temp)


print(commandValues[15][5])

def out(state):
    if state[0]<bf_lower_limit[0] or state[0]>bf_upper_limit[0] or state[1]<bf_lower_limit[1] or state[1]>bf_upper_limit[1]:
        return True
    else:
        return False
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
    rendering = gym.make("Hexapod-v0")
    obs = rendering.reset(load=True,fixed=True)
    #rendering.render()
    #sample_rate = 1 # disable the method
    for i in range(total_timesteps*sample_rate):
        if i%sample_rate == 0:
            obs, _, done, _ = rendering.step(digital_to_analog(np.array(commandValues[i])))
            ret += np.sqrt(np.mean((real_data[i] - analog_to_digital(obs[:18]))**2))
        else:
            _obs, _ , _ = rendering.non_action_step()
            ret += np.sqrt(np.mean((real_data[i] - analog_to_digital(_obs[:18]))**2))

    rendering.close()
    print("value is" + str(ret/total_timesteps/sample_rate))
    return ret/total_timesteps/sample_rate


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


def simulated_annealing(initial_state):
    """Peforms simulated annealing to find a solution"""
    initial_temp = 900
    final_temp = .1
    dT = 0.01

    current_temp = initial_temp

    # Start by initializing the current state with the initial state
    current_state = initial_state
    solution = current_state

    min_state = initial_state

    while current_temp > final_temp:
        #print("now on"+str(solution))
        neighbor = get_neighbors(solution)

        # Check if neighbor is best so far
        cost_diff = get_cost(current_state) - get_cost(neighbor)

        # if the new solution is better, accept it
        if cost_diff > 0:
            solution = neighbor
            if get_cost(solution)<get_cost(min_state):
                min_state = solution
        # if the new solution is not better, accept it with a probability of e^(-cost/temp)
        else:
            if random.uniform(0, 1) < math.exp(-cost_diff / current_temp):
                solution = neighbor
        # decrement the temperature
        current_temp -= dT

    return min_state, get_cost(min_state)


def get_cost(state):
    """Calculates cost of the argument state for your solution."""
    #print("cost of state"+str(state))
    #print(state[0]**2+state[1]**2)
    return E(state)
    return (state[0]-0.2)**2+(state[1]-0.1)**2


def get_neighbors(state):
    """Returns neighbors of the argument state for your solution."""
    '''
    p = random.uniform(0,1)
    if p<0.25:
        return [state[0]+0.01,state[1]]
    elif p<0.5:
        return [state[0] - 0.01, state[1]]
    elif p<0.75:
        return [state[0], state[1] + 0.01 ]
    else:
        return [state[0], state[1] - 0.01 ]
    '''
    ret = random.choice([[state[0]+0.01,state[1]],[state[0] - 0.01, state[1]],[state[0], state[1] + 0.01 ],[state[0], state[1] - 0.01 ]])
    while out(ret):
        ret = random.choice([[state[0] + 0.01, state[1]], [state[0] - 0.01, state[1]], [state[0], state[1] + 0.01],
                             [state[0], state[1] - 0.01]])
    return ret




write_alpha([0,0])

print(E([0.2,0.1]))

print("init")
x0 = [0,0]
print(get_cost(x0))
ANS = 10000
for i in range(1):
    print("in"+str(i))
    t = simulated_annealing(x0)[1]
    print(t)
    ANS = min(t,ANS)
print(ANS)
#print(get_brute_force_alpha())

#print(get_GD_alpha())

# things to keep in mind or test
'''

initial values must be equal 

sim 2 real is not feasible => 


'''