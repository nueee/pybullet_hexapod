import pandas as pd 
from pandas import DataFrame 
import gym
import numpy as np
import hexapod
import time
from stable_baselines3 import PPO

def analog_to_digital(x):
	r = (x + 2.62)*1023/5.24
	return np.floor(r+0.5)  

def write_list_to_csv(x):	
	data={"sequence":x} 
	data_df=DataFrame(data)
	print(data_df)
	data_df.to_csv('command.csv')
	

rendering = gym.make("HexapodRender-v0")
model = PPO.load("./save_model_1212/A/hexapod_model_1212A_10000000_steps.zip")

# start rendering the current model.
obs = rendering.reset()
fullActionData=[]
i=0
SAMPLE_RATE=5
DESIRED_TIMESTEPS=20
while True:
	i+=1
	action, _ = model.predict(obs.astype(np.float32))
	print(action)
	#time.sleep(0.2)
	if i%SAMPLE_RATE==0:
		fullActionData.append(analog_to_digital(action.astype(int))) 
	obs, _, done, _ = rendering.step(action)
	if done:
	    i=0
	    fullActionData=[]
	    obs = rendering.reset()
	if i>SAMPLE_RATE*DESIRED_TIMESTEPS:
	    break
	print("i" + str(i))
print("action 20 data") 
print(len(fullActionData))	    

write_list_to_csv(fullActionData)
