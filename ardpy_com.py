#Arduino forum 2020 - https://forum.arduino.cc/index.php?topic=714968 
import serial
from struct import *
import sys
import time
import random
import ast
import numpy as np
import gym
import hexapod
from typing import Callable
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EveryNTimesteps, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

def analog_to_digital(x):
	r = (x + 2.62)*1023/5.24
	return np.floor(r+0.5)  

def digital_to_analog(x): 
	return (x*5.24/1023)-2.62
	
def _readline(serial): # https://coderedirect.com/questions/211612/using-pyserial-is-it-possible-to-wait-for-data
    eol = b'r'
    leneol = len(eol)
    line = bytearray()
    while True:
        c = serial.read(1)
        if c:
            line += c
            if line[-leneol:] == eol:
                break
        else:
            break
    return bytes(line)

rendering = gym.make("Hexapod-v0")
model = PPO.load(path='./save_model_1204/A/hexapod_model_1204A_1000000_steps', env=rendering)

# start rendering the current model.
obs = np.zeros(108)
jnt_buffer = np.zeros((3,18),dtype=np.float32)
act_buffer = np.zeros((3,18),dtype=np.float32)
try:
    ser=serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=.5)
except:
    print('Port open error')


time.sleep(5)#no delete!
while True:
    try:
        #print("from") 
        #print(obs.astype(np.float32)) 
        action, _ = model.predict(obs.astype(np.float32))
        action = analog_to_digital(action)
        #print("to")
        #print(action)
        #print("action" +  str(action))
        action = action.astype(np.int32)
        #print("actionint" +  str(action))
        
        print("send")
        print(action)
        ser.write(pack ('18h',action[0],action[1],action[2],action[3],action[4],action[5],action[6],action[7],action[8],action[9],action[10],action[11],action[12],action[13],action[14],action[15],action[16],action[17]))#the 15h is 15 element, and h is an int type data
                                                                    #random test, that whether data is updated
                                                                    
        #time.sleep(.01)#delay
        
        dat = _readline(ser)
        #dat = dat.decode("utf-8")
        
        #read a line data
        print("got")
        
        if dat!=b''and dat!=b'\r\n':
            try:                #convert in list type the readed data
                dats=str(dat)
                dat1=dats.replace("b","")
                dat2=dat1.replace("'",'')
                dat3=dat2[:-4]
                list_=ast.literal_eval(dat3) #list_ value can you use in program
                #print("val")
                #print(list_)
                #obs, _, done, _ = rendering.step(action.astype(np.float32))
                # update obs buffer and act buffer
                jnt_buffer[1:] = jnt_buffer[:-1]
                jnt_buffer[0] = np.array(digital_to_analog(np.array(list_).astype(np.int32))) # get recent joint values
                act_buffer[1:] = act_buffer[:-1]
                act_buffer[0] = digital_to_analog(action) # get recent action
                obs = np.concatenate([jnt_buffer,act_buffer])
                #print("finalobs")
                # obs (6,18) -> (108,)
                obs = np.reshape(obs,(108,))
                #print(obs) 
              
                
                

            except:
                print('Error in corvert, readed: ', dats)
            
            #jnt_buffer[0] = np.array(digital_to_analog(np.array(list_).astype(np.int32)))
        time.sleep(.05)
    except KeyboardInterrupt:
        break
    except:
        print(str(sys.exc_info())) #print error
        break

#the delays need, that the bytes are good order

