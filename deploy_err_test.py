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
import pandas as pd 
from pandas import DataFrame 

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


# start rendering the current model.

df=pd.read_csv('command.csv')
print(df['sequence'][0])
Actval = []
finalstr = ""
finalstr = finalstr + "[" 
for i in range(20):
	finalstr = finalstr + str(df['sequence'][i]) + ","
finalstr = finalstr + "]"
finalstr = finalstr.replace(".",",")
finalstr = finalstr.replace("[","{")
finalstr = finalstr.replace("]","}")
print(finalstr)
try:
    ser=serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=.5)
except:
    print('Port open error')


time.sleep(5)#no delete!

for i in range(20):
	start = time.time()
         
	action = np.array(Actval[i]).astype(np.int32)
	
	ser.write(pack ('18h',action[0],action[1],action[2],action[3],action[4],action[5],action[6],action[7],action[8],action[9],action[10],action[11],action[12],action[13],action[14],action[15],action[16],action[17]))#the 
	time.sleep(1-(time.time()-start))
#the delays need, that the bytes are good order

