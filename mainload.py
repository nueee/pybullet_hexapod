import numpy as np
import gym
import pybullet

import hexapod
from typing import Callable
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EveryNTimesteps, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import time

def analog_to_digital(x):
    r = (x + 2.62)*1023/5.24
    return np.floor(r+0.5)

#env = make_vec_env("Hexapod-v0", n_envs=4, seed=0, vec_env_cls=SubprocVecEnv)
#env = SubprocVecEnv([lambda: gym.make("Hexapod-v0")])
#env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

rendering = gym.make("HexapodRender-v2")
model = PPO.load(path='./save_model/save_model_1229/restrictedJoint_branch4/hexapod_model_1229restrictedJoint_branch4_6000012_steps.zip', env=rendering)

pybullet.setRealTimeSimulation(True)
# start rendering the current model.
obs = rendering.reset()
rendering.render()
for i in range(10000):
    action, _ = model.predict(obs.astype(np.float32))
    obs, _, done, _ = rendering.step(action)
    time.sleep(.05)
    #print(action)
    #print(obs)
    #print(analog_to_digital(action))
    #rendering.render()
    # if done:
    #    obs = rendering.reset(load=True,fixed=False)
