import numpy as np
import gym
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

rendering = gym.make("HexapodRender-v0")
model = PPO.load(path='./save_model_1227/power/hexapod_model_1227power_2500000_steps', env=rendering)

# start rendering the current model.
obs = rendering.reset(load=True, fixed=False)
rendering.render()
for i in range(10000):
    action, _ = model.predict(obs.astype(np.float32))
    obs, _, done, _ = rendering.step(action)
    #print(action)
    #print(obs)
    #print(analog_to_digital(action))
    #rendering.render()
    #if done:
    #    obs = rendering.reset(load=True,fixed=False)
