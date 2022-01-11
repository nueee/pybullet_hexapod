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


rendering = gym.make("HexapodRender-v2")
model = PPO.load(path='./save_model_0101/long_batch_without_ST_maxtorque_revised_lr/hexapod_model_0101long_batch_without_ST_maxtorque_revised_lr_8600000_steps.zip', env=rendering)

pybullet.setRealTimeSimulation(True)
# start rendering the current model.
obs = rendering.reset()
rendering.render()
for i in range(10000):
    action, _ = model.predict(obs.astype(np.float32))
    obs, _, done, _ = rendering.step(action)
    time.sleep(0.05)
    # if done:
    #    obs = rendering.reset(load=True,fixed=False)
