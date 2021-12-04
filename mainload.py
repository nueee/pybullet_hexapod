import numpy as np
import gym
import hexapod
from typing import Callable
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EveryNTimesteps, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


#env = make_vec_env("Hexapod-v0", n_envs=4, seed=0, vec_env_cls=SubprocVecEnv)
#env = SubprocVecEnv([lambda: gym.make("Hexapod-v0")])
#env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

rendering = gym.make("Hexapod-v0")
model = PPO.load(path='./save_model_1204/A/hexapod_model_1204A_1000000_steps', env=rendering)

# start rendering the current model.
obs = rendering.reset()
#rendering.render()
for i in range(1000):
	action, _ = model.predict(obs.astype(np.float32))
	obs, _, done, _ = rendering.step(action)
	print(action)
	
	if done:
	    obs = rendering.reset()
