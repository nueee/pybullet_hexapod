import numpy as np
import gym
import hexapod
from typing import Callable
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EveryNTimesteps, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


def lin_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * (initial_value - final_value) + final_value
    return func


checkpoint_on_event = CheckpointCallback(
    save_freq=1,
    save_path='./save_model_{trial_name}/',
    verbose=2,
    name_prefix='hexapod_model_{trial_name}'
)
event_callback = EveryNTimesteps(
    n_steps=int(1e4),  # every n_steps, save the model
    callback=checkpoint_on_event
)


def main():
    env = make_vec_env("Hexapod-v0", n_envs=4, seed=0, vec_env_cls=SubprocVecEnv)
    # env = SubprocVecEnv([lambda: gym.make("Hexapod-v0")])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO(
        "MlpPolicy",
        env=env,
        verbose=2,
        tensorboard_log="./tb_log_{trial_name}",
        learning_rate=lin_schedule(3e-4, 3e-6),
        clip_range=lin_schedule(0.3, 0.1),
        n_epochs=20,  # PPO internal epochs
        ent_coef=1e-4,
        batch_size=2048,
        n_steps=512
    )

    # if you need to continue learning by loading existing model, use below line.
    # model = PPO.load(path='{existing model path...}', env=env)

    model.learn(
        int(1e4),  # total timesteps used for learning
        callback=event_callback,  # every n_steps, save the model.
        tb_log_name='tb_{trial_name}'
        # ,reset_num_timesteps=False   # if you need to continue learning by loading existing model, use this option.
    )

    env.close()

    rendering = gym.make("HexapodRender-v0")

    # start rendering the current model.
    obs = rendering.reset()
    for i in range(1000):
        action, _ = model.predict(obs.astype(np.float32))
        obs, _, done, _ = rendering.step(action)
        rendering.render()
        if done:
            obs = rendering.reset()


if __name__ == '__main__':
    main()
