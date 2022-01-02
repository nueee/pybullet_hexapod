import hexapod
from typing import Callable
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EveryNTimesteps, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize


def lin_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * (initial_value - final_value) + final_value

    return func


date = "0101"
trial = "long_batch_with_ST_value_revised_lr"

checkpoint_on_event = CheckpointCallback(
    save_freq=1,
    save_path='./save_model_' + date + '/' + trial,
    verbose=2,
    name_prefix='hexapod_model_' + date + trial
)
event_callback = EveryNTimesteps(
    n_steps=int(2e5),  # every n_steps, save the model
    callback=checkpoint_on_event
)


def main():
    env = make_vec_env("Hexapod-v2", n_envs=4, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO(
        "MlpPolicy",
        env=env,
        verbose=2,
        tensorboard_log='./tb_log_' + date,
        learning_rate=lin_schedule(5e-4, 5e-6),
        clip_range=lin_schedule(0.3, 0.1),
        n_epochs=20,  # PPO internal epochs
        ent_coef=1e-4,
        batch_size=1024 * 4,
        n_steps=1024
    )

    # if you need to continue learning by loading existing model, use below line.
    # model = PPO.load(path='{existing model path...}', env=env)

    for i in range(10):
        model.learn(
            int(2e6),  # total timesteps used for learning
            callback=event_callback,  # every n_steps, save the model.
            tb_log_name='tb_' + date + trial,
            reset_num_timesteps=False   # if you need to continue learning by loading existing model, use this option.
        )

    env.close()


if __name__ == '__main__':
    main()
