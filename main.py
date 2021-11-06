import gym
import simple_driving
import time
from stable_baselines3 import PPO

def main():
    env = gym.make("SimpleDriving-v0")
    model = PPO(
        "MlpPolicy",
        env=env,
        verbose=2,
    )

    model.learn(int(1e5))

    ob = env.reset()
    while True:
        action, _ = model.predict(ob)
        ob, _, done, _ = env.step(action)
        env.render()
        if done:
            ob = env.reset()
            time.sleep(1/30)


if __name__ == '__main__':
    main()
