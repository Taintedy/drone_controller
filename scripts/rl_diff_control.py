from math import dist
from gym import Env
from gym.spaces import Box 
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Callable
import random
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
import os
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback

thresh = 0.1
def schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return np.arctan(progress_remaining)/(np.pi/2) * initial_value

    return func

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = DronSimEnv()
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

class DronSimEnv(Env):
    def __init__(self):
        super(DronSimEnv, self).__init__()

        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=-2, high=2, shape=(4,), dtype=np.float32)
        
        self.drone_state = np.array([random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0)])
        self.target = np.array ([random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0)])

        self.state = np.concatenate((self.drone_state, self.target), dtype=np.float32)
        
        self.episode_duration = 512
        self.time_left = self.episode_duration
        self.alive_reward = 0
        self.action_penality = 0
        self.pose_penality = 0.5
        self.sum_rewards = 0
        self.prev_reward = 0
        
    def step(self, action):
        global thresh
        dt = 0.1
        self.drone_state = self.drone_state + action * dt
        observation = np.concatenate((self.drone_state, self.target))
        observation = np.clip(observation, self.observation_space.low, self.observation_space.high)
        self.time_left -= 1
        current_dist_to_goal = np.linalg.norm(observation[:2] - observation[2:])
        reward = -(current_dist_to_goal)

        done = False
        if self.time_left <= 0:
            done = True

        info = {}
        self.sum_rewards += reward

        return observation, reward, done, info

    def render(self):
        pass
    
    def reset(self):
        self.drone_state = np.array([random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0)])
        self.target = np.array ([random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0)])
        self.time_left = self.episode_duration
        self.prev_reward = self.sum_rewards
        self.sum_rewards = 0
        observation = np.concatenate((self.drone_state, self.target))
        return observation


if __name__ == "__main__":
        num_cpu = 16  # Number of processes to use
        env = DronSimEnv()
        # Create the vectorized environment
        # env = DummyVecEnv([make_env(i) for i in range(num_cpu)])
        # policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[16, dict(pi=[128, 128, 128], vf=[128, 128, 128])])
        # policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch=[dict(pi=[64, 64], vf=[64, 64])]
        model = PPO(
            'MlpPolicy', 
            env, 
            verbose=1, 
            device='cpu',
            gamma=0.995,
            tensorboard_log="./PPO_Logs/"
            )

        timesteps = 600000

        class TensorboardCallback(BaseCallback):
            """
            Custom callback for plotting additional values in tensorboard.
            """

            def __init__(self, verbose=0):
                super(TensorboardCallback, self).__init__(verbose)

            def _on_step(self) -> bool:
                # Log scalar value (here a random variable)
                self.logger.record("reward_value", self.training_env.envs[0].prev_reward)
                return True

        model.learn(timesteps, progress_bar=True, tb_log_name="first_run", callback=TensorboardCallback())
        # model.save('PPO')


        # env = DronSimEnv()
        # model = PPO.load("PPO", env=env, device='cpu')
        dists = []
        scores = []
        episodes = 20
        obs = env.reset()
        for episode in range(1, episodes+1):
            n_state = env.reset()
            done = False
            score = 0 
            print(n_state)
            while not done:
                action, _states = model.predict(n_state)
                n_state, reward, done, info = env.step(action)
                score+=reward
            print(n_state)
            print(np.linalg.norm(n_state[:2] - n_state[2:]))
            dists.append(np.linalg.norm(n_state[:2] - n_state[2:]))
            scores.append(score)
            print('Episode:{} Score:{}'.format(episode, score))
        
        avg_dist = sum(dists)/len(dists)
        passed_test = sum([1 for i in dists if i <= thresh ])/episode
        print(f"avg_dist = {avg_dist} var = {np.var(dists) ** 0.5}")
        print(f"avg_score = {sum(scores)/len(scores)} var = {np.var(scores)** 0.5}")
        print(f"passed test: {passed_test}")