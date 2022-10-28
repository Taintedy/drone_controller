from gym import Env
from gym.spaces import Box 
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Callable
import random
import os
from stable_baselines3.common.utils import set_random_seed


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
        self.observation_space = Box(low=-2, high=2, shape=(6,), dtype=np.float32)
        
        self.drone_state = np.array([random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0),0.0 ,0.0])
        self.target = np.array ([random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0)])

        self.state = np.concatenate((self.drone_state, self.target), dtype=np.float32)
        
        self.episode_duration = 1024
        self.time_left = self.episode_duration
        self.alive_reward = 0
        self.action_penality = 0
        self.pose_penality = 2
        
    def step(self, action):

        dt = 0.1
        vel = action * 0.5
        prev_state = self.drone_state.copy()
        self.drone_state = np.array([ 
            np.clip(self.drone_state[0] + vel[0] * dt, -2.0, 2.0),
            np.clip(self.drone_state[1] + vel[1] * dt, -2.0, 2.0),
            vel[0],
            vel[1]])
        

        self.state = np.concatenate((self.drone_state, self.target), dtype=np.float32)

        self.time_left -= 1
        reward = 0
        if self.time_left <= 0 or np.linalg.norm(self.target - self.drone_state[:2]) <= 0.01:
            if np.linalg.norm(self.target - self.drone_state[:2]) <= 0.01:
                reward += 100 + self.time_left
            done = True
        else:
            done = False
        
        prev_dist_to_goal = np.linalg.norm(self.target - prev_state[:2])
        current_dist_to_goal = np.linalg.norm(self.target - self.drone_state[:2])
        reward += self.pose_penality * (prev_dist_to_goal - current_dist_to_goal)
        info = {}
        return self.state, reward, done, info

    def render(self):
        pass
    
    def reset(self):
        self.drone_state = np.array([random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0),0.0 ,0.0])
        self.target = np.array ([random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0)])
        self.state = np.concatenate((self.drone_state, self.target), dtype=np.float32)
        self.time_left = self.episode_duration
        return self.state


if __name__ == "__main__":
        log_dir = "tmp/"
        os.makedirs(log_dir, exist_ok=True)
        num_cpu = 12  # Number of processes to use
        # Create the vectorized environment
        env = DummyVecEnv([make_env(i) for i in range(num_cpu)])
        # policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[16, dict(pi=[128, 128, 128], vf=[128, 128, 128])])
        policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[dict(pi=[256, 128, 64], vf=[256, 128, 64])])
        # model = PPO.load("PPO", env=env, device="cpu")
        model = PPO(
            'MlpPolicy', 
            env, 
            verbose=1, 
            device='cpu', 
            policy_kwargs=policy_kwargs, 
            learning_rate=schedule(2.75e-04), 
            clip_range=0.3,
            ent_coef=12.5e-07, 
            n_steps=2048, 
            batch_size=128, 
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.925,
            vf_coef=0.5,
            max_grad_norm=0.5,
            )
        
        timesteps = 500000
        model.learn(timesteps, progress_bar=True)
        model.save('PPO')
        env = DronSimEnv()
        model = PPO.load("PPO", env=env, device='cpu')
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=20)
        print(mean_reward, std_reward)
        
        

        dists = []
        scores = []
        episodes = 20
        obs = env.reset()
        for episode in range(1, episodes+1):
            state = env.reset()
            done = False
            score = 0 
            print(state)
            while not done:
                action, _states = model.predict(state)
                n_state, reward, done, info = env.step(action)
                score+=reward
            print(n_state)
            print(np.linalg.norm(n_state[4:] - n_state[:2]))
            dists.append(np.linalg.norm(n_state[4:] - n_state[:2]))
            scores.append(score)
            print('Episode:{} Score:{}'.format(episode, score))
        
        print(f"avg_dist = {sum(dists)/episodes} var = {np.var(dists)}")
        print(f"avg_score = {sum(scores)/episodes} var = {np.var(scores)}")
        print(f"passed test: {sum([1 for i in dists if i <= 0.01 ])/episode}")


