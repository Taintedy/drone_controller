from gym import Env
from gym.spaces import Box 
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.a2c.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import torch as th
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from typing import Callable



def schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func

class DronSimEnv(Env):
    def __init__(self):
        super(DronSimEnv, self).__init__()

        self.action_space = Box(low=-1, high=1, shape=(2,))
        self.observation_space = Box(low=-20, high=20, shape=(4,), dtype=np.float64)
        
        self.drone_state = np.array([0, 0])
        self.target = np.array ([10, 10])

        self.state = np.concatenate((self.drone_state, self.target))
        
        self.episode_duration = 100
        self.time_left = self.episode_duration
        self.alive_reward = 0
        self.action_penality = 1
        self.pose_penality = 1
        
    def step(self, action):

        dt = 0.1
        self.drone_state = np.array([ 
            np.clip(self.drone_state[0] + action[0] * dt, -20, 20),
            np.clip(self.drone_state[1] + action[1] * dt, -20, 20)])
        


        self.state = np.concatenate((self.drone_state, self.target))



        self.time_left -= 0.1

        reward = self.alive_reward - self.action_penality * np.linalg.norm(action)**2 - self.pose_penality * np.linalg.norm(self.target - self.drone_state) ** 2
        # print(np.linalg.norm(self.target - self.drone_state))
        if self.time_left <= 0:
            done = True
        else:
            done = False

        info = {}
        return self.state, 10**(-5) * reward, done, info

    def render(self):
        pass
    
    def reset(self):
        self.drone_state = np.array([0, 0])
        self.target = np.array ([10, 10])
        self.state = np.concatenate((self.drone_state, self.target))
        self.time_left = self.episode_duration
        return self.state


if __name__ == "__main__":
        # log_dir = "tmp/"
        # os.makedirs(log_dir, exist_ok=True)
        env = DronSimEnv()
        check_env(env)
        policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch=[256, dict(pi=[128, 64, 16], vf=[128, 128])], optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5))

        model = A2C(MlpPolicy, env, learning_rate=schedule(0.01), use_sde=False, verbose=1, tensorboard_log="A2C_LOG", policy_kwargs=policy_kwargs, gamma=0.9, device="cpu")

        # model.set_parameters("A2C", exact_match=True)
        # # model = PPO.load("PPO", device='cpu')
        # callback = SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=log_dir)
        timesteps = 1000000
        model.learn(timesteps, progress_bar=True)
        # plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "A2C drone")
        # plt.show()
        model.save('A2C')


        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=5)
        print(mean_reward, std_reward)
        
        episodes = 3
        obs = env.reset()
        for episode in range(1, episodes+1):
            state = env.reset()
            done = False
            score = 0 
            while not done:
                action, _states = model.predict(state)
                n_state, reward, done, info = env.step(action)
                score+=reward
                print(n_state)
            print('Episode:{} Score:{}'.format(episode, score))


