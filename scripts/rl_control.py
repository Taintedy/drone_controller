#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import gym 
from gym import Env
from gym.spaces import Box 
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy


class DronSimEnv(Env):
    def __init__(self):
        # Actions we can take, down, stay, up
        self.ros_node = RosConnector()
        self.action_space = Box(low=-1, high=1, shape=(3,))
        # Temperature array
        self.observation_space = Box(low=float('-inf'), high=float('inf'), shape=(6,))
        # Set start temp
        self.state = self.ros_node.state
        # Set shower length
        self.start_time = rospy.get_time()
        self.episod_duration = 5
        
    def step(self, action):
        # Apply action
        # print(f"episode len left {self.episode_len}")
        self.ros_node.publish_velocity(action) 
        self.state = self.ros_node.state
        # Reduce shower length by 1 second
        duration = rospy.get_time() - self.start_time
        # print(duration)
        # Calculate reward
        reward = -(np.linalg.norm(self.state[:3] - self.ros_node.target_pose))**2
        
        # Check if shower is done
        if self.episod_duration <= duration or self.ros_node.is_collision: 
            done = True
        else:
            done = False
        
        # Apply temperature noise
        #self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        # Reset shower temperature
        self.ros_node.publish_restart(True)
        self.state = self.ros_node.state
        self.ros_node.is_collision = False
        # Reset shower time
        self.start_time = rospy.get_time()
        return self.state

class RosConnector():

    def __init__(self) -> None:
        self.state = np.zeros(6)
        self.target_pose = np.zeros(3)
        self.is_collision = False
        #subscribers
        rospy.Subscriber("/drone_state", Odometry, self.callback_state, queue_size=10)
        rospy.Subscriber("/target_pose", Pose, self.callback_target_pose, queue_size=10)
        rospy.Subscriber("/collision_detection", Bool, self.callback_collision, queue_size=10)
        #publishers
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.cmd_restart = rospy.Publisher("/restart_request", Bool, queue_size=10)
        #services

    def callback_state(self, msg):
        self.state = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])

    def callback_collision(self, msg):
        self.is_collision = True

    def callback_target_pose(self, msg):
        self.target_pose = np.array([msg.position.x, msg.position.y, msg.position.z])

    def publish_velocity(self, cmd):
        vel = Twist()
        vel.linear.x = cmd[0]
        vel.linear.y = cmd[1]
        vel.linear.z = cmd[2]
        self.cmd_vel_pub.publish(vel)
    
    def publish_restart(self, cmd):
        self.cmd_restart.publish(cmd)
        


def shutdown():
    rospy.logwarn("shutting down")

if __name__ == "__main__":
    try:
        rospy.on_shutdown(shutdown)
        rospy.init_node('mission_execution_node', anonymous=True)
        rospy.logwarn("working")
        rate = rospy.Rate(10)
        env = DronSimEnv()
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=5000000)
        model.save('PPO')
        # episodes = 5
        # for episode in range(1, episodes+1):
        #     state = env.reset()
        #     done = False
        #     score = 0 
        #     while not done:
        #         # print("in while loop")
        #         action = env.action_space.sample()
        #         n_state, reward, done, info = env.step(action)
        #         score+=reward
        #     print('Episode:{} Score:{}'.format(episode, score))
        rospy.spin()
        
    except rospy.ROSInterruptException:
        pass


