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

from vel_controller import MPC


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
        vel.linear.z = 0
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
        ros_node = RosConnector()
        # model = PPO.load("PPO", device='cpu')
        while not rospy.is_shutdown():
            rospy.wait_for_message("/drone_state", Odometry)
            mpc_controller = MPC()
            state = np.array([ros_node.state[0], ros_node.state[1], ros_node.state[2]])
            target = np.array([ros_node.target_pose[0], ros_node.target_pose[1], ros_node.target_pose[2]])
            vel = mpc_controller.get_control(state, target).reshape((3,1))

            # obs = target - state
            # action = model.predict(obs)
            ros_node.publish_velocity(vel)
        rospy.spin()
        
    except rospy.ROSInterruptException:
        pass