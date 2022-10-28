#! /usr/bin/env python3
import rospy
import numpy as np
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist

poses = []

def shutdown():
    np.save("pose_data_exp_forward_10.npy", np.array(poses))

def pose_callback(msg):
    q = [msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z,
             msg.transform.rotation.w]
    angle = euler_from_quaternion(q)[2]

    pose = np.array([
        rospy.get_time(),
        msg.transform.translation.x,
        msg.transform.translation.y,
        msg.transform.translation.z,
        angle])
    
    poses.append(pose)
    print(pose)

if __name__ == "__main__":
    try:
        rospy.on_shutdown(shutdown)
        rospy.init_node('logger', anonymous=True)
        rate = rospy.Rate(100)
        rospy.Subscriber('/vicon/walking_drone/walking_drone', TransformStamped, pose_callback)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass