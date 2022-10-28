# DroneRLControl

A ros package for controling a drone using Stable Baseline 3.

## Requirements 
Stable Baseline and Gym is required this packahe to work:

```
pip install stable-baselines3[extra]
sudo -H pip install gym
```
This package was developed using the ROS noetic framework, so it is recomended to use this version of ROS. To install it follow the tutorial [here](http://wiki.ros.org/noetic/Installation/Ubuntu)

## Usage 
To run the ros node go to your workspace and clone the repository. After that source your workspace and run the command below:

```
rosrun DroneRLControl ros_node.py
```

> Final Project in "Reinforcement Learning" course, 2022.  
> Authors: [Sausar Karaf](https://github.com/Taintedy), [Aleksey Fedoseev](https://github.com/ASFedoseev)
