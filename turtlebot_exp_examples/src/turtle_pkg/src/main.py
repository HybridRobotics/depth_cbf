#!/usr/bin/env python
#The line above tells Linux that this file is a Python script,
#and that the OS should use the Python interpreter in /usr/bin/env
#to run it. Don't forget to use "chmod +x [filename]" to make
#this script executable.

#Import the rospy package. For an import to work, it must be specified
#in both the package manifest AND the Python file in which it is used.
import rospy
import numpy as np

from state_estimation import EgoTurtlebotObserver
from trajectory import Trajectory
from controller import *
from lidar import Lidar
from pointcloud import *

"""
TASKS:
WallTest: Move forward 1m from (0, 0, 0) with a flat wall in front
MovingFrontTest: Move a flat wall towards the system
ClutterTest: Move through the cluttered environment from (0, 0, 0) to (3, 0, 0)
""" 

#define a list containing the control frequencies
freqList =  []

#defome a list containing the h values
hList = []

#Define the method which contains the main functionality of the node.
def task_controller():
  """
  Controls a turtlebot whose position is denoted by turtlebot_frame,
  to go to a position denoted by target_frame
  """

  # Initialization Time
  start_time = rospy.get_time()
  frequency = 300
  r = rospy.Rate(frequency) # 50hz
  
  # Observer
  observer = EgoTurtlebotObserver()

  # Trajectory
  start_position = np.array([[0, 0, 0]]).T
  end_position = np.array([[3, 0, 0]]).T
  time_duration = 1
  trajectory = Trajectory(start_position, end_position, time_duration)

  # Lidar
  lidar = Lidar()

  # Pointcloud -> initialize with an empty dictionary
  ptcloudDict = {}
  ptcloudDict["ptcloud"] = lidar.get_pointcloud()
  ptcloudDict["stateVec"] = np.zeros((3, 1))
  pointcloud = PointcloudTurtlebot(ptcloudDict)

  #set to true to apply CBF-QP control
  useCBFQP = True 
  if not useCBFQP:
    controller = TurtlebotFBLin(observer, trajectory, frequency)
  else:
    # controller = TurtlebotCBFQP(observer, pointcloud, trajectory, lidar, frequency)
    controller = TurtlebotCBFR1(observer, pointcloud, trajectory, lidar)

  # Loop until the node is killed with Ctrl-C
  while not rospy.is_shutdown():
    t = rospy.get_time() - start_time
    t1 = rospy.get_time()
    #update the pointcloud dictionary and pass it into the pointcloud object
    ptcloudDict["stateVec"] = observer.get_state()
    ptcloudDict["ptcloud"] = lidar.get_pointcloud()
    pointcloud.update_pointcloud(ptcloudDict)

    #evaluate and apply the control input
    controller.eval_input(t, save = True)
    # controller.eval_input(t)
    controller.apply_input()
    t2 = rospy.get_time()

    # print("FREQ: ", 1/(t2 - t1))

    #update the frequency and h lists
    freqList.append(1/(t2 - t1))
    r.sleep()

  #After rospy shutdown, save the timing test
  save = True
  if save:
    #save the two lists
    np.save("npy_data/timingtest.npy", np.array(freqList))

# This is Python's sytax for a main() method, which is run by default
# when exectued in the shell
if __name__ == '__main__':
  # Check if the node has received a signal to shut down
  # If not, run the talker method

  #Run this program as a new node in the ROS computation graph 
  #called /turtlebot_controller.
  rospy.init_node('turtlebot_controller', anonymous=True)

  try:
    task_controller()
  except rospy.ROSInterruptException:
    pass