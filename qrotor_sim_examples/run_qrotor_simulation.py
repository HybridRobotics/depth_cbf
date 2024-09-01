import CalSim as cs
import numpy as np
from controllers.force_controllers import *
from trajectories import *

#system initial condition
x0 = np.array([[0, 0, 1, 0, 0, 0, 0, 0]]).T #start the quadrotor at 1 M in the air

#create a dynamics object for the double integrator
dynamics = cs.PlanarQrotor(x0)

#create an obstacle
qObs = np.array([[0, 1, 1], [0, 0.5, 2]]).T
rObs = [0.25, 0.25]
obstacleManager = cs.ObstacleManager(qObs, rObs, NumObs = 2)

#create an observer
observerManager = cs.ObserverManager(dynamics)

#create a depth camera
depthManager = cs.DepthCamManager(observerManager, obstacleManager, mean = None, sd = None)

#create a trajectory
xD = np.array([[0, 2, 1.5, 0, 0, 0, 0, 0]]).T
trajManager = TrajectoryManager(x0, xD, Ts = 10, N = 1)

#create a controller manager 
controllerManager = cs.ControllerManager(observerManager, PlanarQrotorPD, None, trajManager, depthManager)

env = cs.Environment(dynamics, controllerManager, observerManager, obstacleManager, T = 15)
env.reset()

#run the simulation
env.run()