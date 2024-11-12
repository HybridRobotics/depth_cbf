import CalSim as cs
import numpy as np
from controllers.force_controllers import *
from trajectories import *

#system initial condition
pos0 = np.array([[0, 0, 1]]).T
vel0 = np.zeros((3, 1))
omega0 = np.zeros((3, 1))
R0 = np.eye(3).reshape((9, 1))
x0 = np.vstack((pos0, R0, omega0, vel0))

#create a dynamics object for the double integrator
dynamics = cs.Qrotor3D(x0)

#create an observer
observerManager = cs.ObserverManager(dynamics)

#create an obstacle
qObs = np.array([[0, 1, 1], [0, 0.5, 2]]).T
rObs = [0.25, 0.25]
obstacleManager = cs.ObstacleManager(qObs, rObs, NumObs = 2)

#create a depth camera
depthManager = cs.DepthCamManager(observerManager, obstacleManager, mean = None, sd = None)

#create a trajectory
xD = np.vstack((np.array([[0, 2, 1.5]]).T, R0, omega0, vel0))
trajManager = TrajectoryManager(x0, xD, Ts = 5, N = 1)

#create a controller manager with a basic FF controller
controllerManager = cs.ControllerManager(observerManager, QRotorGeometricPD, None, trajManager, depthManager)

#create an environment
env = cs.Environment(dynamics, controllerManager, observerManager, obstacleManager, T = 10)
env.reset()

#run the simulation
env.run()