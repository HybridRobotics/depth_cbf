"""
This file contains all of the velocity tracking controllers.
These controllers take in a desired velocity and output a force vector
for the moment controller to track.
"""

import sys
sys.path.append("..")

#import folders
from depth_processing.pointcloud import *
from depth_processing.depth_proc import *
from .position_controllers import *

#import libraries
import CalSim as cs
import numpy as np
from scipy import sparse
import osqp

class VelocityTrackingPD(cs.Controller):
    def __init__(self, observer, lyapunovBarrierList, trajectory, depthCam):
        """
        Init function for a velocity tracking controller for a planar quadrotor.
        Inputs:
            observer (QuadObserver): quadrotor observer object
            trajectory (Trajectory): trajectory object
        """
        #call the super init function
        super().__init__(observer, lyapunovBarrierList=lyapunovBarrierList, trajectory=trajectory, depthCam=depthCam)
        self.depthCam = depthCam

        #store the gain
        self.Kp = 2*np.eye(3)

        #store quadrotor parameters
        self.m = self.observer.dynamics._m
        self.g = self.observer.dynamics._g
        self.l = self.observer.dynamics._l
        self.Ixx = self.observer.dynamics._Ixx

        #store useful vectors
        self.e3 = np.array([[0, 0, 1]]).T

        #store the CBF controller, which gives the desired velocity
        self.qRotorCBF = QRotorCBFCLF(self.observer, None, self.trajectory, self.depthCam)
        # self.qRotorCBF = QRotorCLF(self.observer, None, self.trajectory, self.depthCam)


    def eval_input(self, t, useCBF = True):
        """
        Evaluate the input to the system at time t for proportional velocity tracking.
        Inputs:
            t (float): current time in simulation
            useCBF (Boolean): True if the desired velocity should come from the CBF controller, false if directly from traj
        Outputs:
            u (3x1 NumPy Array): force vector f for a point mass to track a desired velocity
        """
        if useCBF:
            #get the desired velocity from the CBF controller
            vD = self.qRotorCBF.eval_input(t)
        else:
            #get the desired velocity directly from the trajectory
            xD, vD, aD = self.trajectory.get_state(t)
            vD = vD[0:3].reshape((3, 1)) #extract XYZ velocities

        #get the current velocity of the quadrotor from the observer
        v = self.observer.get_vel()

        #compute and store the input
        self._u = self.m*(self.g*self.e3 + self.Kp @ (vD - v))
        return self._u
    
class VelocityTrackingCLF(cs.Controller):
    def __init__(self, observer, lyapunovBarrierList, trajectory, depthCam):
        """
        Init function for a velocity tracking CLF-QP. This CLF tracks 
        a desired velocity from a trajectory using the quadrotor dynamics
        vDot = f/m - ge3. Provides exponential tracking.
        """
         #call the super init function
        super().__init__(observer, lyapunovBarrierList=lyapunovBarrierList, trajectory=trajectory, depthCam=depthCam)
        self.depthCam = depthCam

        #store the CLF properties
        self.POSQP = sparse.csc_matrix(np.eye(3).tolist(), shape = (3, 3)) #weight matrix
        self.gamma = 3 #constraint gamma

        #store quadrotor parameters
        self.m = self.observer.dynamics._m
        self.g = self.observer.dynamics._g
        self.l = self.observer.dynamics._l
        self.Ixx = self.observer.dynamics._Ixx

        #store useful vectors
        self.e3 = np.array([[0, 0, 1]]).T

        #store the CBF controller, which gives the desired velocity
        self.qRotorCBF = QRotorCBFCLF(self.observer, None, self.trajectory, self.depthCam)
        # self.qRotorCBF = QRotorCLF(self.observer, None, self.trajectory, self.depthCam)
    
    def v(self, t, vD):
        """
        Evaluates the velocity-tracking Lyapunov function at time t.
        """
        #get the quadrotor velocity
        v = self.observer.get_vel()

        #return Lyapunov function
        return (v - vD).T @ (v - vD)

    def vDot(self, t, vD, aD):
        """
        Function to calculate the components of the Lyapunov function's derivative.
        Note that since the Lyapunov function is time varying due to the non-constant
        trajectory, the components returned are *not* Lie derivatives.
        Rather, vDot = vDotDrift + vDotInput * u -> these two components are returned.
        Returns:
            vDotDrift, vDotInput (NumPy Arrays): The two components of the Lyapunov function's derivative (drift & input term)
        """
        #get quadrotor velocity
        v = self.observer.get_vel()

        #compute derivative terms
        vDotDrift = -2*(v - vD).T @ aD #- 2*self.g*(v - vD).T @ self.e3 # Ignore gravity term and linearize later
        vDotInput = 2/self.m * (v - vD).T

        #return the two CLF terms
        return vDotDrift, vDotInput
    
    def solve_opti(self, V, vDotDrift, vDotInput):
        """
        Set up the first CLF-CBF-QP optimization problem using OSQP.
        Note: This function generates a new OSQP instance each time it is called
        Strange issues with updating the parameters causing zeroing out happened otherwise.
        Inputs:
            V, h (1x1 NumPy Arrays): Lyapunov and barrier function values
            LfV, LgV, Lfh, Lgh: Lie derivatives of the system
            t (float): current time in simulation
        """
        #assemble a dummy setup constraint matrix and vector of the correct shape
        A = sparse.csc_matrix(vDotInput.tolist(), shape = (1, 3))
        b = -self.gamma * V - vDotDrift

        # Create an OSQP object and store in self.prob
        self.prob = osqp.OSQP()

        # Setup workspace and change alpha parameter
        self.prob.setup(P = self.POSQP, A = A, u = b,  verbose = False, alpha=1.0)

        # Solve problem
        res = self.prob.solve()
        return res.x.reshape((3, 1)) + self.g * self.m * self.e3
    
    def eval_input(self, t, useCBF = True):
        """
        Evaluate the input of the CBF-CLF-QP controller.
        Inputs:
            t (float): current time in simulation
        Returns:
            v (3x1 NumPy Array): safe velocity that allows for position tracking of the single integrator
        """
        if useCBF:
            #get the desired velocity from the CBF controller
            vD = self.qRotorCBF.eval_input(t)
            aD = np.zeros((3, 1)) #set desired accel to 0
        else:
            #get the desired velocity directly from the trajectory
            xD, vD, aD = self.trajectory.get_state(t)
            vD = vD[0:3].reshape((3, 1)) #extract XYZ velocities
            aD = aD[0:3].reshape((3, 1))

        #compute the derivatives of V
        vDotDrift, vDotInput = self.vDot(t, vD, aD)

        #compute Lyapunov functions
        V = self.v(t, vD)

        #set up optimization problem and solve
        self._u = self.solve_opti(V, vDotDrift, vDotInput)

        #return the input
        return self._u