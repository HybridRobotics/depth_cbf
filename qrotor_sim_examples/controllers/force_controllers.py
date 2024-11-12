"""
This file contains all of the force-tracking controllers.
They take in a desired force vector and output a (f, M) pair where f is in R and M in R3.
These controllers all interact directly with the quadrotor, as opposed to the other
classes, which do not.
"""
import sys
sys.path.append("..")

#import folders
from .velocity_controllers import *
from .position_controllers import *

#import libraries
import CalSim as cs
import numpy as np

import time
from statistics import mean


class PlanarQrotorPD(cs.Controller):
    def __init__(self, observer, lyapunovBarrierList, trajectory, depthCam):
        """
        Init function for a planar quadrotor controller. This controller
        takes in a position tracking controller object that operates on the point mass
        and converts the force vector input into an (f, M) pair that can be sent to 
        the quadrotor.

        Args:
            observer (Observer): state observer object
            trajectory (Trajectory): trajectory for the controller to track (could just be a constant point!)
            trackingController (Controller): controller object that outputs a 3D force vector
        """
        super().__init__(observer, lyapunovBarrierList, trajectory, depthCam)
        
        #Initialize variables for the gain parameters
        self.Ktheta = 0.08 #proportional orientation gain
        self.Komega = 0.02 #derivative orientation gain
        
        #Store quadrotor parameters from the observer
        self.m = self.observer.dynamics._m
        self.Ixx = self.observer.dynamics._Ixx
        self.g = 9.81 #store gravitational constant
        
        #store Euclidean basis vectors
        self.e1 = np.array([[1, 0, 0]]).T
        self.e2 = np.array([[0, 1, 0]]).T
        self.e3 = np.array([[0, 0, 1]]).T

        #store the tracking controller -> this controller outputs a force vector to be tracked by this controller
        # self.trackingController = VelocityTrackingPD(self.observer, lyapunovBarrierList, self.trajectory, depthCam) #velocity tracking controller
        # self.trackingController = VelocityTrackingCLF(self.observer, lyapunovBarrierList, self.trajectory, depthCam)
        # self.trackingController = QRotorCBFCLFR2(self.observer, lyapunovBarrierList, self.trajectory, depthCam)
        self.trackingController = QRotorCBFQPR2(self.observer, lyapunovBarrierList, self.trajectory, depthCam)
        
        #timing option - time the average frequency of the controller
        self.RUN_TIMING = False #NOTE: These timing functions are inaccurate due to simulation computation
        self.freq_arr = []
    
    def eval_force_vec(self, t):
        """
        Function to evaluate the force vector input to the system using point mass dynamics.
        Args:
            t (float): current time in simulation
        Returns:
            f ((3 x 1) NumPy Array): virtual force vector to be tracked by the orientation controller
        """
        return self.trackingController.eval_input(t)
    
    def eval_desired_orient(self, f):
        """
        Function to evaluate the desired orientation of the system.
        Args:
            f ((3 x 1) NumPy array): force vector to track from point mass dynamics
        Returns:
            thetaD (float): desired angle of quadrotor WRT world frame
        """
        return np.arctan2(-f[1, 0], f[2, 0]) #remember to flip the sign!
    
    def eval_orient_error(self, t):
        """
        Evalute the orientation error of the system thetaD - thetaQ
        Args:
            t (float): current time in simulation
        Returns:
            eOmega (float): error in orientation angle
        """
        f = self.eval_force_vec(t) #force we'd like to track
        thetaD = self.eval_desired_orient(f) #desired angle of quadrotor
        thetaQ = self.observer.get_orient() #current angle of quadrotor
        
        #return the difference
        return thetaD - thetaQ
    
    def eval_moment(self, t):
        """
        Function to evaluate the moment input to the system
        Args:
            t (float): current time in simulation
        Returns:
            M (float): moment input to quadrotor
        """
        eTheta = self.eval_orient_error(t)
        eOmega = 0 - self.observer.get_omega() #assume zero angular velocity desired
        thetaDDotD = 0 #Assume a desired theta dddot of 0
        
        #return the PD controller output - assume zero desired angular acceleration
        return self.Ktheta*eTheta + self.Komega*eOmega + self.Ixx*thetaDDotD
    
    def eval_force_scalar(self, t):
        """
        Evaluates the scalar force input to the system.
        Args:
            t (float): current time in simulation
        Returns:
            F (float): scalar force input from PD control
        """
        #first, construct R, a rotation matrix about the x axis
        thetaQ = self.observer.get_orient()
        R = np.array([[1, 0, 0], 
                      [0, np.cos(thetaQ), -np.sin(thetaQ)], 
                      [0, np.sin(thetaQ), np.cos(thetaQ)]])
        
        #find and return the scalar force
        return (self.eval_force_vec(t).T@R@self.e3)[0, 0]
        
    def eval_input(self, t):
        """
        Get the control input F, M to the planar quadrotor system
        Args:
            t (float): current time in simulation
        Returns:
            self._u = [F, M] ((2x1) numpy array): force, moment input to system
        """
        if self.RUN_TIMING:
            #compute the input with timing
            t0 = time.time()
            #store input in the class parameter
            self._u = np.array([[self.eval_force_scalar(t), self.eval_moment(t)]]).T
            t1 = time.time()
            if t1 != t0:
                self.freq_arr.append(1/(t1 - t0))
            if len(self.freq_arr) != 0:
                print("Average frequency: ", mean(self.freq_arr))
        else:
            self._u = np.array([[self.eval_force_scalar(t), self.eval_moment(t)]]).T
        return self._u
    
class QRotorGeometricPD(cs.Controller):
    def __init__(self, observer, lyapunovBarrierList, trajectory, depthCam):
        """
        Init function for a geometric 3D quadrotor controller. This controller
        takes in a position tracking controller object that operates on the point mass
        and converts the force vector input into an (f, M) pair that can be sent to 
        the quadrotor.

        Args:
            observer (Observer): state observer object
            trajectory (Trajectory): trajectory for the controller to track (could just be a constant point!)
            trackingController (Controller): controller object that outputs a 3D force vector
        """
        super().__init__(observer, lyapunovBarrierList, trajectory, depthCam)
        
        #Initialize variables for the gain parameters
        self.KR = 0.08 #proportional orientation gain
        self.Komega = 0.02 #derivative orientation gain
        
        #Store quadrotor parameters from the observer
        self.m = self.observer.dynamics._m
        self.I = self.observer.dynamics._I
        self.g = 9.81 #store gravitational constant
        
        #store Euclidean basis vectors
        self.e1 = np.array([[1, 0, 0]]).T
        self.e2 = np.array([[0, 1, 0]]).T
        self.e3 = np.array([[0, 0, 1]]).T

        #store the tracking controller -> this controller outputs a force vector to be tracked by this controller
        # self.trackingController = QRotor3DTracking(self.observer, lyapunovBarrierList, self.trajectory, depthCam)
        self.trackingController = QRotorCBFQPR2(self.observer, lyapunovBarrierList, self.trajectory, depthCam)
        
        #timing option - time the average frequency of the controller
        self.RUN_TIMING = False #NOTE: These timing functions are inaccurate due to simulation computation
        self.freq_arr = []
    
    def eval_force_vec(self, t):
        """
        Function to evaluate the force vector input to the system using point mass dynamics.
        Args:
            t (float): current time in simulation
        Returns:
            f ((3 x 1) NumPy Array): virtual force vector to be tracked by the orientation controller
        """
        return self.trackingController.eval_input(t)
    
    def eval_desired_orient(self, f):
        """
        Function to evaluate the desired orientation of the system.
        Args:
            f ((3 x 1) NumPy array): force vector to track from point mass dynamics
        Returns:
            Rd ((3x3 NumPy Array)): desired rotation matrix of quadrotor WRT world frame
        """
        #get the current rotation matrix of the quadrotor
        R = self.observer.get_orient()

        #set the b1d vector using R
        b1d = R @ self.e1

        #compute zd by normalizing f
        zd = f/np.linalg.norm(f)

        #compute yd and xd using b1d
        yd = cs.hat(zd) @ b1d
        xd = cs.hat(yd) @ zd

        #assemble and return the desired rotation matrix
        return np.hstack((xd, yd, zd))
    
    def vee_3d(self, wHat):
        """
        Function to compute the vee map of a 3x3 matrix in so(3).
        Inputs:
            wHat (3x3 NumPy Array): matrix in so(3) to compute vee map of
        Returns:
            w (3x1 NumPy Array): 3x1 vector corresponding to wHat
        """
        return np.array([[wHat[2, 1], wHat[0, 2], wHat[1, 0]]]).T
    
    def eval_orient_error(self, t):
        """
        Evalute the orientation error of the system thetaD - thetaQ
        Args:
            t (float): current time in simulation
        Returns:
            eR, eOmega ((3x1) NumPy Arrays): error in orientation angle and angular velocity
        """
        #get current orientation and angular vel
        R = self.observer.get_orient()
        omega = self.observer.get_omega()

        #get force vector
        f = self.eval_force_vec(t)
        
        #get desired rotation matrix, set desired omega to 0
        Rd = self.eval_desired_orient(f)
        omegad = np.zeros((3, 1))

        #compute errors
        eR = 0.5 * self.vee_3d(Rd.T @ R - R.T @ Rd)
        eOmega = omega - R.T @ Rd @ omegad
        
        #return the difference
        return eR, eOmega
    
    def eval_moment(self, t):
        """
        Function to evaluate the moment input to the system
        Args:
            t (float): current time in simulation
        Returns:
            M (float): moment input to quadrotor
        """
        #evaluate orientation error
        eR, eOmega = self.eval_orient_error(t)

        #get current omega
        omega = self.observer.get_omega()
        
        #return the PD controller output - assume zero desired angular acceleration
        return -self.KR * eR - self.Komega * eOmega + cs.hat(omega) @ self.I @ omega
    
    def eval_force_scalar(self, t):
        """
        Evaluates the scalar force input to the system.
        Args:
            t (float): current time in simulation
        Returns:
            F (float): scalar force input from PD control
        """
        #first get rottaion matrix
        R = self.observer.get_orient()
        
        #find and return the scalar force
        return (self.eval_force_vec(t).T @ R @ self.e3)[0, 0]
        
    def eval_input(self, t):
        """
        Get the control input F, M to the planar quadrotor system
        Args:
            t (float): current time in simulation
        Returns:
            self._u = [F, M] ((4x1) numpy array): force, moment input to system
        """
        if self.RUN_TIMING:
            #compute the input with timing
            t0 = time.time()
            #store input in the class parameter
            self._u = np.vstack(([self.eval_force_scalar(t), self.eval_moment(t)]))
            t1 = time.time()
            if t1 != t0:
                self.freq_arr.append(1/(t1 - t0))
            if len(self.freq_arr) != 0:
                print("Average frequency: ", mean(self.freq_arr))
        else:
            self._u = np.vstack(([self.eval_force_scalar(t), self.eval_moment(t)]))
        return self._u