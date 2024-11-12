"""
This file contains a set of CBF controller implementations.
These are generic implementations that may be called by other files.
"""
import sys
sys.path.append("../..")

#import folders
from depth_processing.pointcloud import *
from depth_processing.depth_proc import *

#import libraries
import numpy as np
from scipy import sparse
import osqp
import CalSim as cs

class CBFQPR1:
    """
    Implements a generic relative degree 1 CBF-QP controller.
    This does not interface directly with a dynamics instance.
    """
    def __init__(self, nominalCtrlEval, alpha, POSQP):
        """
        Init function for a CBF-QP Controller. This controller works over the 
        relative degree 2 dynamics of the system and directly outputs a force
        vector, rather than a velocity vector to track.

        Args:
            nominalCtrlEval (Function): Evaluation function from Nominal Contorller object
            alpha (float): constant on h in CBF constraint
            POSQP (Sparse matrix): weight matrix in cost function
        """
        #store CBF parameters
        self.alpha = alpha

        #store OSQP cost weight matrix -> convert to scipy sparse csc matrix
        self.POSQP = POSQP

        #create a nominal controller
        self.nominalCtrlEval = nominalCtrlEval

        #store a variable for the OSQP problem instance
        self.prob = None

    def solve_opti(self, h, Lfh, Lgh, t):
        """
        Set up the first CLF-CBF-QP optimization problem using OSQP.
        Note: This function generates a new OSQP instance each time it is called
        Strange issues with updating the parameters causing zeroing out happened otherwise.
        Inputs:
            V, h (1x1 NumPy Arrays): Lyapunov and barrier function values
            Lfh, Lgh: Lie derivatives of the system
            t (float): current time in simulation
        """
        #assemble a constraint matrix and vector of the correct shape
        Anp = -Lgh
        A = sparse.csc_matrix(Anp.tolist())
        b = self.alpha*h + Lfh
        q = -self.nominalCtrlEval(t) #q = -kX
    
        # Create an OSQP object and store in self.prob
        self.prob = osqp.OSQP()

        # Setup workspace and change alpha parameter
        self.prob.setup(P = self.POSQP, q = q, A = A, u = b,  verbose = False, alpha=1.0)

        # Solve problem
        res = self.prob.solve().x

        #return optimization output
        return res.reshape((res.size, 1))
    
    def eval_cbf_input(self, h, Lfh, Lgh, t):
        """
        Evaluate the input of the CBF-QP controller. This is the "z input" to the 2nd order system.
        Inputs:
            t (float): current time in simulation
        Returns:
            v (3x1 NumPy Array): safe velocity that allows for position tracking of the single integrator
        """
        #return the input
        return self.solve_opti(h, Lfh, Lgh, t)


class CBFQPR2:
    """
    Implements a generic relative degree 2 CBF-QP controller.
    This does not interface directly with a dynamics instance.
    """
    def __init__(self, nominalCtrlEval, alpha0, alpha1, POSQP):
        """
        Init function for a CBF-QP Controller. This controller works over the 
        relative degree 2 dynamics of the system and directly outputs a force
        vector, rather than a velocity vector to track.

        Args:
            nominalCtrlEval (Function): Evaluation function from Nominal Contorller object
            alpha0 (float): constant on h in CBF constraint
            alpha1 (float): constant on hDot
            POSQP (Sparse matrix): weight matrix in cost function
        """
        #store CBF parameters
        self.alpha0 = alpha0 #alpha for h
        self.alpha1 = alpha1 #alpha for hDot

        #store OSQP cost weight matrix -> convert to scipy sparse csc matrix
        self.POSQP = POSQP

        #create a nominal controller
        self.nominalCtrlEval = nominalCtrlEval

        #store a variable for the OSQP problem instance
        self.prob = None

    def solve_opti(self, h, Lfh, Lgh, Lf2h, LgLfh, t):
        """
        Set up the first CLF-CBF-QP optimization problem using OSQP.
        Note: This function generates a new OSQP instance each time it is called
        Strange issues with updating the parameters causing zeroing out happened otherwise.
        Inputs:
            V, h (1x1 NumPy Arrays): Lyapunov and barrier function values
            LfV, LgV, Lfh, Lgh: Lie derivatives of the system
            t (float): current time in simulation
        """
        #assemble a constraint matrix and vector of the correct shape
        Anp = -LgLfh
        A = sparse.csc_matrix(Anp.tolist())
        b = self.alpha0*h + self.alpha1*Lfh + Lf2h
        q = -self.nominalCtrlEval(t) #q = -kX
    
        # Create an OSQP object and store in self.prob
        self.prob = osqp.OSQP()

        # Setup workspace and change alpha parameter
        self.prob.setup(P = self.POSQP, q = q, A = A, u = b,  verbose = False, alpha=1.0)

        # Solve problem
        res = self.prob.solve().x

        #return optimization output
        return res.reshape((res.size, 1))
    
    def eval_cbf_input(self, h, Lfh, Lgh, Lf2h, LgLfh, t):
        """
        Evaluate the input of the CBF-QP controller. This is the "z input" to the 2nd order system.
        Inputs:
            t (float): current time in simulation
        Returns:
            v (3x1 NumPy Array): safe velocity that allows for position tracking of the single integrator
        """
        #return the input
        return self.solve_opti(h, Lfh, Lgh, Lf2h, LgLfh, t)
    

class CBF_QP_R2_3D_Ctrl(cs.Controller):
    """
    This is a skeleton class for a CBF QP for a double integrator 
    system in three dimensions. q = [x, y, z, xDot, yDot, zDot]
    """
    def __init__(self, observer, lyapunovBarrierList, trajectory, depthCam, DELTA, alpha0, alpha1, POSQP, nominalCtrlEval, ptcloud, depthProc):
        """
        Init function for a CBF-QP Controller. This controller works over the 
        relative degree 2 dynamics of the system and directly outputs a force
        vector, rather than a velocity vector to track.

        Args:
            observer (Observer): state observer object
            trajectory (Trajectory): trajectory for the controller to track (could just be a constant point!)
            trackingController (Controller): controller object that outputs a 3D force vector
        """
        #call the super init function
        super().__init__(observer, lyapunovBarrierList, trajectory, depthCam)

        #create pointcloud and DepthProc objects to interface with the depth camera
        self.pointcloud = ptcloud #Initialize pointcloud
        self.depthProc = depthProc
        
        #store CBF parameters
        self.DELTA = DELTA #buffer distance
        self.alpha0 = alpha0 #alpha for h
        self.alpha1 = alpha1 #alpha for hDot

        #store OSQP cost weight matrix -> convert to scipy sparse csc matrix
        self.POSQP = POSQP

        #Store a CBF-QP implementation using the nominal control function passed in
        self.CBFQP = CBFQPR2(nominalCtrlEval, self.alpha0, self.alpha1, self.POSQP)

        #store a variable for the OSQP problem instance
        self.prob = None

    def h(self, q, qC):
        """
        Computes the control barrier function
        Inputs:
            q (3x1 NumPy Array): current position vector of the system
            qC (3x1 NumPy Array): closest point in the pointcloud to the quadrotor
        Returns:
            h (1x1 NumPy Array): value of barrier function
        """
        #add on a buffer length of the arm length of the quadrotor
        return (q - qC).T @ (q - qC) - self.DELTA**2
    
    def LH(self, q, qDot, A, b, c):
        """
        Computes approximations of the Lie derivatives of the barrier function along
        the trajectories of the single integrator.
        Inputs:
            q (3x1 NumPy Array): current position vector of the system
            A (3x3 NumPy Array): quadratic weight
            b (3x1 NumPy Array): linear weight
            c (1x1 NumPy Array): affine weight
        Returns:
            Lfh, Lgh, Lf^2h, LgLfh: approximations of the lie derivatives
        """
        pass

    def eval_input(self, t):
        """
        Evaluate the input of the CBF-CLF-QP controller.
        Inputs:
            t (float): current time in simulation
        Returns:
            v (3x1 NumPy Array): safe velocity that allows for position tracking of the single integrator

        NOTE: The computation of the pointcloud takes significantly longer in the 3D case. This is what adds extra time.
        """
        #get the position of the quadrotor
        q = self.observer.get_pos()
        qDot = self.observer.get_vel()

        #first, update the pointcloud with the latest depth camera reading -> depthProc then gets this latest pointcloud
        self.pointcloud.update_pointcloud(self.depthCam.get_pointcloud())

        #next, compute the approximation of the CBF (use a 2D fit for now)
        A, b, c = self.depthProc.get_cbf_quad_fit_3D(q, self.h)

        #compute the Lie derivatives of the system
        Lfh, Lgh, Lf2h, LgLfh = self.LH(q, qDot, A, b, c)

        #compute closest point and barrier function
        qC, _ = self.depthProc.get_closest_point(q)
        h = self.h(q, qC)
        
        #set up optimization problem and solve.
        self._u = self.CBFQP.eval_cbf_input(h, Lfh, Lgh, Lf2h, LgLfh, t)

        #return the input
        return self._u