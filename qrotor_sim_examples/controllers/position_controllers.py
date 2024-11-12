"""
This file contains all of the position controllers for the planar quadrotor system.
They take in a desired position and output a velocity Vd or a force vector f for
another controller to track.
"""

import sys
sys.path.append("..")

#import folders
from depth_processing.pointcloud import *
from depth_processing.depth_proc import *
from .cbf_controllers import *

#import libraries
import CalSim as cs
import numpy as np
from scipy import sparse
import osqp
import casadi as ca
import cvxpy as cp
from control import lqr

class QRotorCLF(cs.Controller):
    def __init__(self, observer, lyapunovBarrierList, trajectory, depthCam):
        """
        Init function for a CLF-QP tracking Controller. Note that this controller
        uses single integrator dynamics, and has velocity as its input.
        This velocity should then be passed to the velocity tracking controller,
        and in turn the Planar Quadrotor PD.

        Args:
            observer (Observer): state observer object
            trajectory (Trajectory): trajectory for the controller to track (could just be a constant point!)
        """
        #call the super init function
        super().__init__(observer, lyapunovBarrierList, trajectory, depthCam)

        #create pointcloud and DepthProc objects to interface with the depth camera
        self.pointcloud = PointcloudQrotor(self.depthCam.get_pointcloud()) #Initialize pointcloud
        self.depthProc = DepthProc(self.pointcloud)

        #store CBF/CLF parameters
        self.P = np.eye(3) #weight matrix in CLF
        self.H = np.eye(3) #weight matrix in cost function
        self.gamma = 3 #CLF gamma

        #store OSQP cost weight matrix -> convert to scipy sparse csc matrix
        self.POSQP = sparse.csc_matrix(self.H.tolist(), shape = (3, 3))

        #store a variable for the OSQP problem instance
        self.prob = None

        #timing arrays
        self.cvxTime = []
        self.casadiTime = []
        self.osqpTime = []

    def v(self, t):
        """
        Compute the control Lyapunov function, V.
        Inputs:
            t (float): current time in simulation
        Returns:
            V(q): current value of the CLF
        """
        #get the position of the quadrotor
        q = self.observer.get_pos()

        #get the desired position
        qD = self.trajectory.pos(t)[0:3].reshape((3, 1))

        #compute and return CLF value
        return ((qD - q).T @ (qD - q))
    
    def LV(self, t):
        """
        Computes the lie derivatives of the CLF along
        the trajectories of the single integrator.
        Inputs:
            t (float): current time in simulation
        Returns:
            LfV, LgV: Lie derivatives
        """
        #get the position of the quadrotor
        q = self.observer.get_pos()

        #get the desired position
        qD = self.trajectory.pos(t)[0:3].reshape((3, 1))

        return 0, (2*self.P@(q - qD)).T
    
    def solve_opti_osqp(self, V, LfV, LgV):
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
        A = sparse.csc_matrix(LgV.tolist(), shape = (1, 3))
        b = -self.gamma * V

        # Create an OSQP object and store in self.prob
        self.prob = osqp.OSQP()

        # Setup workspace and change alpha parameter
        self.prob.setup(P = self.POSQP, A = A, u = b,  verbose = False, alpha=1.0)

        # Solve problem
        res = self.prob.solve()
        return res.x.reshape((3, 1))

    def solve_opti_cvx(self, V, LfV, LgV):
        # Define and solve the CVXPY problem.
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, self.P)),
                        [LgV @ x <= -self.gamma * V])
        prob.solve()
        return (x.value).reshape((3, 1))

    def solve_opti_casadi(self, V, LfV, LgV):
        opti = ca.Opti()
        u = opti.variable(3, 1)
        cost = u.T @ u
        if np.linalg.norm(LgV) == 0:
            #add a buffer in case zero
            LgV = LgV + 0.0001
        opti.subject_to(LgV @ u <= -self.gamma * V[0, 0])
        opti.minimize(cost)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.solver('ipopt', option)
        sol = opti.solve()
        u = sol.value(u)
        return u.reshape((3, 1))
    
    def solve_opti(self, V, LfV, LgV, solver):
        """
        Solves the CLF-CBF-QP optimization problem using OSQP. Note: setup_opti()
        must be called before using this function!
        Inputs:
            V, h (1x1 NumPy Arrays): Lyapunov and barrier function values
            LfV, LgV, Lfh, Lgh: Lie derivatives of the system
            solver (Python function): reference to one of the optimization solvers above
        """
        #call the QP solver to solve the optimization
        return solver(V, LfV, LgV)

    def eval_input(self, t):
        """
        Evaluate the input of the CBF-CLF-QP controller.
        Inputs:
            t (float): current time in simulation
        Returns:
            v (3x1 NumPy Array): safe velocity that allows for position tracking of the single integrator
        """
        #get the position of the quadrotor
        q = self.observer.get_pos()

        #compute the Lie derivatives of the system
        LfV, LgV = self.LV(t)

        #compute Lyapunov functions
        V = self.v(t)

        #set up optimization problem and solve
        self._u = self.solve_opti(V, LfV, LgV, self.solve_opti_osqp)

        #return the input
        return self._u


class QRotorCBFCLF(cs.Controller):
    def __init__(self, observer, lyapunovBarrierList, trajectory, depthCam):
        """
        Init function for a CBF-CLF-QP Controller. Note that this controller
        uses single integrator dynamics, and has velocity as its input.
        This velocity should then be passed to the velocity tracking controller,
        and in turn the Planar Quadrotor PD.

        Args:
            observer (Observer): state observer object
            trajectory (Trajectory): trajectory for the controller to track (could just be a constant point!)
            trackingController (Controller): controller object that outputs a 3D force vector
        """
        #call the super init function
        super().__init__(observer, lyapunovBarrierList, trajectory, depthCam)

        #create pointcloud and DepthProc objects to interface with the depth camera
        self.pointcloud = PointcloudQrotor(self.depthCam.get_pointcloud()) #Initialize pointcloud
        self.depthProc = DepthProc(self.pointcloud)

        #store CBF/CLF parameters
        self.P = np.eye(3) #weight matrix in CLF
        self.H = np.eye(3) #weight matrix in cost function
        self.p = 0.1 #weight on relaxation term in cost function
        self.alpha = 1 #CBF alpha -> 5 works well for more aggressive, 1 is good for stability
        self.gamma = 3 #CLF gamma
        self.DELTA = 0.1 + self.observer.dynamics._l #buffer distance for CBF

        #store OSQP cost weight matrix -> convert to scipy sparse csc matrix
        POSQPnp = np.vstack((np.hstack((self.H, np.zeros((3, 1)))), np.array([[0, 0, 0, self.p]])))
        self.POSQP = sparse.csc_matrix(POSQPnp.tolist(), shape = (4, 4))

        #store a variable for the OSQP problem instance
        self.prob = None

    def v(self, t):
        """
        Compute the control Lyapunov function, V.
        Inputs:
            t (float): current time in simulation
        Returns:
            V(q): current value of the CLF
        """
        #get the position of the quadrotor
        q = self.observer.get_pos()

        #get the desired position
        qD = self.trajectory.pos(t)[0:3].reshape((3, 1))

        #compute and return CLF value
        return ((qD - q).T @ (qD - q))
    
    def LV(self, t):
        """
        Computes the lie derivatives of the CLF along
        the trajectories of the single integrator.
        Inputs:
            t (float): current time in simulation
        Returns:
            LfV, LgV: Lie derivatives
        """
        #get the position of the quadrotor
        q = self.observer.get_pos()

        #get the desired position
        qD = self.trajectory.pos(t)[0:3].reshape((3, 1))

        return 0, (2*self.P@(q - qD)).T

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
    
    def LH(self, q, A, b, c):
        """
        Computes approximations of the Lie derivatives of the barrier function along
        the trajectories of the single integrator.
        Inputs:
            q (3x1 NumPy Array): current position vector of the system
            A (3x3 NumPy Array): quadratic weight
            b (3x1 NumPy Array): linear weight
            c (1x1 NumPy Array): affine weight
        Returns:
            Lfh, Lgh: approximations of the lie derivatives
        """
        return 0, ((A + A.T) @ q + b).T
    
    def solve_opti_osqp(self, V, h, LfV, LgV, Lfh, Lgh):
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
        Anp = np.hstack((np.vstack((LgV, -Lgh)), np.array([[-1, 0]]).T))
        A = sparse.csc_matrix(Anp.tolist(), shape = (2, 4))
        b = np.array([[-self.gamma * V[0, 0], self.alpha * h[0, 0]]]).T

        # Create an OSQP object and store in self.prob
        self.prob = osqp.OSQP()

        # Setup workspace and change alpha parameter
        self.prob.setup(P = self.POSQP, A = A, u = b,  verbose = False, alpha=1.0)

        # Solve problem
        res = self.prob.solve()
        # print(res.x)
        return (res.x[0:3]).reshape((3, 1))
    
    def solve_opti(self, V, h, LfV, LgV, Lfh, Lgh, solver):
        """
        Solves the CLF-CBF-QP optimization problem using OSQP. Note: setup_opti()
        must be called before using this function!
        Inputs:
            V, h (1x1 NumPy Arrays): Lyapunov and barrier function values
            LfV, LgV, Lfh, Lgh: Lie derivatives of the system
            solver (Python function): reference to one of the optimization solvers above
        """
        #call the QP solver to solve the optimization
        return solver(V, h, LfV, LgV, Lfh, Lgh)

    def eval_input(self, t):
        """
        Evaluate the input of the CBF-CLF-QP controller.
        Inputs:
            t (float): current time in simulation
        Returns:
            v (3x1 NumPy Array): safe velocity that allows for position tracking of the single integrator
        """
        #get the position of the quadrotor
        q = self.observer.get_pos()

        #first, update the pointcloud with the latest depth camera reading -> depthProc then gets this latest pointcloud
        self.pointcloud.update_pointcloud(self.depthCam.get_pointcloud())

        #next, compute the approximation of the CBF (use a 2D fit for now)
        A, b, c = self.depthProc.get_cbf_quad_fit_3D(q, self.h)

        #compute the Lie derivatives of the system
        LfV, LgV = self.LV(t)
        Lfh, Lgh = self.LH(q, A, b, c)

        #compute Lyapunov functions
        V = self.v(t)

        #compute closest point and barrier function
        qC, _ = self.depthProc.get_closest_point(q)
        h = self.h(q, qC)
        
        #set up optimization problem and solve
        self._u = self.solve_opti(V, h, LfV, LgV, Lfh, Lgh, self.solve_opti_osqp)

        #return the input
        return self._u

class QRotorCBFCLFR2(cs.Controller):
    def __init__(self, observer, lyapunovBarrierList, trajectory, depthCam):
        """
        Init function for a CBF-CLF-QP Controller. This controller works over the 
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
        self.pointcloud = PointcloudQrotor(self.depthCam.get_pointcloud()) #Initialize pointcloud
        self.depthProc = DepthProc(self.pointcloud)

        #get system parameters
        self.m = self.observer.dynamics._m
        self.g = self.observer.dynamics._g
        self.e3 = np.array([[0, 0, 1]]).T

        #store optimization parameters
        self.H = np.eye(3) #weight matrix in cost function
        self.p = 0.1 #weight on relaxation term in cost function
        
        #store CBF parameters
        self.DELTA = 0.1 + self.observer.dynamics._l #buffer distance for CBF
        self.alpha0 = 24 #alpha for h
        self.alpha1 = 11 #alpha for hDot

        #Store CLF Parameters
        self.gamma0 = 24
        self.gamma1 = 11

        #store OSQP cost weight matrix -> convert to scipy sparse csc matrix
        POSQPnp = np.vstack((np.hstack((self.H, np.zeros((3, 1)))), np.array([[0, 0, 0, self.p]])))
        self.POSQP = sparse.csc_matrix(POSQPnp.tolist(), shape = (4, 4))

        #store a variable for the OSQP problem instance
        self.prob = None

    def v(self, t):
        """
        Compute the control Lyapunov function, V. This function is WRT the relative
        degree 2 dynamics.
        Inputs:
            t (float): current time in simulation
        Returns:
            V(q): current value of the CLF
        """
        #get the position and velocity from the observer
        q = self.observer.get_pos()

        #get desired position, velocity, and accel from the trajectory
        qD, vD, aD = self.trajectory.get_state(t)
        qD = qD[0:3].reshape((3, 1))

        #calculate the value of the Lyapunov function
        return (q - qD).T @ (q - qD)
    
    def vDot(self, t):
        """
        Computes the components of the CLF derivative. Note: since the CLF 
        is time varying, these are *not* Lie derivatives.
            -> vDDot = vDDotDrift + vDDotInput*u
        Inputs:
            t (float): current time in simulation
        Returns:
            vDot, vDDotDrift, vDDotInput: The two components of the CLF derivative.
        """
        #get the position and velocity from the observer
        q = self.observer.get_pos()
        v = self.observer.get_vel()

        #get desired position, velocity, and accel from the trajectory
        qD, vD, aD = self.trajectory.get_state(t)
        qD = qD[0:3].reshape((3, 1))
        vD = vD[0:3].reshape((3, 1))
        aD = aD[0:3].reshape((3, 1))

        #calculate derivative of Lyapunov function
        vDot = 2*(v - vD).T @ (q - qD)
        vDDotDrift = 2*(vD - vD).T @ (vD - vD) - 2*(q - qD).T @ aD
        vDDotInput = 2/self.m * (q - qD).T
        return vDot, vDDotDrift, vDDotInput

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
        Lfh = ((A + A.T) @ q + b).T @ qDot
        Lgh = np.zeros((1, 3))
        Lf2h = qDot.T @ (A + A.T) @ qDot - self.g *(q.T @ (A + A.T) + b.T) @ self.e3
        LgLfh = 1/self.m * (q.T @ (A + A.T) + b.T)
        return Lfh, Lgh, Lf2h, LgLfh
    
    def solve_opti(self, V, h, vDot, vDDotDrift, vDDotInput, Lfh, Lgh, Lf2h, LgLfh, useCBF = True):
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
        Anp = np.hstack((np.vstack((vDDotInput, -LgLfh)), np.array([[-1, 0]]).T))
        A = sparse.csc_matrix(Anp.tolist(), shape = (2, 4))
        b = np.vstack((-self.gamma0 * V - self.gamma1 * vDot - vDDotDrift, self.alpha0*h + self.alpha1*Lfh + Lf2h))
        
        # Create an OSQP object and store in self.prob
        self.prob = osqp.OSQP()

        # Setup workspace and change alpha parameter
        self.prob.setup(P = self.POSQP, A = A, u = b,  verbose = False, alpha=1.0)

        # Solve problem
        res = self.prob.solve()
        # print(res.x)
        return (res.x[0:3]).reshape((3, 1))

    def eval_input(self, t):
        """
        Evaluate the input of the CBF-CLF-QP controller.
        Inputs:
            t (float): current time in simulation
        Returns:
            v (3x1 NumPy Array): safe velocity that allows for position tracking of the single integrator
        """
        #get the position of the quadrotor
        q = self.observer.get_pos()
        qDot = self.observer.get_vel()

        #first, update the pointcloud with the latest depth camera reading -> depthProc then gets this latest pointcloud
        self.pointcloud.update_pointcloud(self.depthCam.get_pointcloud())

        #next, compute the approximation of the CBF (use a 2D fit for now)
        A, b, c = self.depthProc.get_cbf_quad_fit_3D(q, self.h)

        #compute the Lie derivatives of the system
        vDot, vDDotDrift, vDDotInput = self.vDot(t)
        Lfh, Lgh, Lf2h, LgLfh = self.LH(q, qDot, A, b, c)

        #compute Lyapunov function
        V = self.v(t)

        #compute closest point and barrier function
        qC, _ = self.depthProc.get_closest_point(q)
        h = self.h(q, qC)
        
        #set up optimization problem and solve. Add in gravity term.
        self._u = self.solve_opti(V, h, vDot, vDDotDrift, vDDotInput, Lfh, Lgh, Lf2h, LgLfh) + self.m*self.g*self.e3

        #return the input
        return self._u
    
class QRotorCBFQPR2(CBF_QP_R2_3D_Ctrl):
    """
    Qrotor R2 cbf. inherits from cbf qp.
    """
    def __init__(self, observer, lyapunovBarrierList, trajectory, depthCam, m = None, g = None):
        """
        Init function for a CBF-CLF-QP Controller. This controller works over the 
        relative degree 2 dynamics of the system and directly outputs a force
        vector, rather than a velocity vector to track.

        Args:
            observer (Observer): state observer object
            trajectory (Trajectory): trajectory for the controller to track (could just be a constant point!)
            trackingController (Controller): controller object that outputs a 3D force vector
        """ 
        
        #store CBF parameters
        DELTA = 0.1 + observer.dynamics._l #buffer distance for CBF
        alpha0 = 4 #alpha for h
        alpha1 = 4 #alpha for hDot

        #store OSQP cost weight matrix -> convert to scipy sparse csc matrix
        POSQP = sparse.csc_matrix(np.eye(3).tolist(), shape = (3, 3))

        #create pointcloud and DepthProc objects to interface with the depth camera
        if isinstance(observer.dynamics, cs.PlanarQrotor):
            #initialize planar pointcloud
            pointcloud = PointcloudQrotor(depthCam.get_pointcloud())
        elif isinstance(observer.dynamics, cs.Qrotor3D):
            #initialize 3D qrotor pointcloud
            pointcloud = PointcloudQrotor3D(depthCam.get_pointcloud())
        depthProc = DepthProc(pointcloud)

        #call the super init function
        super().__init__(observer, lyapunovBarrierList, trajectory, depthCam, DELTA, alpha0, alpha1, POSQP, self.eval_lqr_input, pointcloud, depthProc)

        #get system parameters
        if m is None and g is None:
            self.m = self.observer.dynamics._m
            self.g = self.observer.dynamics._g
        else:
            self.m = m
            self.g = g
        
        #store e3 vector
        self.e3 = np.array([[0, 0, 1]]).T
        
        #store LQR matrices
        A = np.vstack((np.hstack((np.zeros((3, 3)), np.eye(3))), np.zeros((3, 6))))
        B = 1/self.m * np.vstack((np.zeros((3, 3)), np.eye(3)))
        Q = np.eye(6)
        R = np.eye(3)
        Klqr, S, E = lqr(A, B, Q, R)
        self.Klqr = Klqr
        
    def eval_lqr_input(self, t):
        """
        Evaluates a nominal LQR input to the system
        """
        xD, vD, aD = self.trajectory.get_state(t)
        xD, vD, aD = xD[0:3].reshape((3, 1)), vD[0:3].reshape((3, 1)), aD[0:3].reshape((3, 1))

        #get position and velocity
        q = np.vstack((self.observer.get_pos(), self.observer.get_vel()))
        qD = np.vstack((xD, vD))

        #Compute LQR input with feedforward gravity and acceleration terms
        return self.Klqr @ (qD - q) + self.m * aD + self.m*self.g*self.e3

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
        Lfh = ((A + A.T) @ q + b).T @ qDot
        Lgh = np.zeros((1, 3))
        Lf2h = qDot.T @ (A + A.T) @ qDot - self.g *(q.T @ (A + A.T) + b.T) @ self.e3
        LgLfh = 1/self.m * (q.T @ (A + A.T) + b.T)
        return Lfh, Lgh, Lf2h, LgLfh
    

class QRotor3DTracking(cs.Controller):
    """
    Class for a state feedback position tracking controller for the full 3D quadrotor.
    """
    def __init__(self, observer, lyapunovBarrierList, trajectory, depthCam):
        """
        Init function for a 3D quadrotor controller. This controller
        finds a 3D force vector that may be tracked by an orientation controller.

        Args:
            observer (Observer): state observer object
            trajectory (Trajectory): trajectory for the controller to track (could just be a constant point!)
        """
        #call the super init function
        super().__init__(observer, lyapunovBarrierList, trajectory, depthCam)

        #store system parameters
        self.m = self.observer.dynamics._m
        self.g = self.observer.dynamics._g
        self.e3 = np.array([[0, 0, 1]]).T

        #compute system matrices and LQR gains
        A = np.vstack((np.hstack((np.zeros((3, 3)), np.eye(3))), np.zeros((3, 6))))
        B = 1/self.m * np.vstack((np.zeros((3, 3)), np.eye(3)))
        Q = np.eye(6)
        R = np.eye(3)
        Klqr, S, E = lqr(A, B, Q, R)
        self.Klqr = Klqr

    def eval_input(self, t):
        """
        Evaluates a tracking LQR input to the system
        """
        xD, vD, aD = self.trajectory.get_state(t)
        xD, vD, aD = xD[0:3].reshape((3, 1)), vD[0:3].reshape((3, 1)), aD[0:3].reshape((3, 1))

        #get position and velocity
        q = np.vstack((self.observer.get_pos(), self.observer.get_vel()))
        qD = np.vstack((xD, vD))

        #Compute LQR input with feedforward gravity and acceleration terms
        return self.Klqr @ (qD - q) + self.m * aD + self.m*self.g*self.e3