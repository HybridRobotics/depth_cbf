import rospy
import numpy as np
from geometry_msgs.msg import Twist, Point
import sys
import time

#import files
sys.path.append('../../../..')
from sim.controllers.cbf_controllers import *
from depth_processing.depth_proc import *

class Controller:
    def __init__(self, observer, trajectory = None):
        """
        Skeleton class for feedback controllers
        Args:
            observer (Observer): state observer object
            trajectory (Trajectory): trajectory for the controller to track (could just be a constant point!)
        """
        #store input parameters
        self.observer = observer
        self.trajectory = trajectory
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.previous_u_input = np.zeros((2, 1))
    
    def eval_input(self, t):
        """
        Solve for and return control input
        Inputs:
            t (float): time in simulation
        Returns:
            u ((Dynamics.inputDimn x 1)): input vector, as determined by controller
        """
        self._u = self.trajectory.vel(t)
        return self._u
    
    def get_input(self):
        """
        Retrieves input stored in class parameter
        Returns:
            self._u: most recent input stored in class paramter
        """
        return self._u
    
    def apply_input(self):
        msg = Twist()

        #OVERRIDE TEMP
        # self._u = np.zeros((2, 1))
        msg.linear.x = self._u[0, 0]
        msg.angular.z = self._u[1, 0]
        self.previous_u_input = self._u
        self.pub.publish(msg)
        return

class TurtlebotFBLin(Controller):
    def __init__(self, observer, trajectory, frequency):
        """
        Class for a feedback linearizing controller for a single turtlebot within a larger system
        Args:
            observer (Observer): state observer object
            traj (Trajectory): trajectory object
        """
        #store input parameters
        self.observer = observer
        self.trajectory = trajectory

        #store the time step dt for integration
        self.dt = 1/frequency #from the control frequency in environment.py

        #store feedback gains
        self.k1 = 4
        self.k2 = 4

        #store value for integral
        self.vDotInt = 0

        #previous control input
        self.previous_u_input = np.array([0., 0.]).reshape((2,1))
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    
    def eval_z_input(self, t):
        """
        Solve for the input z to the feedback linearized system.
        Use linear tracking control techniques to accomplish this.
        """
        #get the state of turtlebot i
        q = self.observer.get_state()

        #get the derivative of q of turtlebot i
        qDot = self.observer.get_vel(self.previous_u_input)

        #get the trajectory states
        xD, vD, aD = self.trajectory.get_state(t)

        #find a control input z to the augmented system
        e = np.array([[xD[0, 0], xD[1, 0]]]).T - np.array([[q[0, 0], q[1, 0]]]).T
        eDot = np.array([[vD[0, 0], vD[1, 0]]]).T - np.array([[qDot[0, 0], qDot[1, 0]]]).T
        z = aD[0:2].reshape((2, 1)) + self.k2 * eDot + self.k1 * e

        #return the z input
        return z
    
    def eval_w_input(self, t, z):
        """
        Solve for the w input to the system
        Inputs:
            t (float): current time in the system
            z ((2x1) NumPy Array): z input to the system
        """
        #get the current phi
        phi = self.observer.get_state()[2, 0]

        #get the (xdot, ydot) velocity
        qDot = self.observer.get_vel(self.previous_u_input)[0:2]
        v = np.linalg.norm(qDot)

        #first, eval A(q)
        Aq = np.array([[np.cos(phi), -v*np.sin(phi)], 
                       [np.sin(phi), v*np.cos(phi)]])

        #invert to get the w input - use pseudoinverse to avoid problems
        w = np.linalg.pinv(Aq)@z

        #return w input
        return w

    def eval_input(self, t):
        """
        Solves for the control input to turtlebot i using a CBF-QP controller.
        Inputs:
            t (float): current time in simulation
            i (int): index of turtlebot in the system we wish to control (zero indexed)
        """
        #get the z input to the system
        z = self.eval_z_input(t)

        #get the w input to the system
        w = self.eval_w_input(t, z)

        #integrate the w1 term to get v
        self.vDotInt += w[0, 0]*self.dt

        #return the [v, omega] input
        self._u = np.array([[self.vDotInt, w[1, 0]]]).T
        return self._u
    

class TurtlebotCBFQP(TurtlebotFBLin):
    def __init__(self, observer, pointcloud, trajectory, depthCam, frequency):
        """
        QRotorCBFQPGazebo(self.observer, self.pointcloud, self.trajectory, self.depthCam, m = self._m, g = self._g)
        Class for an R = 2 CBF-QP controller for a single turtlebot.
        Args:
            observer (EgoTurtlebotObserver): state observer object for a single turtlebot within the system
            barriers (List of TurtlebotBarrier): List of TurtlebotBarrier objects corresponding to that turtlebot
            traj (Trajectory): trajectory object
        """
        #call the super init function
        super().__init__(observer, trajectory, frequency)

        #create pointcloud and DepthProc objects to interface with the depth camera
        self.depthCam = depthCam
        self.pointcloud = pointcloud #Initialize pointcloud
        self.depthProc = DepthProc(self.pointcloud)

        #create a nominal controller
        self.nominalController = TurtlebotFBLin(observer, trajectory, frequency)

        #store CBF parameters
        self.DELTA = 0 #0.5 + 0.15 #buffer distance for CBF, rTurtlebot = 0.15
        self.alpha0 = 0.01 #alpha for h
        self.alpha1 = 0.2 #alpha for hDot

        #store OSQP cost weight matrix -> convert to scipy sparse csc matrix
        self.POSQP = sparse.csc_matrix(np.diag([1, 1]).tolist(), shape = (2, 2))

        #create a CBF QP object
        nominalEvalFunc = self.nominalController.eval_z_input
        self.CBFQP = CBFQPR2(nominalEvalFunc, self.alpha0, self.alpha1, self.POSQP)

        #timing option - time the average frequency of the controller
        self.RUN_TIMING = False
        self.freq_arr = []

    def h(self, q, qC):
        """
        Computes the control barrier function
        Inputs:
            q (3x1 NumPy Array): current position vector of the system
            qC (3x1 NumPy Array): closest point in the pointcloud to the turtlebot
        Returns:
            h (1x1 NumPy Array): value of barrier function
        """
        #add on a buffer length of the turtlebot radius
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
        Lgh = np.zeros((1, 2))
        Lf2h = qDot.T @ (A + A.T) @ qDot
        LgLfh = q.T @ (A + A.T) + b.T
        return Lfh, Lgh, Lf2h, LgLfh
    
    def eval_z_input(self, t):
        """
        Evaluate the input of the CBF-QP controller. This is the "z input" to the 2nd order system.
        Inputs:
            t (float): current time in simulation
        Returns:
            v (3x1 NumPy Array): safe velocity that allows for position tracking of the single integrator
        """
        #get the position of the turtlebot
        q = self.observer.get_pos()[0:2].reshape((2, 1))
        qDot = self.observer.get_vel(self.previous_u_input)[0:2].reshape((2, 1))

        #next, compute the approximation of the CBF
        A, b, c = self.depthProc.get_cbf_quad_fit_3D(self.observer.get_pos(), self.h)

        #extract the relevant components
        A = A[0:2, 0:2]
        b = b[0:2].reshape((2, 1))

        #compute the Lie derivatives of the system
        Lfh, Lgh, Lf2h, LgLfh = self.LH(q, qDot, A, b, c)

        #compute closest point and barrier function
        qC, _ = self.depthProc.get_closest_point(self.observer.get_pos())
        h = self.h(q, qC[0:2].reshape((2, 1)))

        # print(qC.T)
        print(h)

        #return the input
        try:
            return self.CBFQP.eval_cbf_input(h, Lfh, Lgh, Lf2h, LgLfh, t)
        except:
            print("Infeasible optimization")
            return np.zeros((2, 1))
        

class TurtlebotVelTrack(cs.Controller):
    def __init__(self, observer, lyapunovBarrierList, trajectory, depthCam):
        """
        Init function for velocity tracking controller for turtlebot
        """
        super().__init__(observer, lyapunovBarrierList, trajectory, depthCam)

        #store controller gain
        self.kTheta = 1
        self.e3 = np.array([0, 0, 1])

    def eval_phi_error(self, vD):
        """
        Function to evaluate angular error.
        Inputs:
            vD ((3 x 1) NumPy Array): desired velocity of system
        """
        #get current angle from observer
        phi = self.observer.get_orient()

        #form phi vector, vector of current vel heading
        vHat = np.array([np.cos(phi), np.sin(phi), 0])

        #get velocity vector
        vDHat = vD.reshape((3, ))

        #compute desired angle
        return np.arctan2(np.dot(np.cross(vHat, vDHat), self.e3), np.dot(vHat, vDHat))

    def eval_input_pos(self, t, vD = None):
        """
        Compute the input to the system. Note that this function only allows
        for positive velocity inputs to the system. This function can
        encourage more turning for better deadlock resolution but has worse tracking.
        """
        #get the desired velocity
        if vD is None:
            #use the trajectory desired velocity
            vD = self.trajectory.vel(t)

        #assemble inputs
        if np.linalg.norm(vD) != 0:
            v = np.dot(vD.reshape((3, )), vD.reshape(3, )/np.linalg.norm(vD)) #get signed norm of desired velocity
        else:
            #if zero divide, then ||vD|| = 0
            v = 0
        omega = self.kTheta * self.eval_phi_error(vD)

        #return full input vector
        self._u = np.array([[v, omega]]).T
        return self._u
    
    def eval_input(self, t, vD = None):
        """
        Compute the input to the system
        """
        #get the desired velocity
        if vD is None:
            #use the trajectory desired velocity
            vD = self.trajectory.vel(t)


        #get phi error
        ePhi = self.eval_phi_error(vD)

        #assemble inputs
        if np.linalg.norm(vD) != 0:
            # v = np.dot(vD.reshape((3, )), vD.reshape(3, )/np.linalg.norm(vD)) #get signed norm of desired velocity

            #New: get signed norm of v based on ePhi
            if abs(ePhi) > np.pi/2:
                #negate sign of velocity
                v = -np.linalg.norm(vD)

                #update ePhi usingg the negative velocity vector
                ePhi = self.eval_phi_error(-vD)
            else:
                v = np.linalg.norm(vD)
        else:
            #if zero divide, then ||vD|| = 0
            v = 0

        #compute omega based on the update ePhi
        omega = self.kTheta * ePhi

        #return full input vector
        self._u = np.array([[v, omega]]).T
        return self._u
    

class TurtlebotCBFR1(cs.Controller):
    def __init__(self, observer, pointcloud, trajectory, depthCam):
        """
        Init function for velocity tracking controller for turtlebot
        """
        super().__init__(observer, None, trajectory, depthCam)

        #initialize planar pointcloud and depthProc
        self.pointcloud = pointcloud
        self.depthProc = DepthProc(self.pointcloud)

        #store state feedback gain
        self.kX = 1

        #set OSQP weight matrix
        POSQP = sparse.csc_matrix(np.diag([1, 1, 1]).tolist(), shape = (3, 3))

        #set CBF alpha
        alpha = 6

        #set delta for CBF (was 0.2 + 0.15 for turning case)
        self.DELTA = 0.125 + 0.15

        #store CBF QP instance
        self.CBFQP = CBFQPR1(self.nominal_eval, alpha = alpha, POSQP = POSQP)

        #store velocity tracking controller
        self.velControl = TurtlebotVelTrack(observer, None, trajectory, depthCam)

        #store publisher
        self.previous_u_input = np.array([0., 0.]).reshape((2,1))
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        #store a publisher for bagging system data
        self.pub_data = rospy.Publisher('/sys_data', Twist, queue_size=10, tcp_nodelay=True)
        # self.pub_pos = rospy.Publisher('/pos_data', Point, queue_size=10, tcp_nodelay=True)

        #store a list for the data values over time
        self.hList = [] #barrier function values
        self.tList = [] #time values
        self.vList = [] #velocity input to turtlebot
        self.omegaList = [] #omega input to turtlebot
        self.vXDList = [] #safe x velocity inputs from CBF
        self.vYDList = [] #safe y velocity inputs from CBF

        #store tPrev
        self.tPrev = time.time()

    def nominal_eval(self, t):
        """
        Nominal control eval
        """
        x = self.observer.get_pos()
        xD = self.trajectory.pos(t)
        return self.kX * (xD - x)

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
        return np.zeros((1, 1)), ((A + A.T) @ q + b).T

    def eval_input(self, t, save = False):
        """
        Evaluate the input of the CBF-CLF-QP controller.
        Inputs:
            t (float): current time in simulation
            save (bool): save the CBF value
        Returns:
            v (3x1 NumPy Array): safe velocity that allows for position tracking of the single integrator
        """
        #get the position of the turtlebot
        q = self.observer.get_pos()

        #next, compute the approximation of the CBF (use a 2D fit for now)
        A, b, c = self.depthProc.get_cbf_quad_fit_3D(q, self.h)

        #compute the Lie derivatives of the system
        Lfh, Lgh = self.LH(q, A, b, c)

        #compute closest point and barrier function
        qC, _ = self.depthProc.get_closest_point(q)
        h = self.h(q, qC)

        # print(qC.T)
        # print("CBF VALUE: ", h)
        
        #set up optimization problem and solve to get desired velocity
        vD = self.CBFQP.eval_cbf_input(h, Lfh, Lgh, t)

        #convert to turtlebot input
        self._u = self.velControl.eval_input(t, vD)

        if save:
            #publish system data in the form of a twist
            msg = Twist()

            #store CBF value
            msg.linear.x = h[0, 0]

            #store v and omega inputs
            msg.linear.y = self._u[0, 0]
            msg.linear.z = self._u[1, 0]

            #store positions
            msg.angular.x = q[0, 0]#vD[0, 0]
            msg.angular.y = q[1, 0]#vD[1, 0]

            #store current time
            msg.angular.z = t

            #publish the message to the sys_data topic
            self.pub_data.publish(msg)

        #return the input
        return self._u
    
    def apply_input(self):
        msg = Twist()

        #OVERRIDE TEMP
        # self._u = np.zeros((2, 1))
        msg.linear.x = self._u[0, 0]
        msg.angular.z = self._u[1, 0]
        self.previous_u_input = self._u
        self.pub.publish(msg)
        return