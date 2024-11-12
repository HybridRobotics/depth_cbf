import rospy
import tf2_ros
import tf
import numpy as np

class StateObserver:
    def __init__(self):
        """
        Init function for state observer
        """
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
  
    def get_state(self): #Maybe use the IMU and integrate to get velocity here?
        raise NotImplementedError

    def get_position(self):
        """
        Use the odometry data on the Turtlebot to return the current position
        """
        state_not_found = True
        while state_not_found and not rospy.is_shutdown():
            try:
                trans = self.tfBuffer.lookup_transform("odom", "base_footprint", rospy.Time())
                x, y = trans.transform.translation.x, trans.transform.translation.y
                state_not_found = False
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                # print(e)
                pass
        return x,y

class EgoTurtlebotObserver:
    def __init__(self):
        """
        Init function for a state observer for a single turtlebot within a system of N turtlebots
        Args:
            dynamics (Dynamics): Dynamics object for the entire turtlebot system
            mean (float): Mean for gaussian noise. Defaults to None.
            sd (float): standard deviation for gaussian noise. Defaults to None.
            index (Integer): index of the turtlebot in the system
        """
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.singleInputDimn = 2
        self.singleStateDimn = 3
    
    def get_state(self):
        """
        Returns a potentially noisy observation of the system state
        """
        state_not_found = True
        while state_not_found and not rospy.is_shutdown():
            try:
                trans = self.tfBuffer.lookup_transform("odom", "base_footprint", rospy.Time())
                x, y = trans.transform.translation.x, trans.transform.translation.y
                (roll, pitch, yaw) = tf.transformations.euler_from_quaternion([trans.transform.rotation.x, trans.transform.rotation.y,trans.transform.rotation.z, trans.transform.rotation.w])
                state_not_found = False
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                # print(e)
                pass
        return np.array([x,y,yaw]).reshape((3, 1))
    
    def get_orient(self):
        return self.get_state()[2, 0]
    
    def get_pos(self):
        return np.vstack((self.get_state()[0:2].reshape((2, 1)), 0))
    
    def get_vel(self, previous_u_input):
        """
        Returns a potentially noisy measurement of the derivative of the state vector of the ith turtlebot
        Inputs:
            previous_u_input: 2x1 numpy array, previous control input applied
        Returns:
            3x1 numpy array, observed derivative of the state vector of the ith turtlebot in the system (zero indexed)
        """
        PHI = self.get_state()[2, 0]
        xDot = np.array([[np.cos(PHI), 0], [np.sin(PHI), 0], [0, 1]])@previous_u_input
        return xDot.reshape((3, 1))