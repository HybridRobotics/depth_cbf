import numpy as np
from scipy.spatial import KDTree

class Pointcloud:
    """
    Master class for pointclouds. Interfaces with a depth camera.
    Includes utilities for computing pointcloud in the spatial frame,
    storing and updating pointclouds and the states at which they 
    were taken.
    """
    def __init__(self, ptcloudDict):
        """
        Init function for pointclouds.
        Inputs:
            ptcloudDict (Dictionary): {"ptcloud", "statevec"}
        """
        #store pointcloud dictionary
        self._ptcloudDict = None
        self.ptcloudQ = None
        self.q = None
        self.ptcloudS = None
        self.kdtree = None

        #update the class params with the pointcloud
        self.update_pointcloud(ptcloudDict)
    
    def get_ptcloud_q(self):
        """
        Returns the pointcloud in the qrotor frame
        """
        return self.ptcloudQ
    
    def get_state(self):
        """
        Returns the state of the qrotor when the pointcloud was taken
        """
        return self.q
    
    def get_ptcloud_s(self):
        """
        Retrieves the spatial pointcloud.
        """
        return self.ptcloudS
    
    def update_statevec(self, q):
        """
        Updates the state vector attribute
        """
        self.q = q
    
    def update_ptcloud_q(self, ptcloud_q):
        """
        Updates the pointcloud in qrotor frame
        """
        self.ptcloudQ = ptcloud_q
    
    def update_ptcloudDict(self, ptcloudDict):
        """
        Update the pointcloud dictionary with a new pointcloud dictionary
        Also updates ptcloud_q and statevec attributes
        """
        self._ptcloudDict = ptcloudDict
        self.update_ptcloud_q(ptcloudDict["ptcloud"])
        self.update_statevec(ptcloudDict["stateVec"])

    def compute_rotation(self, theta):
        """
        Compute the rotation matrix from the quadrotor frame to the world frame
        Inputs:
            theta: angle of rotation about the x-axis
        Returns:
            Rsq: rotation matrix between spatial and quadrotor frames
        """
        Rsq = np.array([[1, 0, 0], 
                        [0, np.cos(theta), -np.sin(theta)], 
                        [0, np.sin(theta), np.cos(theta)]])
        return Rsq
    
    def get_pos_orient_photo(self):
        """
        Get the 3D position vector and orientation of the system.
        """
        #extract XYZ position
        qPhoto = self.q[0:3].reshape((3, 1))

        #extract orientation angle of quadrotor when photo was taken
        thetaPhoto = self.q[4, 0]

        #compute and return the rotation
        return qPhoto, self.compute_rotation(thetaPhoto)
    
    def calc_ptcloud_s(self):
        """
        Calculates the pointcloud in the spatial frame. Updates the class attribute.
        """
        #calculate position and orientation at time of photo
        qPhoto, RPhoto = self.get_pos_orient_photo()

        #convert the ptcloud into the spatial frame and store in class attribute
        self.ptcloudS = RPhoto @ self.ptcloudQ + qPhoto
        return self.ptcloudS
    
    def update_pointcloud(self, ptcloudDict):
        """
        Master update function. Updates the dictionary and attributes
        and computes the pointcloud in the spatial frame.
        """
        #update the dictionary and qrotor frame attributes
        self.update_ptcloudDict(ptcloudDict)

        #update the world frame pointcloud
        self.calc_ptcloud_s()

        #update the KD tree
        self.kdtree = KDTree(self.get_ptcloud_s().T)

class PointcloudTurtlebot(Pointcloud):
    """
    Turtlebot pointcloud class. Interfaces with a depth camera.
    Includes utilities for computing pointcloud in the spatial frame,
    storing and updating pointclouds and the states at which they 
    were taken.
    """
    def __init__(self, ptcloudDict):
        """
        Init function for pointclouds.
        Inputs:
            ptcloudDict (Dictionary): {"ptcloud", "statevec"}
        """
        #call the super init function
        super().__init__(ptcloudDict)

    def compute_rotation(self, phi):
        """
        Compute the rotation matrix from the turtlebot frame to the world frame
        Inputs:
            phi: angle of rotation about the z-axis
        Returns:
            Rsq: rotation matrix between spatial and quadrotor frames
        """
        Rsq = np.array([[np.cos(phi), -np.sin(phi), 0], 
                        [np.sin(phi), np.cos(phi), 0], 
                        [0, 0, 1]])
        return Rsq
    
    def get_pos_orient_photo(self):
        """
        Get the 3D position vector and orientation of the system.
        """
        #extract XYZ position of quadrotor when photo was taken
        qPhoto = np.vstack((self.q[0:2].reshape((2, 1)), 0))

        #extract orientation angle of turtlebot when photo was taken
        thetaPhoto = self.q[2, 0]

        return qPhoto, self.compute_rotation(thetaPhoto)