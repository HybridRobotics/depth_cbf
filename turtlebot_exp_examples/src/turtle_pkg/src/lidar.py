import rospy
import tf2_ros
import tf

from sensor_msgs.msg import LaserScan

import numpy as np

class Lidar:
	def __init__(self):
		self._sensor_topic = "/scan"
		self._sensor_sub = rospy.Subscriber(self._sensor_topic, LaserScan, self.SensorCallback, queue_size=1)
		self._range_max = 5
		self._range_min = 0.15
		self._ptcloudSize = 375
		self._pointcloud = np.ones((3,self._ptcloudSize))*5
		self._pointcloud[2] = np.zeros((1,self._ptcloudSize))

    # Callback to process sensor measurements.
	def SensorCallback(self, msg):
		# Loop over all ranges in the LaserScan.
		r_min = 20			# Initialize at random impossible values
		angle_min = 1000
		self._pointcloud = np.ones((3,self._ptcloudSize))*5
		for idx, r in enumerate(msg.ranges):
			# Throw out this point if it is too close or too far away.
			if r > self._range_max:
				continue
			if r < self._range_min:
				continue
			# Get angle of this ray
			angle = msg.angle_min + idx * msg.angle_increment
			self._pointcloud[0][idx] = r*np.cos(angle)
			self._pointcloud[1][idx] = r*np.sin(angle)
			self._pointcloud[2][idx] = 0

	def get_pointcloud(self):
		return self._pointcloud
