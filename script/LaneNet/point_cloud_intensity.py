#!/usr/bin/env python

import rospy
import message_filters
#import ros_numpy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped, Vector3
from visualization_msgs.msg import MarkerArray #****************************************
from tf.transformations import quaternion_matrix
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
#from autoware_msgs.msg import _ImageLaneObjects

import copy
import numpy as np
import yaml
import pandas as pd
import cv2
import time
from cv_bridge import CvBridge, CvBridgeError
import rospkg
#import tf
import math
import matplotlib
import matplotlib.pyplot as plt


def points_callback(data):
    print('-----------------')
    I = pc2.read_points(data, skip_nans=True, field_names=("intensity"))
    #pc = ros_numpy.numpify(data)
    #I = pc['intensity']
    I_list = []
    for i in I:
        I_list.append(i)
    print(I_list)
    
    #print(I)
    
    print(data.fields)
    
    #print(data.point_step)
    #print(data.row_step)
    print('-----------------')


if __name__ == '__main__':
    rospy.init_node('point_cloud_intensity_node', anonymous=True)

    try:
        points_topic_name = '/points_raw'
        rospy.loginfo('Setting points topic to %s', points_topic_name)
        #self.points_sub = rospy.Subscriber(points_topic_name, PointCloud2, self.points_callback)
        rospy.Subscriber(points_topic_name, PointCloud2, points_callback)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass