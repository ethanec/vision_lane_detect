#!/usr/bin/env python

import rospy
import message_filters
from sensor_msgs.msg import Image
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

#import tensorflow as tf

import tf


class CosineSimilarity:

    def __init__(self):
        self.cameraMat = np.reshape([9.8253595454109904e+02, 0., 4.8832572821891949e+02, 0., 9.7925117255173427e+02, 3.0329591096350060e+02, 0., 0., 1. ], (3, 3))
        self.imageSize = [960, 604]

        self.first_callback = 1
        self.last_car_position = [0, 0, 0]
        self.present_car_position = [0, 0, 0]

        #self.car_position_Time = []
        #self.car_position_NED = []
        self.similarity_array = []
        self.iter = 0

        self.T_now = 0
        self.T_last = 0

        car_position_topic_name = '/ndt_pose'
        projection_points_topic_name = 'lane_points_topic'
        detection_left_points_topic_name = 'laneNet_left_lane'

        rospy.loginfo('Setting ndt topic to %s', car_position_topic_name)
        rospy.loginfo('Setting ndt topic to %s', projection_points_topic_name)
        rospy.loginfo('Setting ndt topic to %s', detection_left_points_topic_name)

        #---------------------------------- Time Synchronizer
        
        self.car_position_sub = message_filters.Subscriber(car_position_topic_name, PoseStamped)
        self.projection_sub = message_filters.Subscriber(projection_points_topic_name, Float32MultiArray)
        self.detection_sub = message_filters.Subscriber(detection_left_points_topic_name, Int32MultiArray)
        ts = message_filters.ApproximateTimeSynchronizer([self.car_position_sub, self.projection_sub, self.detection_sub], 1, 1, allow_headerless=True) ### Here first 1 is the size of queue and next 1 is the time in sec to consider for aprox
        ts.registerCallback(self.CosineSimilarity_callback)


        self.listener = tf.TransformListener()

    def CosineSimilarity_callback(self, data, pro, det):
        self.T_now = time.time()
        t_cost = self.T_now - self.T_last 
        rospy.loginfo('{} second'.format(t_cost))
        self.T_last = self.T_now
        # /ndt_pose's frequency is 10Hz
        try:
            if self.first_callback == 1:
                self.first_callback = 0
                self.last_car_position = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z]) # Concern NED
            elif (self.first_callback == 0) and (self.iter < 300):
                self.present_car_position = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z]) # Concern NED
                
                vec = self.present_car_position[0:2] - self.last_car_position[0:2] # Vector of the car's heading
                #ratio_to_one_meter = 1 / (((vec[0]) ** 2 + (vec[1]) ** 2) ** 0.5)
                #unit_vec = vec * ratio_to_one_meter
                perpendicular_vec = [vec[1], -vec[0]]

                F = np.polyfit([self.present_car_position[0], self.last_car_position[0]], [self.present_car_position[1], self.last_car_position[1]], 1)
                

                self.listener.waitForTransform('map', 'camera', data.header.stamp, rospy.Duration(4.0))
                
                pro_left_lane = []
                det_left_lane = []
                #print(pro.layout.dim[0].size)

                # Find the left lane line of projection results
                for i in range(0, pro.layout.dim[0].size):
                    if len(pro.data) >= 6:
                        x1 = pro.data[i * 6 + 0]
                        y1 = pro.data[i * 6 + 1]
                        z1 = pro.data[i * 6 + 2]
                        x2 = pro.data[i * 6 + 3]
                        y2 = pro.data[i * 6 + 4]
                        z2 = pro.data[i * 6 + 5]

                        vec1 = [x1 - self.present_car_position[0], y1 - self.present_car_position[1]]
                        vec2 = [x2 - self.present_car_position[0], y2 - self.present_car_position[1]]

                        d1 = abs(F[0] * x1 - 1 * y1 + F[1])/np.sqrt(pow(F[0], 2) + 1)
                        d2 = abs(F[0] * x2 - 1 * y2 + F[1])/np.sqrt(pow(F[0], 2) + 1)


                        if (np.dot(vec1, perpendicular_vec) < 0 and d1 < 2):
                            point_in_map = PoseStamped()
                            point_in_map.header = data.header
                            point_in_map.pose.position.x = x1 # Concern NED
                            point_in_map.pose.position.y = y1 # Concern NED
                            point_in_map.pose.position.z = z1 # Concern NED
                            
                            # Frame Transform
                            point_in_camera = self.listener.transformPose('camera', point_in_map)
                            #rospy.loginfo(center_lane_pose_in_camera)
                            imageCoord = np.dot(self.cameraMat, [point_in_camera.pose.position.x, point_in_camera.pose.position.y, point_in_camera.pose.position.z])
                            imageCoord = ((imageCoord / imageCoord[-1])[0:2])
                            #print(imageCoord)
                            if (imageCoord[0] < self.imageSize[0] and imageCoord[0] > 0) and (imageCoord[1] < self.imageSize[1] and imageCoord[1] > 0):
                                imageCoord[0] = round(imageCoord[0])
                                imageCoord[1] = round(imageCoord[1])
                                pro_left_lane.append(imageCoord)
                            imageCoord = []

                        if (np.dot(vec2, perpendicular_vec) < 0 and d2 < 2):
                            point_in_map = PoseStamped()
                            point_in_map.header = data.header
                            point_in_map.pose.position.x = x2 # Concern NED
                            point_in_map.pose.position.y = y2 # Concern NED
                            point_in_map.pose.position.z = z2 # Concern NED
                            
                            # Frame Transform
                            point_in_camera = self.listener.transformPose('camera', point_in_map)
                            #rospy.loginfo(center_lane_pose_in_camera)
                            imageCoord = np.dot(self.cameraMat, [point_in_camera.pose.position.x, point_in_camera.pose.position.y, point_in_camera.pose.position.z])
                            imageCoord = ((imageCoord / imageCoord[-1])[0:2])
                            #print(imageCoord)
                            if (imageCoord[0] < self.imageSize[0] and imageCoord[0] > 0) and (imageCoord[1] < self.imageSize[1] and imageCoord[1] > 0):
                                imageCoord[0] = round(imageCoord[0])
                                imageCoord[1] = round(imageCoord[1])
                                pro_left_lane.append(imageCoord)
                            imageCoord = []
                pro_left_lane = np.array(pro_left_lane)

                for j in range(0, det.layout.dim[0].size):
                    if len(det.data) >= 2:
                        det_left_lane.append([det.data[j * 2 + 0], det.data[j * 2 + 1]])
                det_left_lane = np.array(det_left_lane)

                #print(np.array(pro_left_lane))
                
                if (len(pro_left_lane) > 0 and len(det_left_lane) > 0):
                    pro_poly = np.polyfit(pro_left_lane[:, 0], pro_left_lane[:, 1], 1)
                    det_poly = np.polyfit(det_left_lane[:, 0], det_left_lane[:, 1], 1)

                    A = np.array([-1, -pro_poly[0]])
                    B = np.array([-1, -det_poly[0]])

                    similarity = abs(np.dot(A, B) / (np.sqrt(pow(A[0], 2) + pow(A[1], 2)) * np.sqrt(pow(B[0], 2) + pow(B[1], 2))))

                    self.similarity_array.append(similarity)
                else:
                    self.similarity_array.append(0)
                    #py = pro_poly[0] * pro_left_lane[:, 0] + pro_poly[1]
                    #dy = det_poly[0] * det_left_lane[:, 0] + det_poly[1]


                    #print('----------')
                    #print(pro_left_lane[:, 0])
                    #print(py)
                    #print('----------')
                    #plt.plot(pro_left_lane[:, 0], py, color='r')
                    #plt.plot(det_left_lane[:, 0], py, color='b')
                    #plt.show(block=False)
                    #plt.close()
                
                #print(pro_left_lane)
                #print('----------')
                #print(det_left_lane)


                #print(vec)
                #print(pro.layout.dim[0].size)

                self.last_car_position = self.present_car_position
                self.iter = self.iter + 1
            else:
                print(self.similarity_array)


        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) or CvBridgeError as e:
            print(e)

    def main(self):
        
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('cosine_similarity_node', anonymous=True)

    CS = CosineSimilarity()
    try:
        CS.main()
    except rospy.ROSInterruptException:
        pass