#!/usr/bin/env python

import rospy
import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped, Vector3
from visualization_msgs.msg import MarkerArray #****************************************
from tf.transformations import quaternion_matrix
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

#import tensorflow as tf

import tf

from lanenet_model import lanenet_merge_model
from lanenet_model import lanenet_cluster
from lanenet_model import lanenet_postprocess
from config import global_config

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]



# TODO(lucasw) these no longer exist, demo.launch supersedes this test
# is there any reason to keep this around?
# from rviz_textured_quads.msg import TexturedQuad, TexturedQuadArray
'''
def pub_image():

    rospy.init_node('rviz_display_image_test', anonymous=True)
    rospack = rospkg.RosPack()

    image_pub = rospy.Publisher("/targets", Image, queue_size=10)

    texture_path = rospack.get_path('rviz_textured_quads') + '/tests/textures/'
    img1 = cv2.imread(texture_path + 'shalun.png', cv2.IMREAD_COLOR)
    img_msg1 = CvBridge().cv2_to_imgmsg(img1, "bgr8")

    #img2 = cv2.imread(texture_path + 'Decal.png', cv2.IMREAD_COLOR)
    #img_msg2 = CvBridge().cv2_to_imgmsg(img2, "bgr8")

    #rate = rospy.Rate(30)
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        image_pub.publish(img_msg1)
        rate.sleep()
'''

class HDPipeline:

    def __init__(self):
        '''
        with open('/home/mec-lab/horiz_revise.yml', 'r') as stream:
            cam_data = yaml.load(stream)
        self.cameraExtrinsicMat = np.reshape(cam_data['CameraExtrinsicMat']['data'], (4, 4))
        self.cameraMat = np.reshape(cam_data['CameraMat']['data'], (3, 3))
        self.imageSize = cam_data['ImageSize']
        self.distCoeff = cam_data['DistCoeff']
        self.reProjectionError = cam_data['ReprojectionError']
        '''
        self.cameraMat = np.reshape([9.8253595454109904e+02, 0., 4.8832572821891949e+02, 0., 9.7925117255173427e+02, 3.0329591096350060e+02, 0., 0., 1. ], (3, 3))
        self.imageSize = [960, 604]

        self.first_callback = 1
        self.last_car_position = [0, 0, 0]
        self.present_car_position = [0, 0, 0]

        self.car_position_Time = []
        self.car_position_NED = []
        self.edge_NE = []
        self.central_NE = []

        edge_file = '/home/mec-lab/Autoware/ros/src/computing/perception/detection/vision_detector/packages/vision_lane_detect/script/position_data/sl_edge.xls'
        central_file = '/home/mec-lab/Autoware/ros/src/computing/perception/detection/vision_detector/packages/vision_lane_detect/script/position_data/sl_central.xls'

        edge_df = pd.read_excel(edge_file)
        central_df = pd.read_excel(central_file)

        self.edge_NE = np.array(edge_df.iloc[:, 0:2])
        self.central_NE = np.array(central_df.iloc[:, 0:2])

        self.edge_NE[:, [0, 1]] = self.edge_NE[:, [1, 0]]
        self.central_NE[:, [0, 1]] = self.central_NE[:, [1, 0]]
        
        #---------------------------------- Set topic
        self.image_pub = rospy.Publisher("laneNet_discrepancy", Image)
        self.bridge = CvBridge()
        
        image_topic_name = '/image_raw'
        car_position_topic_name = '/ndt_pose'
        #vector_map_topic_name = '/vector_map'

        rospy.loginfo('Setting image topic to %s', image_topic_name)
        rospy.loginfo('Setting ndt topic to %s', car_position_topic_name)
        #rospy.loginfo('Setting ndt topic to %s', vector_map_topic_name)
        #self.image_sub = rospy.Subscriber(image_topic_name, Image, self.image_callback)
        #self.car_position_sub = rospy.Subscriber(car_position_topic_name, PoseStamped, self.ndt_callback)

        #---------------------------------- Time Synchronizer
        self.image_sub = message_filters.Subscriber(image_topic_name, Image)
        self.car_position_sub = message_filters.Subscriber(car_position_topic_name, PoseStamped)
        ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.car_position_sub], 1, 1) ### Here first 1 is the size of queue and next 1 is the time in sec to consider for aprox
        ts.registerCallback(self.HDPipeline_callback)

        #---------------------------------- 
        #self.vector_map = message_filters.Subscriber(vector_map_topic_name, MarkerArray)
        #print(self.vector_map)

        self.listener = tf.TransformListener()

    def front_direction_estimation(self, now_position, last_position):
        estimate_range = 15 #meter
        #to_lidar_range = 1.72 #meter
        ne_vector = now_position - last_position # Vector of the car's heading
        #distance = (((ne_vector[0]) ** 2 + (ne_vector[1]) ** 2) ** 0.5)
        ratio_to_one_meter = 1 / (((ne_vector[0]) ** 2 + (ne_vector[1]) ** 2) ** 0.5)
        unit_NE_vector = ne_vector * ratio_to_one_meter
        point1 = now_position + unit_NE_vector * estimate_range
        #oint2 = now_position + unit_NE_vector * to_lidar_range
        return point1, ne_vector, unit_NE_vector

    def sort_closet_to_farthest(self, p, lane):
        d = (((lane[:] - p)[:, 0]) ** 2 + ((lane[:] - p)[:, 1]) ** 2) ** 0.5
        d = d.tolist()
        lane = lane.tolist()
        for idx, i in enumerate(lane):
            i.append(d[idx])
        
        lane = sorted(lane, key = lambda x: x[2])
        
        sorted_lane = []
        for j in lane:
            sorted_lane.append(j[0:2])
        
        return np.array(sorted_lane)

    def find_center_lane_point(self, central_NE, point, ne_vector, unit_ne_vector):
        measure_range = 5 #meter
        '''
        long_measure_range = 4
        lat_measure_range = 5

        unit_ne_vector = np.array(unit_ne_vector)
        anti_unit_ne_vector = np.array((-1) * unit_ne_vector)
        right_ver_unit_ne_vector = np.array([unit_ne_vector[1], (-1) * unit_ne_vector[0]])
        left_ver_unit_ne_vector = np.array([(-1) * unit_ne_vector[1], unit_ne_vector[0]])
        
        left_front_point = point + long_measure_range * unit_ne_vector + lat_measure_range * left_ver_unit_ne_vector
        right_front_point = point + long_measure_range * unit_ne_vector + lat_measure_range * right_ver_unit_ne_vector
        left_rear_point = point + long_measure_range * anti_unit_ne_vector + lat_measure_range * left_ver_unit_ne_vector
        right_rear_point = point + long_measure_range * anti_unit_ne_vector + lat_measure_range * right_ver_unit_ne_vector
        '''
        
        # Find points using circle range
        distance = (((central_NE[:] - point)[:, 0]) ** 2 + ((central_NE[:] - point)[:, 1]) ** 2) ** 0.5 # Distance of central lanes to the base point, which is 4 meters in front of the car.
        lane_point = central_NE[distance < measure_range] # Find the distance which is in range of 4 meters.
        #lane_point = self.sort_closet_to_farthest(point, central_NE[distance < measure_range])
        
        '''
        front_lane = []
        for i in lane_point:
            a_vec = i - point #
            b_vec = ne_vector # Vector of the car's heading
            a_len = (a_vec[0] ** 2 + a_vec[1] ** 2) ** 0.5
            b_len = (b_vec[0] ** 2 + b_vec[1] ** 2) ** 0.5
            cosine = np.dot(a_vec, b_vec) / ((a_len) * (b_len))
            if cosine >= 0:
                front_lane.append(i)
        #print('----------')
        #print(lane_point)
        #print('----------')
        
        front_lane = np.array(front_lane)
        #print(front_lane)
        return front_lane
        '''
        return lane_point
    
    def find_edge_lane_point(self, edge_NE, point, ne_vector, unit_ne_vector):
        measure_range = 4 #meter 
        
        # Find points using circle range
        distance = (((edge_NE[:] - point)[:, 0]) ** 2 + ((edge_NE[:] - point)[:, 1]) ** 2) ** 0.5 # Distance of central lanes to the base point, which is 4 meters in front of the car.
        lane_point = edge_NE[distance < measure_range] # Find the distance which is in range of 4 meters.
        #lane_point = self.sort_closet_to_farthest(point, edge_NE[distance < measure_range])
        
        '''
        front_lane = []
        for i in lane_point:
            a_vec = i - point #
            b_vec = ne_vector # Vector of the car's heading
            a_len = (a_vec[0] ** 2 + a_vec[1] ** 2) ** 0.5
            b_len = (b_vec[0] ** 2 + b_vec[1] ** 2) ** 0.5
            cosine = np.dot(a_vec, b_vec) / ((a_len) * (b_len))
            if cosine >= 0:
                front_lane.append(i)
        #print('----------')
        #print(lane_point)
        #print('----------')
        
        front_lane = np.array(front_lane)
        #print(front_lane)
        return front_lane
        '''
        return lane_point
    

    def HDPipeline_callback(self, img, data):
        
        try:
            #print('-----------------------------------------')
            #rospy.loginfo(data.header)
            if self.first_callback == 1:
                self.first_callback = 0
                self.last_car_position = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z]) # Concern NED
            else:
                
                self.present_car_position = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z]) # Concern NED
                #rospy.loginfo(self.last_car_position)
                #rospy.loginfo(self.present_car_position)
                base_point, NE_vector, unit_NE_vector = self.front_direction_estimation(self.present_car_position[0:2], self.last_car_position[0:2])
                self.last_car_position = self.present_car_position

                #rospy.loginfo(base_point)
                #rospy.loginfo(lidar_point)
                #rospy.loginfo(NE_vector)
                #rospy.loginfo(displacement)

                #rospy.loginfo(self.central_NE)
                #rospy.loginfo(self.edge_NE)

                image_center_lane = []
                image_edge_lane = []
                center_lane_points = self.find_center_lane_point(self.central_NE, base_point, NE_vector, unit_NE_vector)
                edge_lane_points = self.find_edge_lane_point(self.edge_NE, base_point, NE_vector, unit_NE_vector)

                #rospy.loginfo(center_lane_points)

                self.listener.waitForTransform('map', 'camera', data.header.stamp, rospy.Duration(4.0))

                #(trans, rot) = self.listener.lookupTransform('camera', 'map', rospy.Time(0))
                #rospy.loginfo(pose_in_camera)
                
                for j in center_lane_points:
                    
                    center_lane_pose_in_map = PoseStamped()
                    center_lane_pose_in_map.header = data.header
                    center_lane_pose_in_map.pose.position.x = j[0] # Concern NED
                    center_lane_pose_in_map.pose.position.y = j[1] # Concern NED
                    center_lane_pose_in_map.pose.position.z = data.pose.position.z # Concern NED
                    #print(center_lane_pose_in_map)
                    # Frame Transform
                    center_lane_pose_in_camera = self.listener.transformPose('camera', center_lane_pose_in_map)
                    #rospy.loginfo(center_lane_pose_in_camera)
                    imageCoord = np.dot(self.cameraMat, [center_lane_pose_in_camera.pose.position.x, center_lane_pose_in_camera.pose.position.y, center_lane_pose_in_camera.pose.position.z])
                    imageCoord = ((imageCoord / imageCoord[-1])[0:2])
                    '''
                    temp_imageCoord = imageCoord
                    imageCoord = np.zeros(2)
                    imageCoord[0] = temp_imageCoord[1]
                    imageCoord[1] = temp_imageCoord[0]
                    '''
                    #rospy.loginfo(imageCoord)
                    if (imageCoord[0] < self.imageSize[0] and imageCoord[0] > 0) and (imageCoord[1] < self.imageSize[1] and imageCoord[1] > 0):
                        imageCoord[0] = round(imageCoord[0])
                        imageCoord[1] = round(imageCoord[1])
                        image_center_lane.append(imageCoord)
                    imageCoord = []
                    
                image_center_lane = np.array(image_center_lane)
                #rospy.loginfo("Center Lanes:")
                #rospy.loginfo(image_center_lane)
                #print(edge_lane_points)
                
                for m in edge_lane_points:
                    
                    edge_lane_pose_in_map = PoseStamped()
                    edge_lane_pose_in_map.header = data.header
                    edge_lane_pose_in_map.pose.position.x = m[0] # Concern NED
                    edge_lane_pose_in_map.pose.position.y = m[1] # Concern NED
                    edge_lane_pose_in_map.pose.position.z = data.pose.position.z # Concern NED
                    # Frame Transform
                    edge_lane_pose_in_camera = self.listener.transformPose('camera', edge_lane_pose_in_map)
                    #rospy.loginfo(edge_lane_pose_in_camera)
                    imageCoord = np.dot(self.cameraMat, [edge_lane_pose_in_camera.pose.position.x, edge_lane_pose_in_camera.pose.position.y, edge_lane_pose_in_camera.pose.position.z])
                    #rospy.loginfo(imageCoord)
                    imageCoord = ((imageCoord / imageCoord[-1])[0:2])
                    '''
                    temp_imageCoord = imageCoord
                    imageCoord = np.zeros(2)
                    imageCoord[0] = temp_imageCoord[1]
                    imageCoord[1] = temp_imageCoord[0]
                    '''
                    #rospy.loginfo(imageCoord)
                    if (imageCoord[0] < self.imageSize[0] and imageCoord[0] > 0) and (imageCoord[1] < self.imageSize[1] and imageCoord[1] > 0):
                        imageCoord[0] = round(imageCoord[0])
                        imageCoord[1] = round(imageCoord[1])
                        image_edge_lane.append(imageCoord)

                    imageCoord = []
                    
                image_edge_lane = np.array(image_edge_lane)
                #rospy.loginfo("Edge Lanes:")
                #rospy.loginfo(image_edge_lane)
                
                #---------------------------------- Plot the discrepancy lanes
                cv_image = CvBridge().imgmsg_to_cv2(img, "bgr8")
                cv2.resize(cv_image, (self.imageSize[0], self.imageSize[1]))
                #cv_image = cv2.flip(cv_image, 0) # Flip the image vertically
                
                for kdx, k in enumerate(image_center_lane):
                    '''
                    if kdx + 1 == len(image_center_lane):
                        break
                    cv2.line(cv_image, (int(image_center_lane[kdx][0]), int(image_center_lane[kdx][1])), \
                    (int(image_center_lane[kdx + 1][0]), int(image_center_lane[kdx + 1][1])), (0, 100, 255), thickness = 2)
                    '''
                    #image_center_lane[kdx][0] = image_center_lane[kdx][0] - 50
                    #image_center_lane[kdx][1] = image_center_lane[kdx][1] - 400
                    cv2.circle(cv_image, (int(image_center_lane[kdx][0]), int(image_center_lane[kdx][1])), 5, (0, 100, 255), thickness = 5)
                
                for ldx, l in enumerate(image_edge_lane):
                    
                    '''
                    if ldx + 1 == len(image_edge_lane):
                        break
                    cv2.line(cv_image, (int(image_edge_lane[ldx][0]), int(image_edge_lane[ldx][1])), \
                    (int(image_edge_lane[ldx + 1][0]), int(image_edge_lane[ldx + 1][1])), (0, 100, 255), thickness = 2)
                    '''
                    #image_edge_lane[ldx][0] = image_edge_lane[ldx][0] - 50
                    #image_edge_lane[ldx][1] = image_edge_lane[ldx][1] - 400
                    cv2.circle(cv_image, (int(image_edge_lane[ldx][0]), int(image_edge_lane[ldx][1])), 5, (0, 255, 0), thickness = 5)

                #---------------------------------- Publish the output image
                try:
                    #cv_image = cv2.flip(cv_image, 0) # Flip the image vertically
                    self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, 'bgr8'))
                except CvBridgeError as e:
                    print(e)
                
            #print('-----------------------------------------')

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) or CvBridgeError as e:
            print(e)
        '''
        rate = rospy.Rate(10.0)
        while not rospy.is_shutdown():
            try:
                cv_image = CvBridge().imgmsg_to_cv2(img, "bgr8")
                (trans, rot) = self.listener.lookupTransform('camera', 'map', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) or CvBridgeError as e:
                print(e)
                continue

            rate.sleep()
        '''
        

    def main(self):
        
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('laneNet_discrepancy_node', anonymous=True)

    HP = HDPipeline()
    try:
        HP.main()
    except rospy.ROSInterruptException:
        pass
    
    
    
