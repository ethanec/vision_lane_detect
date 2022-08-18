#!/usr/bin/env python

import rospy
import message_filters
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import Bool

from vision_lane_detect.msg import Int32MultiArray_H
from vision_lane_detect.msg import Float32MultiArray_H

from math import *
import matplotlib.pyplot as plt
import numpy as np
import random
import csv

from utils import lateral_distance


class KalmanFilter():

    def __init__(self):
        car_position_topic_name = '/xsens/gps/pose'
        whiteline_world_points_topic_name = 'world_line_points_topic'  # four coordinate x, y, z
        #detection_image_points_topic_name = 'erfNet_line_points_topic'
        detection_image_points_topic_name = 'opencv_image_line_points_topic' # detected x, y coordiante of two lane
        world_lane_points_topic_name = 'world_lane_points_topic' # one coordinate x, y, z and sometimes shift
        
        rospy.loginfo('Setting topic to %s', car_position_topic_name)
        rospy.loginfo('Setting topic to %s', whiteline_world_points_topic_name)
        rospy.loginfo('Setting topic to %s', detection_image_points_topic_name)
        rospy.loginfo('Setting topic to %s', world_lane_points_topic_name)

        rospy.loginfo('Waiting for topics...')
        rospy.wait_for_message(car_position_topic_name, PoseStamped)
        rospy.wait_for_message(world_lane_points_topic_name, PoseStamped)
        #rospy.wait_for_message(whiteline_world_points_topic_name, Float32MultiArray)
        #rospy.wait_for_message(detection_image_points_topic_name, Int32MultiArray)
        rospy.wait_for_message(whiteline_world_points_topic_name, Float32MultiArray_H)
        rospy.wait_for_message(detection_image_points_topic_name, Int32MultiArray_H)
        rospy.loginfo('Time synchronize!')
        #---------------------------------- Time Synchronizer
        self.car_position_sub = message_filters.Subscriber(car_position_topic_name, PoseStamped)
        #self.whiteline_world_points_sub = message_filters.Subscriber(whiteline_world_points_topic_name, Float32MultiArray)
        #self.detection_image_points_sub = message_filters.Subscriber(detection_image_points_topic_name, Int32MultiArray)
        self.whiteline_world_points_sub = message_filters.Subscriber(whiteline_world_points_topic_name, Float32MultiArray_H)
        self.detection_image_points_sub = message_filters.Subscriber(detection_image_points_topic_name, Int32MultiArray_H)
        self.world_lane_points_sub = message_filters.Subscriber(world_lane_points_topic_name, PoseStamped)
        ts = message_filters.ApproximateTimeSynchronizer([self.car_position_sub, self.whiteline_world_points_sub, \
            self.detection_image_points_sub, self.world_lane_points_sub], 1, 1, allow_headerless = False)
        ts.registerCallback(self.KalmanFilter_callback)
        #rospy.Subscriber(whiteline_world_points_topic_name, Float32MultiArray, self.callback)
        #test = message_filters.ApproximateTimeSynchronizer([self.car_position_sub, self.world_lane_points_sub], 1, 1, allow_headerless = True)
        #test.registerCallback(self.callback)
        
        self.lat_distance = [1.6262, 1.6262]
        self.last_lat_distance = [1.6262, 1.6262]
        self.center_point = np.zeros(2)
        self.last_center_point = np.zeros(2)
        self.last_last_center_point = np.zeros(2)
        self.xsens_position = np.zeros(2)
        self.last_xsens_position = np.zeros(2)
        self.last_last_xsens_position = np.zeros(2)
        self.last_last_last_xsens_position = np.zeros(2)
        self.first_callback = 0
        self.callback_times = 0
        self.selected_right_line = None
        self.selected_left_line = None
        self.last_selected_right_line = None
        self.last_selected_left_line = None

        self.car_position_KF = np.zeros((1, 2))
        self.car_corvarince_KF = []
        self.last_four_car_position_measurement = []
        self.observation_matrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 1, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0]])
        
        self.transition = np.array([[1.52529103, 0, -0.08417148, 0, -0.44112014, 0],
                                    [0, 2.10470444, 0, -1.21387597, 0,   0.1091747],
                                    [1,          0, 0,           0, 0,           0],
                                    [0,          1, 0,           0, 0,           0],
                                    [0,          0, 1,           0, 0,           0],
                                    [0,          0, 0,           1, 0,           0]]) ## 1.58435104, -0.18515405, -0.39919775
        '''
        self.transition = np.array([[1.71619066, 0, -0.44376319, 0, -0.27242998, 0],
                                    [0, 1.86885376, 0, -0.74109612, 0,  -0.1277627],
                                    [1,          0, 0,           0, 0,           0],
                                    [0,          1, 0,           0, 0,           0],
                                    [0,          0, 1,           0, 0,           0],
                                    [0,          0, 0,           1, 0,           0]])
        '''
        self.save_ndt_position = []
        self.save_xsens_position = []
        self.save_covariance = []
        self.save_rtk_position = []
        self.save_kf_position = []
        self.save_projected_heading = []
        self.save_selected_right_lines = []
        self.save_selected_left_lines = []
        self.save_lateral_distance = []
        
        
        self.x_A = []
        self.x_b = []
        self.y_A = []
        self.y_b = []

        self.error = []

        self.kf_pub = rospy.Publisher('KalmanFilter_pose', PoseStamped, queue_size = 10)
        self.kf_pose = PoseStamped()

    #def callback(self, data1, data2):
    #    print(data1)
    #    print(data2)

    def KalmanFilter_callback(self, car_position, whiteline_world_points, detection_image_points, world_lane_points):
        ### ---------------------------------- Adjust the list for storing the past car's position
        #self.adjust_last_four_position(car_position.pose.position.x, car_position.pose.position.y)
        self.xsens_position = np.array([car_position.pose.position.x, car_position.pose.position.y])
        
        self.x_A.append(np.array([self.last_xsens_position[0], self.last_last_xsens_position[0], self.last_last_last_xsens_position[0]]))
        self.y_A.append(np.array([self.last_xsens_position[1], self.last_last_xsens_position[1], self.last_last_last_xsens_position[1]]))
        self.x_b.append(np.array(self.xsens_position[0]))
        self.y_b.append(np.array(self.xsens_position[1]))
            
        ### ---------------------------------- Computed lateral distance
        self.lat_distance = lateral_distance.LateralDistance().calculate_lateral_distance(car_position, whiteline_world_points, detection_image_points)
        
        if self.lat_distance is None:
            self.lat_distance = self.last_lat_distance
            self.center_point = self.last_center_point
        if self.first_callback == 0:
            self.last_lat_distance = self.lat_distance
            self.last_center_point = self.center_point
            self.last_xsens_position = self.xsens_position
            self.first_callback = 1

        self.center_point = np.array([world_lane_points.pose.position.x, world_lane_points.pose.position.y])
        if self.center_point[0] == self.last_center_point[0] and self.center_point[1] == self.last_center_point[1]:
            vec = self.xsens_position - self.last_xsens_position
        else:
            vec = self.center_point - self.last_center_point
        unit_vec = vec / np.sqrt(vec[0] ** 2 + vec[1] ** 2)
        for i in range(0, whiteline_world_points.layout.dim[0].size):
            if len(whiteline_world_points.data) >= 6:
                x1 = whiteline_world_points.data[i * 6 + 0]
                y1 = whiteline_world_points.data[i * 6 + 1]
                z1 = whiteline_world_points.data[i * 6 + 2]
                x2 = whiteline_world_points.data[i * 6 + 3]
                y2 = whiteline_world_points.data[i * 6 + 4]
                z2 = whiteline_world_points.data[i * 6 + 5]
                if i == 0:
                    self.selected_right_line = np.array([[x1, y1, z1], [x2, y2, z2]])
                else:
                    self.selected_left_line = np.array([[x1, y1, z1], [x2, y2, z2]])
        

        if self.selected_right_line is not None and self.selected_left_line is not None:
            
            #self.last_four_car_position_measurement.append(self.center_point)
            
            ### ---------------------------------- Xsens as the raw state
            vec = self.center_point - self.last_center_point
            d_to_lane = abs(vec[1] * self.xsens_position[0] - vec[0] * self.xsens_position[1] + \
                      (-1 * vec[1] * self.center_point[0] + vec[0] * self.center_point[1])) / np.sqrt(vec[0] ** 2 + vec[1] ** 2)
            if d_to_lane > 2:
                self.last_four_car_position_measurement.append(self.center_point)
            else:
                self.last_four_car_position_measurement.append(self.xsens_position)
            
            #self.last_four_car_position_measurement.append(self.xsens_position)

            RL = self.bivariate_linear_equation(self.selected_right_line[0], self.selected_right_line[1]) # y = ax + b -> np.array([[a], [b]])
            LL = self.bivariate_linear_equation(self.selected_left_line[0], self.selected_left_line[1]) # y = ax + b -> np.array([[a], [b]])
            
            if self.Distance([self.xsens_position[0], self.xsens_position[1]], RL) > 3 and self.last_selected_right_line is not None:
                r_vec = np.array([self.selected_right_line[1, 0] - self.selected_right_line[0, 0], self.selected_right_line[1, 1] - self.selected_right_line[0, 1]])
                unit_r_vec = r_vec / np.sqrt(r_vec[0] ** 2 + r_vec[1] ** 2)
                right_normal = np.array([unit_r_vec[1], -1 * unit_r_vec[0], 0])
                r_anchor = np.array([self.center_point[0] + 1.6262 * unit_vec[1], self.center_point[1] + 1.6262 * -1 * unit_vec[0]])
                right_normal[2] = -1 * right_normal[0] * r_anchor[0] - right_normal[1] * r_anchor[1]
                RL[0, 0] = right_normal[0] * -1 / right_normal[1]
                RL[1, 0] = right_normal[2] * -1 / right_normal[1]
            else:
                right_normal = np.array([RL[0, 0] / np.sqrt(1 + RL[0, 0] ** 2), -1 / np.sqrt(1 + RL[0, 0] ** 2), RL[1, 0] / np.sqrt(1 + RL[0, 0] ** 2)])
            if self.Distance([self.xsens_position[0], self.xsens_position[1]], LL) > 3 and self.last_selected_left_line is not None:
                l_vec = np.array([self.selected_left_line[1, 0] - self.selected_left_line[0, 0], self.selected_left_line[1, 1] - self.selected_left_line[0, 1]])
                unit_l_vec = l_vec / np.sqrt(l_vec[0] ** 2 + l_vec[1] ** 2)
                left_normal = np.array([unit_l_vec[1], -1 * unit_l_vec[0], 0])
                l_anchor = np.array([self.center_point[0] - 1.6262 * unit_vec[1], self.center_point[1] - 1.6262 * -1 * unit_vec[0]])
                left_normal[2] = -1 * left_normal[0] * l_anchor[0] - left_normal[1] * l_anchor[1]
                LL[0, 0] = left_normal[0] * -1 / left_normal[1]
                LL[1, 0] = left_normal[2] * -1 / left_normal[1]
            else:
                left_normal = np.array([LL[0, 0] / np.sqrt(1 + LL[0, 0] ** 2), -1 / np.sqrt(1 + LL[0, 0] ** 2), LL[1, 0] / np.sqrt(1 + LL[0, 0] ** 2)])
            
            
            if len(self.last_four_car_position_measurement) >= 7:
                
                #right_normal[2] = self.lat_distance[1]
                #left_normal[2] = self.lat_distance[0]
                #print(right_normal)
                #print(left_normal)
                #################################################################
                ######################### Kalman Filter #########################
                #################################################################
                X_previous = self.car_position_KF[0:3].reshape(6,1).ravel()
                X_predicted = np.dot(self.transition, X_previous)
                Q = np.array([[5.56312142239, 0, 0, 0, 0, 0],
                              [0, 6.83361335893, 0, 0, 0, 0],
                              [0, 0, 5.56312142239, 0, 0, 0],
                              [0, 0, 0, 6.83361335893, 0, 0],
                              [0, 0, 0, 0, 5.56312142239, 0],
                              [0, 0 ,0, 0, 0, 6.83361335893]])## 5.56312142239, 6.83361335893
                                                              ## 2.30062452698, 1.63990116538
                P_predicted = self.car_corvarince_KF[-1] + Q if len(self.car_corvarince_KF) > 0 \
                                                             else np.dot(np.dot(self.transition, np.diag([1] * 6)), self.transition.transpose(1, 0)) + Q
                #X_1 = np.array(self.last_four_car_position_measurement[-6:-1])
                #X_2 = np.array(self.last_four_car_position_measurement[-7:-2])
                
                if X_predicted[0] * right_normal[0] + X_predicted[1] * right_normal[1] + right_normal[2] < 0:
                    right_normal = -1 * right_normal
                if X_predicted[0] * left_normal[0] + X_predicted[1] * left_normal[1] + left_normal[2] < 0:
                    left_normal = -1 * left_normal
                H = np.array([[1, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0],
                              [left_normal[0], left_normal[1], 0, 0, 0, 0],
                              [right_normal[0], right_normal[1], 0, 0, 0, 0]])
                
                self.error.append(np.array([self.last_four_car_position_measurement[-1][0] - X_predicted[0], \
                                            self.last_four_car_position_measurement[-1][1] - X_predicted[1]]))

                R = [1, 1, 0.4, 0.4]
                if self.lat_distance[0] > 3:
                    self.lat_distance[0] = 1.6262
                if self.lat_distance[1] > 3:
                    self.lat_distance[1] = 1.6262
                X_measurement = [self.last_four_car_position_measurement[-1][0], self.last_four_car_position_measurement[-1][1], \
                    self.lat_distance[0] - left_normal[2], self.lat_distance[1] - right_normal[2]]
                    

                Kg = np.dot(np.dot(P_predicted, H.transpose(1, 0)), np.linalg.inv(np.dot(np.dot(H, P_predicted), H.transpose(1, 0)) + np.diag(R)))
                X_updated = X_predicted + np.dot(Kg, X_measurement - np.dot(H, X_predicted))
                #print('Kalman Gain: {}'.format(Kg))
                #print('Actual Measurement: {}'.format(X_measurement))
                #print('Predicted Measurement: {}'.format(np.dot(H, X_predicted)))
                #print('X_updated: {}'.format(X_updated))
                #print('Weighted Residual: {}'.format(np.dot(Kg, X_measurement - np.dot(H, X_predicted))))
                P_updated = np.dot(np.eye(6) - np.dot(Kg, H), P_predicted)
                
                ### ---------------------------------- Record data
                self.car_position_KF = np.concatenate((X_updated.reshape(3, 2)[0:1], self.car_position_KF))
                self.car_corvarince_KF.append(P_updated)
                
                raw_x, raw_y = car_position.pose.position.x, car_position.pose.position.y
                kf_x, kf_y = self.car_position_KF[0, 0], self.car_position_KF[0, 1]
                pro_x, pro_y = self.center_point[0], self.center_point[1]
                print('---------------------------------------------------')
                print('Xsens Lateral Distance: {} m'.format(self.Distance([raw_x, raw_y], RL)))
                #print('Projected Lateral Distance: {} m'.format(self.Distance([pro_x, pro_y], RL)))
                print('Localized Lateral Distance: {} m'.format(self.Distance([kf_x, kf_y], RL)))
                print('---------------------------------------------------')

                #############
                ## Publish ##
                #############
                self.kf_pose.header.frame_id = car_position.header.frame_id
                self.kf_pose.header.stamp = car_position.header.stamp
                self.kf_pose.pose.position.x = kf_x
                self.kf_pose.pose.position.y = kf_y
                self.kf_pose.pose.position.z = car_position.pose.position.z
                self.kf_pose.pose.orientation.x = 0
                self.kf_pose.pose.orientation.y = 0
                self.kf_pose.pose.orientation.z = 0
                self.kf_pose.pose.orientation.w = 1
                self.kf_pub.publish(self.kf_pose)

                #print(self.car_position_KF[0])
                self.save_selected_right_lines.append(self.selected_right_line)
                self.save_selected_left_lines.append(self.selected_left_line)
                self.save_xsens_position.append(np.array([raw_x, raw_y]))
                self.save_covariance.append(np.array([P_updated[0, 0], P_updated[0, 1], P_updated[1, 0], P_updated[1, 1]]))
                self.save_kf_position.append(np.array([kf_x, kf_y]))
                self.save_projected_heading.append(np.array([pro_x, pro_y]))
                self.save_lateral_distance.append(np.array([self.lat_distance[0], self.lat_distance[1]]))
            else:
                self.car_position_KF = np.concatenate(([self.last_four_car_position_measurement[-1]], self.car_position_KF))
                
                if self.callback_times == 0:
                    self.car_position_KF = np.delete(self.car_position_KF, -1, 0)

        self.last_last_center_point = self.last_center_point
        self.last_center_point = self.center_point
        self.last_last_last_xsens_position = self.last_last_xsens_position
        self.last_last_xsens_position = self.last_xsens_position
        self.last_xsens_position = self.xsens_position
        self.last_lat_distance = self.lat_distance            
        self.last_selected_right_line = self.selected_right_line
        self.last_selected_left_line = self.selected_left_line
        self.callback_times = self.callback_times + 1

    def bivariate_linear_equation(self, p1, p2):
        A = np.array([[p1[0], 1],
                      [p2[0], 1]])
        if np.linalg.det(A) != 0: # caluclate determinate
            B = np.array([p1[1], p2[1]]).reshape(2, 1)
            A_inv = np.linalg.inv(A)
            C = A_inv.dot(B)
            return C
        else:
            A = np.array([[p1[0] + 1, 1],
                          [p2[0], 1]])
            B = np.array([p1[1], p2[1]]).reshape(2, 1)
            A_inv = np.linalg.inv(A)
            C = A_inv.dot(B)
            return C
    
    def Distance(self, point, line_coefficients):
        up = abs(point[0] * line_coefficients[0, 0] - 1 * point[1] + line_coefficients[1, 0])
        down = sqrt(line_coefficients[0, 0] ** 2 + 1)
        return up / down

    def main(self):
        rospy.spin()
        ################################################################
        ## Calculate Transition Parameters and Process Noise Variance ##
        ################################################################
        self.x_A = np.array(self.x_A)
        self.x_b = np.array(self.x_b)
        self.y_A = np.array(self.y_A)
        self.y_b = np.array(self.y_b)
        self.x_A = self.x_A[2:]
        self.x_b = self.x_b[2:]
        self.y_A = self.y_A[2:]
        self.y_b = self.y_b[2:]
        transition_x_param = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(self.x_A), self.x_A)), np.transpose(self.x_A)), self.x_b)
        transition_y_param = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(self.y_A), self.y_A)), np.transpose(self.y_A)), self.y_b)
        print('Transition X Parameters: {}'.format(transition_x_param))
        print('Transition Y Parameters: {}'.format(transition_y_param))
        self.error = np.array(self.error)
        print('X Variance: {}'.format(np.std(self.error[:, 0]) ** 2))
        print('Y Variance: {}'.format(np.std(self.error[:, 1]) ** 2))
        
        save_data_path = "/home/meclab/autoware.ai/src/autoware/core_perception/vision_lane_detect/script/self_localization/result"
        out1 = csv.writer(open(save_data_path + "/ndt_position.csv","w"), delimiter=',')
        for r in self.save_ndt_position:
            out1.writerow(r)
        out2 = csv.writer(open(save_data_path + "/xsens_position.csv","w"), delimiter=',')
        for r in self.save_xsens_position:
            out2.writerow(r)
        out3 = csv.writer(open(save_data_path + "/kf_position.csv","w"), delimiter=',')
        for r in self.save_kf_position:
            out3.writerow(r)
        out4 = csv.writer(open(save_data_path + "/projected_heading.csv","w"), delimiter=',')
        for r in self.save_projected_heading:
            out4.writerow(r)
        out5 = csv.writer(open(save_data_path + "/selected_right_lines.csv","w"), delimiter=',')
        for l in self.save_selected_right_lines:
            r = l.reshape((1, 6)).ravel()
            out5.writerow(r)
        out6 = csv.writer(open(save_data_path + "/selected_left_lines.csv","w"), delimiter=',')
        for l in self.save_selected_left_lines:
            r = l.reshape((1, 6)).ravel()
            out6.writerow(r)
        out7 = csv.writer(open(save_data_path + "/lateral_distance.csv","w"), delimiter=',')
        for r in self.save_lateral_distance:
            out7.writerow(r)
        out8 = csv.writer(open(save_data_path + "/rtk_position.csv","w"), delimiter=',')
        for r in self.save_rtk_position:
            out8.writerow(r)
        out9 = csv.writer(open(save_data_path + "/covariance.csv","w"), delimiter=',')
        for r in self.save_covariance:
            out9.writerow(r)
        
        


if __name__ == '__main__':
    #rospy.wait_for_message('gnss_stat', Bool)
    rospy.init_node('KF_lane_self_localization_node')
    KF = KalmanFilter()
    
    try:
        KF.main()
    except rospy.ROSInterruptException:
        pass