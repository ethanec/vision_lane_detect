#!/usr/bin/env python

import rospy
import message_filters
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension

from math import *
import matplotlib.pyplot as plt
import numpy as np
import random
import csv

from utils import lateral_distance


class KalmanFilter():

    def __init__(self):
        gt_car_position_topic_name = '/ndt_pose'
        car_position_topic_name = '/xsens/gps/pose'
        rtk_topic_name = '/novatel/RTK/pose'
        projection_world_points_topic_name = 'world_line_points_topic'
        projection_image_points_topic_name = 'image_line_points_topic'
        #detection_image_points_topic_name = 'erfNet_line_points_topic'
        detection_image_points_topic_name = 'opencv_image_line_points_topic'
        world_lane_points_topic_name = 'world_lane_points_topic'
        
        rospy.loginfo('Setting topic to %s', gt_car_position_topic_name)
        rospy.loginfo('Setting topic to %s', car_position_topic_name)
        rospy.loginfo('Setting topic to %s', rtk_topic_name)
        rospy.loginfo('Setting topic to %s', projection_world_points_topic_name)
        rospy.loginfo('Setting topic to %s', projection_image_points_topic_name)
        rospy.loginfo('Setting topic to %s', detection_image_points_topic_name)
        rospy.loginfo('Setting topic to %s', world_lane_points_topic_name)

        #---------------------------------- Time Synchronizer
        
        self.gt_car_position_sub = message_filters.Subscriber(gt_car_position_topic_name, PoseStamped)
        self.car_position_sub = message_filters.Subscriber(car_position_topic_name, PoseStamped)
        self.rtk_sub = message_filters.Subscriber(rtk_topic_name, PoseStamped)
        self.projection_world_points_sub = message_filters.Subscriber(projection_world_points_topic_name, Float32MultiArray)
        self.projection_image_points_sub = message_filters.Subscriber(projection_image_points_topic_name, Int32MultiArray)
        self.detection_image_points_sub = message_filters.Subscriber(detection_image_points_topic_name, Int32MultiArray)
        self.world_lane_points_sub = message_filters.Subscriber(world_lane_points_topic_name, PoseStamped)
        ts = message_filters.ApproximateTimeSynchronizer([self.gt_car_position_sub, self.car_position_sub, self.rtk_sub, \
            self.projection_world_points_sub, self.projection_image_points_sub, self.detection_image_points_sub, self.world_lane_points_sub], \
                1, 1, allow_headerless=True) ### Here first 1 is the size of queue and next 1 is the time in sec to consider for aprox
        ts.registerCallback(self.KalmanFilter_callback)

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
        
        self.save_ndt_position = []
        self.save_xsens_position = []
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

    def KalmanFilter_callback(self, gt_car_position, car_position, rtk_position, projection_world_points, projection_image_points, detection_image_points, world_lane_points):
        #print('x:{}, y:{}, z:{}'. format(gt_car_position.pose.position.x - car_position.pose.position.x, \
        #                                gt_car_position.pose.position.y - car_position.pose.position.y, \
        #                                gt_car_position.pose.position.z - car_position.pose.position.z))
        
        ### ---------------------------------- Adjust the list for storing the past car's position
        
        #self.adjust_last_four_position(car_position.pose.position.x, car_position.pose.position.y)
        self.xsens_position = np.array([car_position.pose.position.x, car_position.pose.position.y])
        
        self.x_A.append(np.array([self.last_xsens_position[0], self.last_last_xsens_position[0], self.last_last_last_xsens_position[0]]))
        self.y_A.append(np.array([self.last_xsens_position[1], self.last_last_xsens_position[1], self.last_last_last_xsens_position[1]]))
        self.x_b.append(np.array(self.xsens_position[0]))
        self.y_b.append(np.array(self.xsens_position[1]))
        
        if (len(projection_world_points.data) / 3) == (len(projection_image_points.data) / 2) \
            and (len(projection_world_points.data) / 3) >= 4 \
            and (len(projection_image_points.data) / 2) >= 4:
            
            ### ---------------------------------- Fetch the projection result and the computed lateral distance
            self.lat_distance = lateral_distance.LateralDistance().calculate_lateral_distance(car_position, projection_world_points, projection_image_points, detection_image_points)
            
            if self.lat_distance is None:
                self.lat_distance = self.last_lat_distance
                self.center_point = self.last_center_point
            if self.first_callback == 0:
                self.last_lat_distance = self.lat_distance
                self.last_center_point = self.center_point
                self.last_xsens_position = self.xsens_position
                self.first_callback = 1

            self.center_point = np.array([world_lane_points.pose.position.x, world_lane_points.pose.position.y])
            vec = self.center_point - self.last_center_point # Vector of the car's heading
            #vec = self.xsens_position - self.last_xsens_position # Vector of the car's heading
            perpendicular_vec = [vec[1], -vec[0]]

            C = self.bivariate_linear_equation(self.center_point, self.last_center_point)
            #C = self.bivariate_linear_equation(self.xsens_position, self.last_xsens_position)
            ### ---------------------------------- Distinguish the left lines and right lines
            right_lines, right_closest_distance, left_lines, left_closest_distance = self.find_lines(projection_world_points, C, perpendicular_vec)

            ### ---------------------------------- Check the lines' vector is in same direction with car's heading
            if len(right_lines) > 0 and len(right_closest_distance) > 0:
                self.selected_right_line = right_lines[np.argmin(right_closest_distance)]
                if np.dot(self.selected_right_line[0, 0:2] - self.selected_right_line[1, 0:2], vec) < 0:
                    self.selected_right_line = np.flip(self.selected_right_line, 0)
                #print(self.selected_right_line)
            else:
                self.selected_right_line = self.last_selected_right_line
            if len(left_lines) > 0 and len(left_closest_distance) > 0:
                self.selected_left_line = left_lines[np.argmin(left_closest_distance)]
                if np.dot(self.selected_left_line[0, 0:2] - self.selected_left_line[1, 0:2], vec) < 0:
                    self.selected_left_line = np.flip(self.selected_left_line, 0)
                #print(self.selected_left_line)
            else:
                self.selected_left_line = self.last_selected_left_line

        else:
            #print(len(projection_world_points.data) / 3)
            #print(len(projection_image_points.data) / 2)
            print('Data too short or doesn\'t match!')
            self.selected_right_line = self.last_selected_right_line
            self.selected_left_line = self.last_selected_left_line

        if self.selected_right_line is not None and self.selected_left_line is not None:
            #self.last_four_car_position_measurement.append(self.center_point)
            ### ---------------------------------- Xsens as the raw state
            self.last_four_car_position_measurement.append(self.xsens_position) 

            if len(self.last_four_car_position_measurement) >= 7:
                RL = self.bivariate_linear_equation(self.selected_right_line[0], self.selected_right_line[1])
                LL = self.bivariate_linear_equation(self.selected_left_line[0], self.selected_left_line[1])
                right_normal = np.array([RL[0, 0] / np.sqrt(1 + RL[0, 0] ** 2), -1 / np.sqrt(1 + RL[0, 0] ** 2), RL[1, 0] / np.sqrt(1 + RL[0, 0] ** 2)])
                left_normal = np.array([LL[0, 0] / np.sqrt(1 + LL[0, 0] ** 2), -1 / np.sqrt(1 + LL[0, 0] ** 2), LL[1, 0] / np.sqrt(1 + LL[0, 0] ** 2)])
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
                ''' Method 1
                X_1 = np.array(self.car_position_KF[0:5])
                X_2 = np.array(self.car_position_KF[1:6])
                v1_x, v1_y = np.std(X_1[:, 0]) ** 2, np.std(X_1[:, 1]) ** 2
                v2_x, v2_y = np.std(X_2[:, 0]) ** 2, np.std(X_2[:, 1]) ** 2
                
                F = np.array([[(v2_x / (v1_x + v2_x)), 0, (v1_x / (v1_x + v2_x)), 0, 0, 0],
                              [0, (v2_y / (v1_y + v2_y)), 0, (v1_y / (v1_y + v2_y)), 0, 0],
                              [1, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0]])
                X_previous = self.car_position_KF[0:3].reshape(6,1).ravel()
                
                X_predicted = np.dot(F, X_previous)
                P_predicted = self.car_corvarince_KF[-1] + np.diag([1] * 6) if len(self.car_corvarince_KF) > 0 \
                    else np.dot(np.dot(F, np.diag(np.array([v1_x, v1_y, v2_x, v2_y, 0, 0]))), F.transpose(1, 0)) + np.diag([1] * 6)
                '''
                '''### ---------------------------------- This part may be fitted to the theorem.
                if X_predicted[0] * right_normal[0] + X_predicted[1] * right_normal[1] < 0:
                    right_normal = -1 * right_normal
                if X_predicted[0] * left_normal[0] + X_predicted[1] * left_normal[1] < 0:
                    left_normal = -1 * left_normal
                H = np.array([[1, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0],
                              [left_normal[0], left_normal[1], 0, 0, 0, 0],
                              [right_normal[0], right_normal[1], 0, 0, 0, 0]])
                print('Left Line: {}'.format(left_normal))
                print('Right Line: {}'.format(right_normal))
                R = [2, 2, 0.4, 0.4]
                if self.lat_distance[0] > 3:
                    self.lat_distance[0] = 1.6262
                if self.lat_distance[1] > 3:
                    self.lat_distance[1] = 1.6262
                
                X_measurement = [self.last_four_car_position_measurement[-1][0], self.last_four_car_position_measurement[-1][1], \
                    self.lat_distance[0] + abs(left_normal[2]), -1 * self.lat_distance[1] + abs(right_normal[2])]
                '''
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
                print('Actual Measurement: {}'.format(X_measurement))
                print('Predicted Measurement: {}'.format(np.dot(H, X_predicted)))
                print('X_updated: {}'.format(X_updated))
                #print('Weighted Residual: {}'.format(np.dot(Kg, X_measurement - np.dot(H, X_predicted))))
                P_updated = np.dot(np.eye(6) - np.dot(Kg, H), P_predicted)
                
                ### ---------------------------------- Record data
                self.car_position_KF = np.concatenate((X_updated.reshape(3, 2)[0:1], self.car_position_KF))
                self.car_corvarince_KF.append(P_updated)
                
                gt_x, gt_y = gt_car_position.pose.position.x, gt_car_position.pose.position.y
                print('NDT: [{}, {}]'.format(gt_x, gt_y))
                raw_x, raw_y = car_position.pose.position.x, car_position.pose.position.y
                kf_x, kf_y = self.car_position_KF[0, 0], self.car_position_KF[0, 1]
                pro_x, pro_y = self.center_point[0], self.center_point[1]
                #original_distance = np.sqrt((gt_x - raw_x) ** 2 + (gt_y - raw_y) ** 2)
                #localized_distance = np.sqrt((gt_x - kf_x) ** 2 + (gt_y - kf_y) ** 2)
                print('---------------------------------------------------')
                print('Original Lateral Distance: {} m'.format(self.Distance([gt_x, gt_y], RL)))
                print('Xsens Lateral Distance: {} m'.format(self.Distance([raw_x, raw_y], RL)))
                #print('Projected Lateral Distance: {} m'.format(self.Distance([pro_x, pro_y], RL)))
                print('Localized Lateral Distance: {} m'.format(self.Distance([kf_x, kf_y], RL)))
                print('---------------------------------------------------')
                #print(self.car_position_KF[0])
                self.save_selected_right_lines.append(self.selected_right_line)
                self.save_selected_left_lines.append(self.selected_left_line)
                self.save_ndt_position.append(np.array([gt_x, gt_y]))
                self.save_xsens_position.append(np.array([raw_x, raw_y]))
                self.save_rtk_position.append(np.array([rtk_position.pose.position.x, rtk_position.pose.position.y]))
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

    '''
    def update_center_point(self):
        #print('updating......................................................................')
        #v = self.xsens_position - self.last_xsens_position
        #_v = self.last_xsens_position - self.last_last_xsens_position
        #a = v - _v
        #C = self.bivariate_linear_equation(self.last_center_point, self.last_last_center_point)
        #if abs(C[0] * self.center_point[0] - 1 * self.center_point[1] + C[1]) / np.sqrt(pow(C[0], 2) + 1) > 1.6262:
        #    self.center_point = (self.last_xsens_position + self.last_center_point) / 2 + _v + 0.5 * a
        #else:
        #    self.center_point = self.last_center_point + _v + 0.5 * a

        v = self.center_point - self.last_center_point
        _v = self.last_center_point - self.last_last_center_point
        a = v - _v
        C = self.bivariate_linear_equation(self.last_center_point, self.last_last_center_point)
        self.center_point = self.last_center_point + _v + 0.5 * a
    '''
    def find_lines(self, projection_world_points, C, perpendicular_vec):
        left_lines = []
        right_lines = []
        left_closest_distance = []
        right_closest_distance = []
        for i in range(0, projection_world_points.layout.dim[0].size):
            if len(projection_world_points.data) >= 6:
                x1 = projection_world_points.data[i * 6 + 0]
                y1 = projection_world_points.data[i * 6 + 1]
                z1 = projection_world_points.data[i * 6 + 2]
                x2 = projection_world_points.data[i * 6 + 3]
                y2 = projection_world_points.data[i * 6 + 4]
                z2 = projection_world_points.data[i * 6 + 5]

                vec1 = [x1 - self.center_point[0], y1 - self.center_point[1]]
                vec2 = [x2 - self.center_point[0], y2 - self.center_point[1]]
                #vec1 = [x1 - self.xsens_position[0], y1 - self.xsens_position[1]]
                #vec2 = [x2 - self.xsens_position[0], y2 - self.xsens_position[1]]

                point2line_distance1 = abs(C[0] * x1 - 1 * y1 + C[1]) / np.sqrt(pow(C[0], 2) + 1)
                point2line_distance2 = abs(C[0] * x2 - 1 * y2 + C[1]) / np.sqrt(pow(C[0], 2) + 1)
                point2point_distance1 = np.sqrt((x1 - self.center_point[0]) ** 2 + (y1 - self.center_point[1]) ** 2)
                point2point_distance2 = np.sqrt((x2 - self.center_point[0]) ** 2 + (y2 - self.center_point[1]) ** 2)
                #point2point_distance1 = np.sqrt((x1 - self.xsens_position[0]) ** 2 + (y1 - self.xsens_position[1]) ** 2)
                #point2point_distance2 = np.sqrt((x2 - self.xsens_position[0]) ** 2 + (y2 - self.xsens_position[1]) ** 2)
                A = self.center_point - self.last_center_point
                #A = self.xsens_position - self.last_xsens_position
                B = np.array([x1 - x2, y1 - y2])
                line_angle = abs(np.dot(A, B) / (np.sqrt(pow(A[0], 2) + pow(A[1], 2)) * np.sqrt(pow(B[0], 2) + pow(B[1], 2))))

                if (np.dot(vec1, perpendicular_vec) > 0 and point2line_distance1 < 2) and (np.dot(vec2, perpendicular_vec) > 0 and point2line_distance2 < 2) \
                    and line_angle >= 0.966:
                #if np.dot(vec1, perpendicular_vec) > 0 and np.dot(vec2, perpendicular_vec) > 0 and min(point2point_distance1, point2point_distance2) < 5:
                    right_lines.append(np.array([[x1, y1, z1], [x2, y2, z2]]))
                    right_closest_distance.append(min(point2point_distance1, point2point_distance2))

                if (np.dot(vec1, perpendicular_vec) < 0 and point2line_distance1 < 2) and (np.dot(vec2, perpendicular_vec) < 0 and point2line_distance2 < 2) \
                    and line_angle >= 0.966:
                #if np.dot(vec1, perpendicular_vec) < 0 and np.dot(vec2, perpendicular_vec) < 0 and min(point2point_distance1, point2point_distance2) < 5:
                    left_lines.append(np.array([[x1, y1, z1], [x2, y2, z2]]))
                    left_closest_distance.append(min(point2point_distance1, point2point_distance2))

        return right_lines, right_closest_distance, left_lines, left_closest_distance
    '''
    def adjust_last_four_position(self, x, y):
        self.last_four_car_position_measurement = np.delete(self.last_four_car_position_measurement, 4, 0)
        self.last_four_car_position_measurement = np.insert(self.last_four_car_position_measurement, 0, [[x, y]], 0)
    '''
    def bivariate_linear_equation(self, p1, p2):
        A = np.array([[p1[0], 1],
                      [p2[0], 1]])
        if np.linalg.det(A) != 0:
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

        
        out1 = csv.writer(open("save_data/ndt_position.csv","w"), delimiter=',')
        for r in self.save_ndt_position:
            out1.writerow(r)
        out2 = csv.writer(open("save_data/xsens_position.csv","w"), delimiter=',')
        for r in self.save_xsens_position:
            out2.writerow(r)
        out3 = csv.writer(open("save_data/kf_position.csv","w"), delimiter=',')
        for r in self.save_kf_position:
            out3.writerow(r)
        out4 = csv.writer(open("save_data/projected_heading.csv","w"), delimiter=',')
        for r in self.save_projected_heading:
            out4.writerow(r)
        out5 = csv.writer(open("save_data/selected_right_lines.csv","w"), delimiter=',')
        for l in self.save_selected_right_lines:
            r = l.reshape((1, 6)).ravel()
            out5.writerow(r)
        out6 = csv.writer(open("save_data/selected_left_lines.csv","w"), delimiter=',')
        for l in self.save_selected_left_lines:
            r = l.reshape((1, 6)).ravel()
            out6.writerow(r)
        out7 = csv.writer(open("save_data/lateral_distance.csv","w"), delimiter=',')
        for r in self.save_lateral_distance:
            out7.writerow(r)
        out8 = csv.writer(open("save_data/rtk_position.csv","w"), delimiter=',')
        for r in self.save_rtk_position:
            out8.writerow(r)
        


if __name__ == '__main__':
    rospy.init_node('KF_lane_self_localization_node', anonymous=True)
    
    KF = KalmanFilter()
    try:
        KF.main()
    except rospy.ROSInterruptException:
        pass