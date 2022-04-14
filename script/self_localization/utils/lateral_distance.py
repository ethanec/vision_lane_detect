#!/usr/bin/env python

import numpy as np
import math
import homography

class LateralDistance(object):

    def calculate_lateral_distance(self, car_position, projection_world_points, detection_image_points):
        #######################################
        # Derived from the camera calibration #
        #######################################
        
        # world frame using xsens receiver as origin
        projection_matrix = np.array([-1.52306968e+02, 2.87346122e+02, 1.46375024e+02, \
                                      -8.39373683e+01, -4.78781647e+00, -2.15631911e+02, \
                                      -3.56601520e-01, -9.89849613e-03])
        
        '''
        # wrold frame using NCKU EE as origin
        projection_matrix = np.array([-1.91320508e-01, 1.25239824e-01, 2.31733273e+03, \
                                      -5.90708714e-02, -2.11830447e-02, 2.25675126e+02, \
                                      -2.51188611e-04, -8.49871968e-05])
        '''
        num_line = detection_image_points.layout.dim[0].size
        num_point = detection_image_points.layout.dim[1].size
        num_coordinate = detection_image_points.layout.dim[2].size
        image_points = np.array(detection_image_points.data).reshape((num_line, num_point, num_coordinate))
        #print(image_points)
        
        #print('--------------------')
        ###########################
        # erfnet detection method #
        ###########################
        #image_left_1_s = image_points[1, 1]
        #image_left_1_e = image_points[1, 0]
        #image_right_1_s = image_points[2, 1]
        #image_right_1_e = image_points[2, 0]
        ###############################################
        # opencv detection method only output 2 lines #
        ###############################################
        image_left_1_s = image_points[0, 1]
        image_left_1_e = image_points[0, 0]
        image_right_1_s = image_points[1, 1]
        image_right_1_e = image_points[1, 0]
        world_left_1_s = self.UV2XY(projection_matrix, image_left_1_s)
        world_left_1_e = self.UV2XY(projection_matrix, image_left_1_e)
        world_right_1_s = self.UV2XY(projection_matrix, image_right_1_s)
        world_right_1_e = self.UV2XY(projection_matrix, image_right_1_e)
        #print(world_left_1_s)
        #print(world_left_1_e)
        #print(world_right_1_s)
        #print(world_right_1_e)

        left_vec = world_left_1_e - world_left_1_s
        right_vec = world_right_1_e - world_right_1_s
        #cosine_similarity = np.dot(left_vec[0:2], right_vec[0:2]) / (math.sqrt(left_vec[0] ** 2 + left_vec[1] ** 2) * math.sqrt(right_vec[0] ** 2 + right_vec[1] ** 2))
        #print('Cosine Similarity: {}'.format(cosine_similarity))

        #if abs(cosine_similarity) >= 0.9:
        
        LA = np.array([[world_left_1_s[0], 1],
                       [world_left_1_e[0], 1]])
        if np.linalg.det(LA) != 0:
            LB = np.array([world_left_1_s[1], world_left_1_e[1]]).reshape(2, 1)
            LA_inv = np.linalg.inv(LA)
            LC = LA_inv.dot(LB)
        else:
            LA = np.array([[world_left_1_s[0] + 1, 1],
                           [world_left_1_e[0], 1]])
            LB = np.array([world_left_1_s[1], world_left_1_e[1]]).reshape(2, 1)
            LA_inv = np.linalg.inv(LA)
            LC = LA_inv.dot(LB)
        #print(LC)
        
        RA = np.array([[world_right_1_s[0], 1],
                       [world_right_1_e[0], 1]])
        if np.linalg.det(RA) != 0:
            RB = np.array([world_right_1_s[1], world_right_1_e[1]]).reshape(2, 1)
            RA_inv = np.linalg.inv(RA)
            RC = RA_inv.dot(RB)
        else:
            RA = np.array([[world_right_1_s[0] + 1, 1],
                           [world_right_1_e[0], 1]])
            RB = np.array([world_right_1_s[1], world_right_1_e[1]]).reshape(2, 1)
            RA_inv = np.linalg.inv(RA)
            RC = RA_inv.dot(RB)

        #LD = self.Distance([car_position.pose.position.x, car_position.pose.position.y], LC)
        #RD = self.Distance([car_position.pose.position.x, car_position.pose.position.y], RC)
        
        Ow = self.UV2XY(projection_matrix, [400, 503])
        LD = self.Distance(Ow, LC)
        RD = self.Distance(Ow, RC)
        
        #print(Ow, [LD, RD])
        print('Left Distance: {}, Right Distance: {}'.format(LD, RD))
        return [LD, RD]

    def UV2XY(self, H, p): ### Same as doing (inverse H * point)
        a1 = H[0] - p[0] * H[6]
        a2 = H[3] - p[1] * H[6]
        b1 = H[1] - p[0] * H[7]
        b2 = H[4] - p[1] * H[7]
        d1 = H[2] - p[0] * 1
        d2 = H[5] - p[1] * 1
        x = (b1 * d2 - b2 * d1) / (a1 * b2 - b1 * a2)
        y = (a2 * d1 - a1 * d2) / (a1 * b2 - b1 * a2)
        return np.array([x, y])

    def Distance(self, point, line_coefficients):
        up = abs(point[0] * line_coefficients[0, 0] - 1 * point[1] + line_coefficients[1, 0])
        down = math.sqrt(line_coefficients[0, 0] ** 2 + 1)
        return up / down