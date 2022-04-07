#!/usr/bin/env python

import numpy as np
import math
import random

#image_w = 1920
#image_h = 1208
image_w = 800
image_h = 503

def calculate_projection_matrix(projection_world_points, projection_image_points):
    #print('hi')
    num_line = projection_world_points.layout.dim[0].size
    num_point = projection_world_points.layout.dim[1].size
    num_coordinate = projection_world_points.layout.dim[2].size
    world_points = np.array(projection_world_points.data).reshape((num_line * num_point, num_coordinate))

    num_line = projection_image_points.layout.dim[0].size
    num_point = projection_image_points.layout.dim[1].size
    num_coordinate = projection_image_points.layout.dim[2].size
    image_points = np.array(projection_image_points.data).reshape((num_line * num_point, num_coordinate))

    num_sample = 0
    sample_world_points = []
    sample_image_points = []
    
    toDelete = []
    for p, point in enumerate(image_points):
        if point[0] <= 0 or point[1] <= 0 or point[0] > image_w or point[1] > image_h:
            toDelete.append(p)
    image_points = np.delete(image_points, toDelete, 0)
    world_points = np.delete(world_points, toDelete, 0)

    #print(len(image_points))
    world_XY = []
    image_UV = []
    if len(image_points) >= 4:
    #if len(image_points) >= 10:
        for idx, i in enumerate(world_points):
            X, Y = world_points[idx, 0], world_points[idx, 1]
            U, V = image_points[idx, 0], image_points[idx, 1]
            world_XY.append([X, Y, 1, 0, 0, 0, -1 * U * X, -1 * U * Y])
            world_XY.append([0, 0, 0, X, Y, 1, -1 * V * X, -1 * V * Y])
            image_UV.append(U)
            image_UV.append(V)
        world_XY = np.array(world_XY)
        image_UV = np.transpose(np.array(image_UV))
        
        transpose_world_XY = np.transpose(world_XY)
        if np.linalg.det(np.dot(transpose_world_XY, world_XY)) == 0:
            return None
        else:
            #inv_world_XY = np.linalg.inv(world_XY)
            #projection_matrix = np.dot(inv_world_XY, image_UV)
            projection_matrix = np.dot(np.dot(np.linalg.inv(np.dot(transpose_world_XY, world_XY)), transpose_world_XY), image_UV)

            #print('--------------------')
            #print(world_XY)
            #print(image_UV)
            #print(projection_matrix)
            #print('--------------------')

            return projection_matrix
        '''
        center_x = (max(image_points[:, 0]) + min(image_points[:, 0])) / 2
        center_y = (max(image_points[:, 1]) + min(image_points[:, 1])) / 2

        image_top_left = image_points[((image_points[:, 0] < center_x) & (image_points[:, 1] < center_y))]
        image_bottom_left = image_points[((image_points[:, 0] < center_x) & (image_points[:, 1] > center_y))]
        image_top_right = image_points[((image_points[:, 0] > center_x) & (image_points[:, 1] < center_y))]
        image_bottom_right = image_points[((image_points[:, 0] > center_x) & (image_points[:, 1] > center_y))]

        world_top_left = world_points[((image_points[:, 0] < center_x) & (image_points[:, 1] < center_y))]
        world_bottom_left = world_points[((image_points[:, 0] < center_x) & (image_points[:, 1] > center_y))]
        world_top_right = world_points[((image_points[:, 0] > center_x) & (image_points[:, 1] < center_y))]
        world_bottom_right = world_points[((image_points[:, 0] > center_x) & (image_points[:, 1] > center_y))]
        
        if len(image_top_left) > 0 and len(image_bottom_left) > 0 and len(image_top_right) > 0 and len(image_bottom_right) > 0:
            image_top_left_point = image_top_left[np.argmax(np.sqrt((image_top_left[:, 0] - center_x) ** 2 + (image_top_left[:, 1] - center_y) ** 2))]
            image_bottom_left_point = image_bottom_left[np.argmax(np.sqrt((image_bottom_left[:, 0] - center_x) ** 2 + (image_bottom_left[:, 1] - center_y) ** 2))]
            image_top_right_point = image_top_right[np.argmax(np.sqrt((image_top_right[:, 0] - center_x) ** 2 + (image_top_right[:, 1] - center_y) ** 2))]
            image_bottom_right_point = image_bottom_right[np.argmax(np.sqrt((image_bottom_right[:, 0] - center_x) ** 2 + (image_bottom_right[:, 1] - center_y) ** 2))]

            world_top_left_point = world_top_left[np.argmax(np.sqrt((image_top_left[:, 0] - center_x) ** 2 + (image_top_left[:, 1] - center_y) ** 2))]
            world_bottom_left_point = world_bottom_left[np.argmax(np.sqrt((image_bottom_left[:, 0] - center_x) ** 2 + (image_bottom_left[:, 1] - center_y) ** 2))]
            world_top_right_point = world_top_right[np.argmax(np.sqrt((image_top_right[:, 0] - center_x) ** 2 + (image_top_right[:, 1] - center_y) ** 2))]
            world_bottom_right_point = world_bottom_right[np.argmax(np.sqrt((image_bottom_right[:, 0] - center_x) ** 2 + (image_bottom_right[:, 1] - center_y) ** 2))]
        
            #print(image_top_left)
            #print(world_top_left)
            #print(top_left_point)
        
            X1, Y1, Z1 = world_top_left_point[0], world_top_left_point[1], world_top_left_point[2]
            X2, Y2, Z2 = world_bottom_left_point[0], world_bottom_left_point[1], world_bottom_left_point[2]
            X3, Y3, Z3 = world_top_right_point[0], world_top_right_point[1], world_top_right_point[2]
            X4, Y4, Z4 = world_bottom_right_point[0], world_bottom_right_point[1], world_bottom_right_point[2]
            U1, V1 = image_top_left_point[0], image_top_left_point[1]
            U2, V2 = image_bottom_left_point[0], image_bottom_left_point[1]
            U3, V3 = image_top_right_point[0], image_top_right_point[1]
            U4, V4 = image_bottom_right_point[0], image_bottom_right_point[1]

            world_XY = np.array([[X1, Y1, 1, 0, 0, 0, -1*U1*X1, -1*U1*Y1],
                                [0, 0, 0, X1, Y1, 1, -1*V1*X1, -1*V1*Y1],
                                [X2, Y2, 1, 0, 0, 0, -1*U2*X2, -1*U2*Y2],
                                [0, 0, 0, X2, Y2, 1, -1*V2*X2, -1*V2*Y2],
                                [X3, Y3, 1, 0, 0, 0, -1*U3*X3, -1*U3*Y3],
                                [0, 0, 0, X3, Y3, 1, -1*V3*X3, -1*V3*Y3],
                                [X4, Y4, 1, 0, 0, 0, -1*U4*X4, -1*U4*Y4],
                                [0, 0, 0, X4, Y4, 1, -1*V4*X4, -1*V4*Y4]])
            if np.linalg.det(world_XY) == 0:
                world_XY[0, 0] = world_XY[0, 0] + 1

            #print(np.linalg.det(world_XY))
            if np.linalg.det(world_XY) == 0:
                return None
            else:
                inv_world_XY = np.linalg.inv(world_XY)
                image_UV = np.array([U1, V1, U2, V2, U3, V3, U4, V4]).reshape(8, 1)
                projection_matrix = np.dot(inv_world_XY, image_UV)

                #print('--------------------')
                #print(projection_matrix)
                #print('--------------------')

                return projection_matrix
        else:
            return None
        '''
    else:
        return None
        