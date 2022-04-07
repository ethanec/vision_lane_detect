#!/usr/bin/env python
########################################################
## Use Pseudo-Inverse Matrix to do Projective Mapping ##
########################################################
import rospy
import message_filters
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension

import numpy as np
import math
import random

#image_w = 1920
#image_h = 1208
image_w = 800
image_h = 503

class projective_mapping():

    def __init__(self):
        projection_world_points_topic_name = 'world_line_points_topic'
        projection_image_points_topic_name = 'image_line_points_topic'
        rospy.loginfo('Setting topic to %s', projection_world_points_topic_name)
        rospy.loginfo('Setting topic to %s', projection_image_points_topic_name)
        #---------------------------------- Time Synchronizer
        self.projection_world_points_sub = message_filters.Subscriber(projection_world_points_topic_name, Float32MultiArray)
        self.projection_image_points_sub = message_filters.Subscriber(projection_image_points_topic_name, Int32MultiArray)

        ts = message_filters.ApproximateTimeSynchronizer([self.projection_world_points_sub, self.projection_image_points_sub], \
                1, 1, allow_headerless=True) ### Here first 1 is the size of queue and next 1 is the time in sec to consider for aprox
        ts.registerCallback(self.calculate_projection_matrix_callback)
        self.projection_matrix_list = []

    def calculate_projection_matrix_callback(self, projection_world_points, projection_image_points):
        
        if (len(projection_world_points.data) / 3) == (len(projection_image_points.data) / 2) \
            and (len(projection_world_points.data) / 3) >= 4 \
            and (len(projection_image_points.data) / 2) >= 4:

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

            print(len(image_points))
            world_XY = []
            image_UV = []
            if len(image_points) >= 100:
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
                if np.linalg.det(np.dot(transpose_world_XY, world_XY)) != 0:
                    ####################
                    ## Pseudo-Inverse ##
                    ####################
                    #inv_world_XY = np.linalg.inv(world_XY)
                    #projection_matrix = np.dot(inv_world_XY, image_UV)
                    projection_matrix = np.dot(np.dot(np.linalg.inv(np.dot(transpose_world_XY, world_XY)), transpose_world_XY), image_UV)

                    print('--------------------')
                    #print(world_XY)
                    #print(image_UV)
                    print(projection_matrix)
                    self.projection_matrix_list.append(projection_matrix)
                    print('--------------------')

        else:
            print('Data too short or doesn\'t match!')
    
    def main(self):
        rospy.spin()
        self.projection_matrix_list = np.array(self.projection_matrix_list)
        projection_matrix_output = []
        for i in range(len(self.projection_matrix_list[0])):
            projection_matrix_output.append(np.mean(self.projection_matrix_list[:, i]))
        print('Projection Matrix: {}'.format(np.array(projection_matrix_output)))


if __name__ == '__main__':
    rospy.init_node('projective_mapping_node', anonymous = True)
    
    PM = projective_mapping()
    try:
        PM.main()
    except rospy.ROSInterruptException:
        pass
        