#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
#from autoware_msgs.msg import _ImageLaneObjects
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension

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

import tensorflow as tf

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
class laneNet:
    def __init__(self):

        #---------------------------------- Load model
        self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
        self.phase_tensor = tf.constant('test', tf.string)
        net = lanenet_merge_model.LaneNet(phase = self.phase_tensor, net_flag='vgg')
        self.binary_seg_ret, self.instance_seg_ret = \
                                net.inference(input_tensor = self.input_tensor, name = 'lanenet_model')
        self.cluster = lanenet_cluster.LaneNetCluster()
        self.postprocessor = lanenet_postprocess.LaneNetPoseProcessor()
        saver = tf.train.Saver()

        #---------------------------------- Set session configuration
        '''
        if use_gpu:
            sess_config = tf.ConfigProto(device_count={'GPU': 1})
        else:
            sess_config = tf.ConfigProto(device_count={'CPU': 0})
        '''
        #sess_config = tf.ConfigProto(device_count={'CPU': 1})# ***************************
        sess_config = tf.ConfigProto(device_count={'GPU': 1})

        sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'
        self.sess = tf.Session(config = sess_config)
        saver.restore(sess = self.sess, save_path = '/home/mec-lab/Autoware/ros/src/computing/perception/detection/vision_detector/packages/vision_lane_detect/script/LaneNet/model/tusimple_lanenet/tusimple_lanenet_vgg_2018-10-19-13-33-56.ckpt-200000')

        #---------------------------------- Set topic
        self.image_pub = rospy.Publisher("laneNet_image", Image)
        self.left_lane_pub = rospy.Publisher("laneNet_left_lane", Int32MultiArray)
        self.left_mat = Int32MultiArray()
        self.left_mat.layout.dim.append(MultiArrayDimension())
        self.left_mat.layout.dim.append(MultiArrayDimension())
        
        self.bridge = CvBridge()

        image_topic_name = '/image_raw'
        rospy.loginfo('Setting image topic to %s', image_topic_name)
        self.image_sub = rospy.Subscriber(image_topic_name, Image, self.laneNet_callback)

    def laneNet_inference(self, frame, H):
        frame_vis = cv2.resize(frame, (512, 256))
        frame = cv2.resize(frame, (512, 256), interpolation = cv2.INTER_LINEAR) - VGG_MEAN
        
        #---------------------------------- Do inference
        t_start = time.time()
        binary_seg_frame, instance_seg_frame = self.sess.run([self.binary_seg_ret, self.instance_seg_ret], \
                                                        feed_dict={self.input_tensor: [frame]})
        t_cost = time.time() - t_start
        #rospy.loginfo('Inference time: {} second'.format(t_cost))
        rospy.loginfo('FPS: {} second'.format(1 / t_cost))

        binary_seg_frame[0] = self.postprocessor.postprocess(binary_seg_frame[0])
        mask_frame, left_lane, right_lane = self.cluster.get_lane_mask(binary_seg_ret = binary_seg_frame[0], \
                                        instance_seg_ret = instance_seg_frame[0])
        output_frame = cv2.addWeighted(frame_vis, 1, mask_frame, 1, 0)
        output_frame = cv2.resize(output_frame, (960, 604))

        if len(left_lane) > 0:
            self.left_mat.layout.dim[0].label = "height"
            self.left_mat.layout.dim[1].label = "width"
            self.left_mat.layout.dim[0].size = len(left_lane[0])
            self.left_mat.layout.dim[1].size = 2
            self.left_mat.layout.dim[0].stride = len(left_lane[0]) * 2
            self.left_mat.layout.dim[1].stride = len(left_lane[0])
            #self.left_mat.layout.data_offset = 0
            self.left_mat.data = []
            for i in left_lane[0]:
                self.left_mat.data.append(i[0])
                self.left_mat.data.append(i[1])

        output_frame = self.bridge.cv2_to_imgmsg(output_frame, 'bgr8')
        output_frame.header = H
        try:
            self.image_pub.publish(output_frame)
            self.left_lane_pub.publish(self.left_mat)
        except CvBridgeError as e:
            print(e)
        

    def laneNet_callback(self, data):
        try:
            H = data.header
            cv_image = CvBridge().imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.laneNet_inference(cv_image, H)
        cv2.waitKey(2)
    
    def main(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('laneNet_node', anonymous=True)
    #HP = HDPipeline()
    LN = laneNet()
    try:
        LN.main()
    except rospy.ROSInterruptException:
        pass
    
    
    
