#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
#from autoware_msgs.msg import _ImageLaneObjects
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from cv_bridge import CvBridge, CvBridgeError

import os
import os.path as ops
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import cv2
import utils.transforms as tf
from utils import cluster
from utils import tracker
import numpy as np
import models
import dataset as ds
from options.options import parser
import torch.nn.functional as F

#from keras.models import load_model
selected_lines = 2

class erfNet:
    def __init__(self):
        #---------------------------------- Load model
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        num_gpus = 1
        num_class = 5
        ignore_label = 255
        self.model = models.ERFNet(num_class, partial_bn = not False)
        self.input_mean = self.model.input_mean
        self.input_std = self.model.input_std
        policies = self.model.get_optim_policies()
        self.model = torch.nn.DataParallel(self.model, device_ids = range(num_gpus)).cuda()
        
        resume = '/home/mec-lab/Autoware/ros/src/computing/perception/detection/vision_detector/packages/vision_lane_detect/script/ERFNet-CULane-PyTorch/trained/ERFNet_trained.tar'
        print(("=> loading checkpoint '{}'".format(resume)))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        best_mIoU = checkpoint['best_mIoU']
        torch.nn.Module.load_state_dict(self.model, checkpoint['state_dict'])
        
        #---------------------------------- Set configuration
        cudnn.benchmark = True
        cudnn.fastest = True

        weights = [1.0 for _ in range(5)]
        weights[0] = 0.4
        learning_rate = 0.01
        momentum = 0.9
        weight_decay = 1e-4
        class_weights = torch.FloatTensor(weights).cuda()
        #self.criterion = torch.nn.NLLLoss(ignore_index=ignore_label, weight=class_weights).cuda()
        for group in policies:
            print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
        optimizer = torch.optim.SGD(policies, learning_rate, momentum = momentum, weight_decay = weight_decay)
        #self.evaluator = EvalSegmentation(num_class, ignore_label)

        #---------------------------------- other parameters
        self.past_five_right_lines = []
        self.erf_cluster = cluster.ERFNetCluster()
        #self.line_tracker = tracker.LaneTracker()
        #self.tracked_lines = np.zeros((2, 2)).astype(int)
        #self.lstm_model = load_model('/home/mec-lab/Autoware/ros/src/computing/perception/detection/vision_detector/packages/vision_lane_detect/script/ERFNet-CULane-PyTorch/trained/en_lstm_de_model_bs64.h5')
        #self.lstm_model._make_predict_function() # !?!?!?!?!?!?!?!?!?!?!?!?!?!?!?!?!?!?!?!?
        
        #---------------------------------- Set topic
        self.image_pub = rospy.Publisher("erfNet_image", Image, queue_size = 1)
        self.image_lines_pub = rospy.Publisher("erfNet_line_points_topic", Int32MultiArray, queue_size = 1)
        self.lines_mat = Int32MultiArray()
        self.lines_mat.layout.dim.append(MultiArrayDimension())
        self.lines_mat.layout.dim.append(MultiArrayDimension())
        self.lines_mat.layout.dim.append(MultiArrayDimension())
        
        self.bridge = CvBridge()

        #image_topic_name = '/image_raw'
        image_topic_name = '/gmsl_camera/port_0/cam_0/image_raw'
        rospy.loginfo('Setting image topic to %s', image_topic_name)
        self.image_sub = rospy.Subscriber(image_topic_name, Image, self.erfNet_callback)

        ############
        ## Kalman ##
        ############
        self.processed = 0
        self.x = np.zeros((8, 1))
        self._x = np.zeros((8, 1))
        self.__x = np.zeros((8, 1))
        self.x_ = np.zeros((8, 1))
        self.x__ = np.zeros((8, 1))
        self.p_ = np.zeros((8, 8))
        self.new_x = np.zeros((8, 1))
        self.new_p = np.zeros((8, 8))

        self.F = np.eye(8) * 2
        #self.F = np.eye(8)
        self.w = np.array([[9.2156998498e-06], [1.33320556859e-05], [7.79305519721e-06], [7.43892609626e-06], \
                           [6.48961210021e-06], [3.25286542643e-06], [5.3053705588e-06], [7.7387367644e-07]])
        self.P = np.eye(8)
        self.H = np.eye(8)
        #self.Q = np.eye(8) * 1.5
        self.Q = np.diag([490.935248044, 2404.96210096, 311.542448381, 691.064533102, \
                          784.576892201, 1101.46631196, 1910.0385795, 3636.16743755])
        self.R = np.eye(8) * 2.5
        self.I = np.eye(8)


        self.last_l1 = np.zeros(2)
        self.last_l2 = np.zeros(2)

        self.error = []

    
    def erfNet_inference(self, frame, H):
        self.model.eval()
        #---------------------------------- Read frame
        frame_vis = frame
        h = frame_vis.shape[0]
        w = frame_vis.shape[1]
        frame = frame[240:, :, :]
        cliped_h = frame.shape[0]
        cliped_w = frame.shape[1]
        frame = cv2.resize(frame, (976, 208), interpolation = cv2.INTER_LINEAR)

        t_start = time.time()

        #---------------------------------- GroupNormalize
        img_group = [frame]
        mean = (self.input_mean, (0, ))
        std = (self.input_std, (1, ))
        out_images = list()
        for img, m, s in zip(img_group, mean, std):
            if len(m) == 1:
                img = img - np.array(m)  # single channel image
                img = img / np.array(s)
            else:
                img = img - np.array(m)[np.newaxis, np.newaxis, ...]
                img = img / np.array(s)[np.newaxis, np.newaxis, ...]
            out_images.append(img)

        frame = out_images[0]
        frame = torch.from_numpy(frame).permute(2, 0, 1).contiguous().float()
        frame = frame.reshape(1, 3, 208, 976)
        
        
        input_var = torch.autograd.Variable(frame, volatile = True)

        #---------------------------------- Compute output
        with torch.no_grad():
            output, output_exist = self.model(input_var)

        #---------------------------------- Measure accuracy and record loss
        output = F.softmax(output, dim=1)
        pred = output.data.cpu().numpy() # BxCxHxW (1, 5, 208, 976)
        pred_exist = output_exist.data.cpu().numpy() # BxO
        
        #---------------------------------- Image post-processing
        pred_input = cv2.resize(pred[0].transpose((1, 2, 0)), (448, 208), interpolation = cv2.INTER_LINEAR) # (208, 976, 3) -> (208, 448, 3)
        '''
        merge = ((pred_input[:, :, 1] * 255).astype(np.uint8) + \
                (pred_input[:, :, 2] * 255).astype(np.uint8) + \
                (pred_input[:, :, 3] * 255).astype(np.uint8) + \
                (pred_input[:, :, 4] * 255).astype(np.uint8)) / 4
        '''
        #mask_list = erf_cluster.get_lane_mask(merge, pred_list)
        #mask_list, points_on_image = self.erf_cluster.get_lane_mask(pred_input[:, :, 1:])
        mask_list, points_on_image = self.erf_cluster.get_lane_mask(pred_input[:, :, 2:4], selected_lines)
        #print(points_on_image)
        points_on_image = self.Kalman(points_on_image)
        #print(points_on_image)
        #---------------------------------- Go through tracker
        #self.past_five_right_lines.append(points_on_image[2])
        #if len(self.past_five_right_lines) == 6:
        #    points_on_image[2] = self.line_tracker.right_line_tracker(self.lstm_model, self.past_five_right_lines[1:])
        #    self.past_five_right_lines[-1] = points_on_image[2]
        #    del self.past_five_right_lines[0]
        
        #---------------------------------- Compose output
        '''
        for m in range(selected_lines):
            tmp = mask_list[m]
            mask_list[m] = []
            mask_list[m] = cv2.resize(tmp, (cliped_w, cliped_h), interpolation = cv2.INTER_LINEAR) # (208, 448) -> (cliped_h, cliped_w)

        mask_output = np.zeros([cliped_h, cliped_w, 3]).astype(np.uint8) # (364, 960, 3)
        mask_output[:, :, 0] = np.zeros([cliped_h, cliped_w]).astype(np.uint8) # B
        #mask_output[:, :, 1] = mask_list[0].astype(np.uint8) + mask_list[1].astype(np.uint8) # G
        mask_output[:, :, 1] = mask_list[0].astype(np.uint8) # G
        #mask_output[:, :, 2] = mask_list[2].astype(np.uint8) + mask_list[3].astype(np.uint8) # R
        mask_output[:, :, 2] = mask_list[1].astype(np.uint8) # R
        
        mask_frame = np.zeros([h, w, 3]).astype(np.uint8) # (604, 960, 3)
        for i in range(0, 3):
            compensate_zero = np.zeros([240, cliped_w]).astype(np.uint8)
            mask_frame[:, :, i] = np.concatenate((compensate_zero, mask_output[:, :, i]), axis = 0)
        '''
        ########output_frame = (cv2.addWeighted(frame_vis, 1, mask_frame, 1, 0)) # (604, 960, 3)

        out_line1 = np.zeros((2, 2), np.int32)
        out_line2 = np.zeros((2, 2), np.int32)
        
        for n, p in enumerate(points_on_image[0]):
            x = int(p[0] * w / 448)
            y = int((p[1] + 240) * h / 448)
            out_line1[n, 0] = x
            out_line1[n, 1] = y
        for n, p in enumerate(points_on_image[1]):
            x = int(p[0] * w / 448)
            y = int((p[1] + 240) * h / 448)
            out_line2[n, 0] = x
            out_line2[n, 1] = y
        

        cv2.polylines(img = frame_vis, pts = [out_line1], isClosed = False, color = (0, 255, 0), thickness = 5)
        cv2.polylines(img = frame_vis, pts = [out_line2], isClosed = False, color = (0, 0, 255), thickness = 5)
        #---------------------------------- Plot tracked line
        #self.tracked_lines[:, 0], self.tracked_lines[:, 1] = (points_on_image[2, :, 0] * w / 448).astype(int), ((points_on_image[2, :, 1] + 240) * h / 448).astype(int)
        #cv2.polylines(img = output_frame, pts = [self.tracked_lines], isClosed = False, color = (255, 0, 0), thickness = 5)

        t_cost = time.time() - t_start
        print(('FPS: {:.5f}'.format(1 / t_cost)))
        
        #---------------------------------- Save detected lines' points
        if len(points_on_image) > 0:
            self.lines_mat.layout.dim[0].label = "no."
            self.lines_mat.layout.dim[1].label = "start_end"
            self.lines_mat.layout.dim[2].label = "U_V"
            self.lines_mat.layout.dim[0].size = len(points_on_image)
            self.lines_mat.layout.dim[1].size = 2 # Start point and end point
            self.lines_mat.layout.dim[2].size = 2 # (U, V)
            self.lines_mat.layout.dim[0].stride = len(points_on_image) * 2 * 2
            self.lines_mat.layout.dim[1].stride = 2 * 2
            self.lines_mat.layout.dim[2].stride = 2
            #self.lines_mat.layout.data_offset = 0

            self.lines_mat.data = []
            for l in points_on_image:
                for p in l:
                    x = int(p[0] * w / 448)
                    y = int((p[1] + 240) * h / 448)
                    if x <= w and x > 0:
                        self.lines_mat.data.append(x)
                    else:
                        self.lines_mat.data.append(0)
                    if y <= h and y > 0:
                        self.lines_mat.data.append(y)
                    else:
                        self.lines_mat.data.append(0)
        
        self.processed = self.processed + 1
        ########output_frame = self.bridge.cv2_to_imgmsg(output_frame, 'bgr8')
        ########output_frame.header = H
        output_frame = self.bridge.cv2_to_imgmsg(frame_vis, 'bgr8')
        output_frame.header = H
        try:
            self.image_pub.publish(output_frame)
            self.image_lines_pub.publish(self.lines_mat)
        except CvBridgeError as e:
            print(e)

    def Kalman(self, data):
        H = 208
        W = 448
        l1 = self.bivariate_linear_equation(data[0, 0], data[0, 1])
        l2 = self.bivariate_linear_equation(data[1, 0], data[1, 1])

        if l1[0] == 0 or l1[0] == 0:
            np.copyto(l1, self.last_l1)
        if l2[0] == 0 or l2[0] == 0:
            np.copyto(l2, self.last_l2)
        
        s = 0
        for h in [0, 69, 138, 207]:
            self.x[s * 2, 0] = (h - l1[1]) / l1[0]
            self.x[s * 2 + 1, 0] = (h - l2[1]) / l2[0]
            s = s + 1
        
        if self.processed > 1:
            
            self.x_ = np.dot(self.F, (self._x - 0.5 * self.__x)) + self.w
            self.p_ = np.dot(np.dot(self.F, self.P), self.F.transpose()) + self.Q

            
            self.error.append(np.array([ self.x[0, 0] - self.x_[0, 0], \
                                         self.x[1, 0] - self.x_[1, 0], \
                                         self.x[2, 0] - self.x_[2, 0], \
                                         self.x[3, 0] - self.x_[3, 0], \
                                         self.x[4, 0] - self.x_[4, 0], \
                                         self.x[5, 0] - self.x_[5, 0], \
                                         self.x[6, 0] - self.x_[6, 0], \
                                         self.x[7, 0] - self.x_[7, 0] ]))
            
            #print(self.x)
            #print(self.x_)
            '''
            for i in range(0, 4):
                if abs(self.x[i * 2, 0] - self.x__[i * 2, 0]) > 10: #or \
                   #abs(self.x[i * 2, 0] - self._x[i * 2, 0]) > 20 or \
                   #abs(self.x[i * 2, 0] - self.__x[i * 2, 0]) > 20:
                    #print('fuck')
                    #print(self.x)
                    #print(self.x_)
                    self.x[i * 2, 0] = self.x_[i * 2, 0]

                if abs(self.x[i * 2 + 1, 0] - self.x__[i * 2 + 1, 0]) > 10: #or \
                   #abs(self.x[i * 2 + 1, 0] - self._x[i * 2 + 1, 0]) > 20 or \
                   #abs(self.x[i * 2 + 1, 0] - self.__x[i * 2 + 1, 0]) > 20:

                    #print('shit')
                    #print(self.x)
                    #print(self.x_)
                    self.x[i * 2 + 1, 0] = self.x_[i * 2 + 1, 0]
            '''
            Kg = np.dot(np.dot(self.p_ , self.H.transpose()), np.linalg.inv(np.dot(np.dot(self.H, self.p_), self.H.transpose()) + self.R))
            

            self.new_x = self.x_ + np.dot(Kg, (self.x - np.dot(self.H, self.x_)))
            self.new_p = np.dot((self.I - np.dot(Kg, self.H)), self.p_)

            #print(self.new_x)
            
            np.copyto(self.P, self.new_p)
            np.copyto(self.x__, self.x_)
            np.copyto(self.__x, self._x)
            np.copyto(self._x, self.x)
            np.copyto(self.last_l1, l1)
            np.copyto(self.last_l2, l2)

            return np.array([[[self.new_x[0, 0], 0], [self.new_x[6, 0], 207]], [[self.new_x[1, 0], 0], [self.new_x[7, 0], 207]]])
        else:
            np.copyto(self.__x, self._x)
            np.copyto(self._x, self.x)
            np.copyto(self.last_l1, l1)
            np.copyto(self.last_l2, l2)

            return data

    def bivariate_linear_equation(self, p1, p2):
        A = np.array([[p1[0], 1],
                      [p2[0], 1]])
        if np.linalg.det(A) != 0:
            B = np.array([p1[1], p2[1]]).reshape(2, 1)
            A_inv = np.linalg.inv(A)
            C = A_inv.dot(B)
            return C.ravel()
        else:
            A = np.array([[p1[0] + 1, 1],
                          [p2[0], 1]])
            B = np.array([p1[1], p2[1]]).reshape(2, 1)
            A_inv = np.linalg.inv(A)
            C = A_inv.dot(B)
            return C.ravel()

    def getLineEqu(self, p1, p2):
        # AX=B
        A = np.array([[p1[0], 1],[p2[0],1]])
        B = np.array([p1[1], p2[1]])
        
        # X = [a, b]^t
        # y = ax + b
        a, b = np.linalg.solve(A, B)

        #print('y = ax + b. a = ', a,  ' b = ', b)
        return np.array([a, b])

    def erfNet_callback(self, data):
        try:
            H = data.header
            cv_image = CvBridge().imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.erfNet_inference(cv_image, H)
        #cv2.waitKey(1)

    def main(self):
        rospy.spin()
        
        self.error = np.array(self.error)

        print('1 Variance: {}'.format(np.std(self.error[:, 0]) ** 2))
        print('2 Variance: {}'.format(np.std(self.error[:, 1]) ** 2))
        print('3 Variance: {}'.format(np.std(self.error[:, 2]) ** 2))
        print('4 Variance: {}'.format(np.std(self.error[:, 3]) ** 2))
        print('5 Variance: {}'.format(np.std(self.error[:, 4]) ** 2))
        print('6 Variance: {}'.format(np.std(self.error[:, 5]) ** 2))
        print('7 Variance: {}'.format(np.std(self.error[:, 6]) ** 2))
        print('8 Variance: {}'.format(np.std(self.error[:, 7]) ** 2))
        

class EvalSegmentation(object):
    def __init__(self, num_class, ignore_label=None):
        self.num_class = num_class
        self.ignore_label = ignore_label

    def __call__(self, pred, gt):
        assert (pred.shape == gt.shape)
        gt = gt.flatten().astype(int)
        pred = pred.flatten().astype(int)
        locs = (gt != self.ignore_label)
        sumim = gt + pred * self.num_class
        hs = np.bincount(sumim[locs], minlength=self.num_class**2).reshape(self.num_class, self.num_class)
        return hs

if __name__ == '__main__':
    rospy.init_node('erfNet_node', anonymous=True)
    
    EN = erfNet()
    try:
        EN.main()
    except rospy.ROSInterruptException:
        pass