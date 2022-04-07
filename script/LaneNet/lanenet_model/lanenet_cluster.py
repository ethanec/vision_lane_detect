#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-15 下午4:29
# @Author  : Luo Yao
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_cluster.py
# @IDE: PyCharm Community Edition
"""
实现LaneNet中实例分割的聚类部分
"""
import numpy as np
import glog as log
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
import time
import warnings
import cv2
import math
import PIL.Image as pil
import numpy.linalg as LA
try:
    from cv2 import cv2
except ImportError:
    pass


class LaneNetCluster(object):
    """
    实例分割聚类器
    """

    def __init__(self):
        """

        """
        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]
        pass

    @staticmethod
    def _cluster(prediction, bandwidth):
        """
        实现论文SectionⅡ的cluster部分
        :param prediction:
        :param bandwidth:
        :return:
        """
        ms = MeanShift(bandwidth, bin_seeding=True)
        # log.info('开始Mean shift聚类 ...')
        tic = time.time()
        try:
            ms.fit(prediction)
        except ValueError as err:
            log.error(err)
            return 0, [], []
        # log.info('Mean Shift耗时: {:.5f}s'.format(time.time() - tic))
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        num_clusters = cluster_centers.shape[0]

        # log.info('聚类簇个数为: {:d}'.format(num_clusters))

        return num_clusters, labels, cluster_centers

    @staticmethod
    def _cluster_v2(prediction):
        """
        dbscan cluster
        :param prediction:
        :return:
        """
        db = DBSCAN(eps=0.7, min_samples=200).fit(prediction)
        db_labels = db.labels_
        unique_labels = np.unique(db_labels)
        unique_labels = [tmp for tmp in unique_labels if tmp != -1]
        log.info('聚类簇个数为: {:d}'.format(len(unique_labels)))

        num_clusters = len(unique_labels)
        cluster_centers = db.components_

        return num_clusters, db_labels, cluster_centers

    @staticmethod
    def _get_lane_area(binary_seg_ret, instance_seg_ret):
        """
        通过二值分割掩码图在实例分割图上获取所有车道线的特征向量
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        idx = np.where(binary_seg_ret == 1)

        lane_embedding_feats = []
        lane_coordinate = []
        for i in range(len(idx[0])):
            lane_embedding_feats.append(instance_seg_ret[idx[0][i], idx[1][i]])
            lane_coordinate.append([idx[0][i], idx[1][i]])

        return np.array(lane_embedding_feats, np.float32), np.array(lane_coordinate, np.int64)

    @staticmethod
    def _thresh_coord(coord):
        """
        过滤实例车道线位置坐标点,假设车道线是连续的, 因此车道线点的坐标变换应该是平滑变化的不应该出现跳变
        :param coord: [(x, y)]
        :return:
        """
        pts_x = coord[:, 0]
        mean_x = np.mean(pts_x)

        idx = np.where(np.abs(pts_x - mean_x) < mean_x)

        return coord[idx[0]]

    @staticmethod
    def _lane_fit(lane_pts):
        """
        车道线多项式拟合
        :param lane_pts:
        :return:
        """
        if not isinstance(lane_pts, np.ndarray):
            lane_pts = np.array(lane_pts, np.float32)

        x = lane_pts[:, 0]
        y = lane_pts[:, 1]
        x_fit = []
        y_fit = []
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                f1 = np.polyfit(x, y, 3)
                p1 = np.poly1d(f1)
                x_min = int(np.min(x))
                x_max = int(np.max(x))
                x_fit = []
                for i in range(x_min, x_max + 1):
                    x_fit.append(i)
                y_fit = p1(x_fit)
            except Warning as e:
                x_fit = x
                y_fit = y
            finally:
                return zip(x_fit, y_fit)

    @staticmethod          
    def _transparent_back(img):
    #img = pil.open(mess)
        img = pil.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = img.convert('RGBA')
        L, H = img.size
        color_0 = img.getpixel((0,0))
        for h in range(H):
            for l in range(L):
                dot = (l,h)
                color_1 = img.getpixel(dot)
                if color_1 == color_0:
                    color_1 = color_1[:-1] + (0,)
                    img.putpixel(dot,color_1)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        #img = img.astype(float)
        return img

    @staticmethod
    def _cal_curverature(x, y):
        """
        input  : the coordinate of the three point
        output : the curvature and norm direction
        refer to https://github.com/Pjer-zhang/PJCurvature for detail
        """
        t_a = LA.norm([x[1]-x[0],y[1]-y[0]])
        t_b = LA.norm([x[2]-x[1],y[2]-y[1]])
        
        M = np.array([
            [1, -t_a, t_a**2],
            [1, 0,    0     ],
            [1,  t_b, t_b**2]
        ])

        a = np.matmul(LA.inv(M),x)
        b = np.matmul(LA.inv(M),y)

        kappa = 2*(a[2]*b[1]-b[2]*a[1])/(a[1]**2.+b[1]**2.)**(1.5)
        return kappa, [b[1],-a[1]]/np.sqrt(a[1]**2.+b[1]**2.)

    def get_lane_mask(self, binary_seg_ret, instance_seg_ret):
        """

        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        img_x_center = 256
        img_y_center = 128
        left_lane = []
        right_lane = []
        center_lane = []
        lane_embedding_feats, lane_coordinate = self._get_lane_area(binary_seg_ret, instance_seg_ret)

        num_clusters, labels, cluster_centers = self._cluster(lane_embedding_feats, bandwidth=1.5)

        # 聚类簇超过八个则选择其中类内样本最多的八个聚类簇保留下来
        if num_clusters > 8:
            cluster_sample_nums = []
            for i in range(num_clusters):
                cluster_sample_nums.append(len(np.where(labels == i)[0]))
            sort_idx = np.argsort(-np.array(cluster_sample_nums, np.int64))
            cluster_index = np.array(range(num_clusters))[sort_idx[0:8]]
        else:
            cluster_index = range(num_clusters)

        #mask_image = np.zeros(shape = [binary_seg_ret.shape[0], binary_seg_ret.shape[1], 3], dtype=np.uint8)
        mask_img = cv2.cvtColor(np.asarray(pil.new(mode = 'RGBA', size = (binary_seg_ret.shape[1], binary_seg_ret.shape[0]))), cv2.COLOR_RGBA2RGB)
        mask_image = cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR)
        mask_image = mask_image.astype(np.uint8)

        for index, i in enumerate(cluster_index):# Plot one lane for each iteration
            idx = np.where(labels == i)
            coord = lane_coordinate[idx]
            #print(coord.shape)
            coord = self._thresh_coord(coord)
            coord = np.flip(coord, axis=1)
            #print(np.shape(coord))
            #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Remove those data with y coord which is above the 1/3 of the frame
            '''
            toDelete = []
            for yndex, y in enumerate(coord[:, 1]):
                if y < ((img_y_center * 2) / 3) * 2:
                    toDelete.append(yndex)
            coord = np.delete(coord, toDelete, 0)
            if len(coord) == 0:
                continue
            '''
            #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            #cv2.circle(img = mask_image, center = tuple(coord[-1, :]), radius = 10, color = (0, 100, 200), thickness = 1)# Plot the botton point of the lane line
            #cv2.circle(img = mask_image, center = (img_x_center, 256), radius = 10, color = (200, 0, 220), thickness = 1)# Plot the cetral button point
            #cv2.circle(img = mask_image, center = (img_x_center, img_y_center), radius = 10, color = (200, 0, 220), thickness = 1)# Plot the cetral point
            #cv2.putText(img = mask_image, text = str(index + 1), org = tuple(coord[-1, :]), fontFace = 0, fontScale = 1, color = (0, 100, 200))
            #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Seperate which lane is at the left and which is at the right
            if(coord[-1, 0] < img_x_center):
                if(len(left_lane) == 0):
                    left_lane = [coord]
                else:
                    left_lane.append(coord)
            elif(coord[-1, 0] >= img_x_center):
                if(len(right_lane) == 0):
                    right_lane = [coord]
                else:
                    right_lane.append(coord)
            #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # coord = (coord[:, 0], coord[:, 1])
            '''
            color = (int(self._color_map[index][0]),
                     int(self._color_map[index][1]),
                     int(self._color_map[index][2]))'''
            #coord = np.array([coord])
            #cv2.polylines(img = mask_image, pts = coord, isClosed = False, color = color, thickness = 1)
        '''
        left_lane = np.array([left_lane])
        right_lane = np.array([right_lane])
        '''
        #L_cloest = left_lane[0]
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ To make sure the lanes which is cloest to center is in the 1st index of lane list
        '''
        left_lane_distance = []
        right_lane_distance = []
        if (len(left_lane) > 1):
            for l in left_lane:
                #print(np.polyfit(l[:, 0], l[:, 1], 1))
                F = np.polyfit(l[:, 0], l[:, 1], 1)
                D = abs(F[0] * img_x_center - 1 * img_y_center * 2 + F[1]) / math.sqrt(math.pow(F[0], 2) + 1)
                left_lane_distance.append(D)
            left_lane_distance = np.array(left_lane_distance)
            left_lane = np.array(left_lane)[np.argsort(left_lane_distance)]
        
        if (len(right_lane) > 1):
            for l in right_lane:
                #print(np.polyfit(l[:, 0], l[:, 1], 1))
                F = np.polyfit(l[:, 0], l[:, 1], 1)
                D = abs(F[0] * img_x_center - 1 * img_y_center * 2 + F[1]) / math.sqrt(math.pow(F[0], 2) + 1)
                right_lane_distance.append(D)
            right_lane_distance = np.array(right_lane_distance)
            right_lane = np.array(right_lane)[np.argsort(right_lane_distance)]
        '''
        
        
        tmp = []
        if (len(left_lane) > 1):
            for index in range(len(left_lane)):
                if (left_lane[index][-1, 0] > left_lane[0][-1, 0]) and (left_lane[index][-1, 1] > left_lane[0][-1, 1]):
                    tmp = left_lane[0]
                    left_lane[0] = left_lane[index]
                    left_lane[index] = tmp
        tmp = []
        if (len(right_lane) > 1):
            for index in range(len(right_lane)):
                if (right_lane[index][-1, 0] < right_lane[0][-1, 0]) and (right_lane[index][-1, 1] > right_lane[0][-1, 1]):
                    tmp = right_lane[0]
                    right_lane[0] = right_lane[index]
                    right_lane[index] = tmp
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        '''
        #:::::::::::::::::::::::::::::::::::::::::::::: polynomial to fit scatter points and extend lane lines
        if (len(left_lane) > 0):
            for ldx, l in enumerate(left_lane):
                P_X = []
                P_Y = []
                x = [l[0, 0], l[len(l) / 2, 0], l[-1, 0]]
                y = [l[0, 1], l[len(l) / 2, 1], l[-1, 1]]
                curv, norm = self._cal_curverature(x, y)
                o = 2
                if (abs(curv) > 0.001 and len(l) < 100):
                    o = 1
                #print(curv)
                if len(l) < 150:
                    F = np.polyfit(l[:, 0], l[:, 1], o)
                    P = np.poly1d(F)

                    if max(l[:, 0]) < int((512 / 3) * 1):
                        #P_X = np.arange(int((512 / 4) * 1), int((512 / 3) * 1))
                        P_X = np.linspace(int((512 / 3) * 1), 0, 10).astype(int)
                        P_Y = P(P_X).astype(int)
                    
                    else:
                        #P_X = np.arange(int((512 / 4) * 1), max(l[:, 0]))
                        P_X = np.linspace(max(l[:, 0]), 0, 10).astype(int)
                        P_Y = P(P_X).astype(int)
                    
                    #print(np.stack((P_X, P_Y), axis = 1))
                    left_lane[ldx] = np.stack((P_X, P_Y), axis = 1)
                #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Remove those data with y coord which is above the 1/3 of the frame
                toDelete = []
                for yndex, y in enumerate(l[:, 1]):
                    if y < ((img_y_center * 2) / 3) * 2:
                        toDelete.append(yndex)
                l = np.delete(l, toDelete, 0)
        
        if (len(right_lane) > 0):
            for ldx, l in enumerate(right_lane):
                P_X = []
                P_Y = []
                x = [l[0, 0], l[len(l) / 2, 0], l[-1, 0]]
                y = [l[0, 1], l[len(l) / 2, 1], l[-1, 1]]
                curv, norm = self._cal_curverature(x, y)
                o = 2
                if (abs(curv) > 0.001 and len(l) < 100):
                    o = 1
                if len(l) < 150:
                    F = np.polyfit(l[:, 0], l[:, 1], o)
                    P = np.poly1d(F)
                    if min(l[:, 0]) > int((512 / 3) * 2):
                        P_X = np.arange(int((512 / 3) * 2), 512)
                        P_Y = P(P_X).astype(int)
                        
                    else:
                        P_X = np.arange(min(l[:, 0]), 512)
                        P_Y = P(P_X).astype(int)
                        
                    #print(np.stack((P_X, P_Y), axis = 1))
                    right_lane[ldx] = np.stack((P_X, P_Y), axis = 1)
                #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Remove those data with y coord which is above the 1/3 of the frame
                toDelete = []
                for yndex, y in enumerate(l[:, 1]):
                    if y < ((img_y_center * 2) / 3) * 2:
                        toDelete.append(yndex)
                l = np.delete(l, toDelete, 0)
        #::::::::::::::::::::::::::::::::::::::::::::::
	'''
        '''
        #********************************************** Plot the center dot/arrow line
        if (len(left_lane) > 0 and len(right_lane) > 0):
            L = len(right_lane[0]) if (len(left_lane[0]) > len(right_lane[0])) else len(left_lane[0])
            if (len(left_lane[0]) > len(right_lane[0])):
                N = np.arange(len(left_lane[0]) - len(right_lane[0]))
                #left_lane[0] = np.delete(left_lane[0], N, 0)
                center_lane = (np.delete(left_lane[0], N, 0) + right_lane[0]) / 2
            else:
                N = np.arange(len(right_lane[0]) - len(left_lane[0]))
                #right_lane[0] = np.delete(right_lane[0], N, 0)
                center_lane = (left_lane[0] + np.delete(right_lane[0], N, 0)) / 2
            #print(center_lane)
            
            #for i in range(0, L / 5):
            #    cv2.circle(img = mask_image, center = tuple(center_lane[i * 5, :]), radius = 2, color = (0, 100, 200), thickness = -1, lineType = 4)
            
            for i in range(0, 5):
                if i == 0:
                    continue
                    #cv2.arrowedLine(img = mask_image, pt1 = tuple(center_lane[(L / 4) * (i + 1), :]), pt2 = tuple(center_lane[0, :]), color = (0, 100, 200), thickness = 5, tipLength = 1)
                elif i < 4:
                    cv2.line(img = mask_image, pt1 = tuple(center_lane[(L / 4) * (i + 1) - 1, :]), pt2 = tuple(center_lane[(L / 4) * i, :]), color = (0, 100, 200), thickness = 5)
                #else:
                #    cv2.line(img = mask_image, pt1 = tuple(center_lane[(L / 5) * (i + 1), :]), pt2 = tuple(center_lane[(L / 5) * i, :]), color = (0, 100, 200), thickness = 5)
            #cv2.arrowedLine(img = mask_image, pt1 = tuple(center_lane[-1, :]), pt2 = tuple(center_lane[0, :]), color = (0, 100, 200), thickness = 5, tipLength = 0.5)
            #cv2.polylines(img = mask_image, pts = center_lane, isClosed = False, color = (0, 100, 200), thickness = 2)
        #**********************************************
        '''
        '''
        #////////////////////////////////////////////// Plot the rigion between two cloest lanes
        if (len(left_lane) > 0 and len(right_lane) > 0):
            roi = np.arange(40).reshape((20,2))
            L_mask_list = [0] * len(left_lane[0])
            R_mask_list = [0] * len(right_lane[0])
            for i in range(0, 10):
                L_mask_list[i * (len(left_lane[0]) / 10)] = 1
                R_mask_list[i * (len(right_lane[0]) / 10)] = 1
            L_mask = np.array(L_mask_list, dtype = bool)
            R_mask = np.array(R_mask_list, dtype = bool)
            roi[0:10, :] = left_lane[0][L_mask, :]
            roi[10:20, :] = np.flip(right_lane[0][R_mask, :], axis = 0)
            roi = np.array([roi])

            cv2.fillPoly(img = mask_image, pts = roi, color = (255, 0, 0))
        #//////////////////////////////////////////////
        '''
        #if len(left_lane) > 0:
        #    print(left_lane[0])
        #++++++++++++++++++++++++++++++++++++++++++++++ Plot all the lanes on to the image
        for index, i in enumerate(left_lane):
            color = (0, 255, 0)
            thickness = 2
            if index > 0:
                color = (0, 0, 255)
                thickness = 1
            i = np.array([i])
            cv2.polylines(img = mask_image, pts = i, isClosed = False, color = color, thickness = thickness)
        for index, j in enumerate(right_lane):
            color = (0, 255, 0)
            thickness = 2
            if index > 0:
                color = (0, 0, 255)
                thickness = 1
            j = np.array([j])
            cv2.polylines(img = mask_image, pts = j, isClosed = False, color = color, thickness = thickness)
        #++++++++++++++++++++++++++++++++++++++++++++++
            # mask_image[coord] = color
        #mask_img = self._transparent_back(mask_image)
        return mask_image, left_lane, right_lane


if __name__ == '__main__':
    binary_seg_image = cv2.imread('binary_ret.png', cv2.IMREAD_GRAYSCALE)
    binary_seg_image[np.where(binary_seg_image == 255)] = 1
    instance_seg_image = cv2.imread('instance_ret.png', cv2.IMREAD_UNCHANGED)
    ele_mex = np.max(instance_seg_image, axis=(0, 1))
    for i in range(3):
        if ele_mex[i] == 0:
            scale = 1
        else:
            scale = 255 / ele_mex[i]
        instance_seg_image[:, :, i] *= int(scale)
    embedding_image = np.array(instance_seg_image, np.uint8)
    cluster = LaneNetCluster()
    mask_image = cluster.get_lane_mask(instance_seg_ret=instance_seg_image, binary_seg_ret=binary_seg_image)
    plt.figure('embedding')
    plt.imshow(embedding_image[:, :, (2, 1, 0)])
    plt.figure('mask_image')
    plt.imshow(mask_image[:, :, (2, 1, 0)])
    plt.show()
