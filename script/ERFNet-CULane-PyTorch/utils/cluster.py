

import numpy as np
#import glog as log
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
import time
import warnings
import cv2
import math
import PIL.Image as pil
try:
    from cv2 import cv2
except ImportError:
    pass


class ERFNetCluster(object):

    
    def get_lane_area(self, merge, instance_seg):
    #print(instance_seg_ret.shape) # (288, 800, 4)
        lane_embedding_feats = []
        lane_coordinate = []
        
        idx = np.where(merge >= 1)
        lane_embedding_feats = instance_seg[idx[0], idx[1]]
        lane_coordinate = np.concatenate(([idx[0]], [idx[1]]), axis = 0).transpose((1, 0))
        
        return np.array(lane_embedding_feats, np.float32), np.array(lane_coordinate, np.int64)
    
    def _thresh_coord(self, coord):
        """
        :param coord: [(x, y)]
        :return:
        """
        pts_x = coord[:, 0]
        mean_x = np.mean(pts_x)

        idx = np.where(np.abs(pts_x - mean_x) < mean_x)
        #idx = np.where(np.abs(pts_x - mean_x) < 75)

        return coord[idx[0]]

    def get_lane_mask(self, instance_seg, selected_lines):
        
        mask_list = []
        cofficients = []
        points_on_image = np.zeros((4, 2, 2)).astype(int)

        #print(instance_seg_list[0])
        
        for i in range(selected_lines):
            f = instance_seg[:, :, i]
            mask_img = []
            mask_img = cv2.cvtColor(np.asarray(pil.new(mode = 'RGBA', size = (f.shape[1], f.shape[0]))), cv2.COLOR_RGBA2RGB)
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
            mask_img = mask_img.astype(np.uint8)
            
            frame = (f * 255).astype(np.uint8)
            idx = np.where(frame >= 60)
            #print(frame.shape) # (208, 448)
            if len(idx[0]) > 0 and len(idx[1]) > 0:
                row = idx[0]
                column = idx[1]
                max_X = max(column) if len(column > 0) else f.shape[1]
                min_X = min(column) if len(column > 0) else 0
                max_Y = max(row) if len(row > 0) else f.shape[0]
                min_Y = min(row) if len(row > 0) else f.shape[0] / 2

                coord = np.concatenate(([idx[0]], [idx[1]]), axis = 0).transpose((1, 0))
                coord = np.flip(coord, axis=1)
                coord = self._thresh_coord(coord)
                
                ############# Recieve lane line
                X = []
                Y = np.unique(coord[:, 1])
                for y in Y:
                    idx_x = np.where(coord[:, 1] == y)
                    X.append(np.mean(coord[idx_x, 0]))
                X = np.array(X)
                line = np.concatenate(([X], [Y]), axis = 0).transpose((1, 0))
                
                ############# Recieve 1st order polynomial line
                if len(X) > 5 and len(Y) > 5:
                    #C = np.polyfit(X, Y, 1)
                    #poly_X = np.arange(min_X, max_X, 1)
                    #poly_Y = (C[0] * poly_X + C[1])

                    C = np.polyfit(Y, X, 1)
                    poly_Y = np.arange(min_Y, max_Y, 1)
                    poly_X = (C[0] * poly_Y + C[1])
                
                    poly_line = np.concatenate(([poly_X], [poly_Y]), axis = 0).transpose((1, 0))
                    poly_line = poly_line.astype(int)

                    ############# Delete those points which are far from polynomial line
                    D = np.abs(C[0] * line[:, 1] - line[:, 0] + C[1]) / np.sqrt(C[0] ** 2 + 1)
                    #mean_D = np.mean(D)
                    D_idx = np.where(D > 5)
                    line = np.delete(line, D_idx, axis = 0)
                    
                    fit_min_X = int(C[0] * min_Y + C[1])
                    fit_max_X = int(C[0] * max_Y + C[1])
                    points_on_image[i, 0] = [fit_min_X, min_Y]
                    points_on_image[i, 1] = [fit_max_X, max_Y]
                    '''
                    ############# Modified 1st order polynomial line
                    if len(line[:, 0]) > 5 and len(line[:, 1]) > 5:
                        Modified_C = np.polyfit(line[:, 0], line[:, 1], 1)
                        cofficients.append(Modified_C)
                        fit_min_X = int((min_Y - Modified_C[1]) / Modified_C[0])
                        fit_max_X = int((max_Y - Modified_C[1]) / Modified_C[0])
                        points_on_image[i, 0] = [fit_min_X, min_Y]
                        points_on_image[i, 1] = [fit_max_X, max_Y]
                    
                    else:
                        cofficients.append([])
                    '''
                    #cv2.polylines(img = mask_img, pts = [poly_line], isClosed = False, color = (255, 255, 255), thickness = 2)
                line = line.astype(int)
                #cv2.polylines(img = mask_img, pts = [line], isClosed = False, color = (255, 255, 255), thickness = 5)
                cv2.polylines(img = mask_img, pts = [points_on_image[i]], isClosed = False, color = (255, 255, 255), thickness = 5)
                

                mask_list.append(mask_img)
            else:
                mask_list.append(mask_img)
                continue
        #print(points_on_image)
        #print(cofficients)
        return mask_list, points_on_image

