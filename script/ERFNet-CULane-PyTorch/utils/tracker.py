

import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import json
import random

class LaneTracker(object):

    def right_line_tracker(self, lstm_model, past_5RL):
        past_5RL = np.array(past_5RL)
        x = []
        for l in range(len(past_5RL)):
            x.append(np.linspace(past_5RL[l, 0, 0], past_5RL[l, 1, 0], 19))
        x = np.array(x)
        x = x / 448
        nn_pred_y = (lstm_model.predict([x.reshape((1, 5, 19)), x[:, 0].reshape((1, 5, 1)), x[:, -1].reshape((1, 5, 1))])).ravel()
        result = np.array([[nn_pred_y[0] * 448, int(240 - 240)], [nn_pred_y[-1] * 448, int(448 - 240)]])
        #print(result)

        return result