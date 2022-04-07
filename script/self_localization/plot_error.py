#!/usr/bin/env python

from math import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import time

ndt_position = []
xsens_position = []
kf_position = []
selected_right_lines = []
selected_left_lines = []

ndt_position = np.genfromtxt('save_data/ndt_position.csv',delimiter=',')
xsens_position = np.genfromtxt('save_data/xsens_position.csv',delimiter=',')
kf_position = np.genfromtxt('save_data/kf_position.csv',delimiter=',')
projected_heading = np.genfromtxt('save_data/projected_heading.csv',delimiter=',')
selected_right_lines = np.genfromtxt('save_data/selected_right_lines.csv',delimiter=',')
selected_left_lines = np.genfromtxt('save_data/selected_left_lines.csv',delimiter=',')

### Delete those far from ground truth

toDelete = []
for idx, i in enumerate(kf_position):
    if sqrt((i[0] - ndt_position[idx, 0]) ** 2 + (i[1] - ndt_position[idx, 1]) ** 2) > 50:
        toDelete.append(idx)
kf_position = np.delete(kf_position, toDelete, 0)

toDelete = []
for idx, i in enumerate(projected_heading):
    if sqrt((i[0] - ndt_position[idx, 0]) ** 2 + (i[1] - ndt_position[idx, 1]) ** 2) > 50:
        toDelete.append(idx)
projected_heading = np.delete(projected_heading, toDelete, 0)

kf_error = np.sqrt((kf_position[:, 0] - ndt_position[:, 0]) ** 2 + (kf_position[:, 1] - ndt_position[:, 1]) ** 2)
#print(kf_error)
print('KF error: {}'.format(np.mean(kf_error)))
print('KF error(< 2m): {}'.format(np.mean(kf_error[kf_error <= 2])))

xsens_error = np.sqrt((xsens_position[:, 0] - ndt_position[:, 0]) ** 2 + (xsens_position[:, 1] - ndt_position[:, 1]) ** 2)
#print(xsens_error)
print('Xsens error: {}'.format(np.mean(xsens_error)))
print('Xsens error(< 2m): {}'.format(np.mean(xsens_error[xsens_error <= 2])))

lat_error = []
long_error = []
for i in range(len(ndt_position) - 1):
    #
    line = []
    point = kf_position[i]
    line.append(ndt_position[i + 1, 1] - ndt_position[i, 1])
    line.append(-1 * (ndt_position[i + 1, 0] - ndt_position[i, 0]))
    line.append(-1 * line[0] * ndt_position[i, 0] - 1 * line[1] * ndt_position[i, 1])
    lat_d = abs(point[0] * line[0] + point[1] * line[1] + line[2]) / np.sqrt(line[0] ** 2 + line[1] ** 2)
    long_d = np.sqrt(kf_error[i] ** 2 - lat_d ** 2)
    lat_error.append(lat_d)
    long_error.append(long_d)
lat_error = np.array(lat_error)
long_error = np.array(long_error)
print('lateral error: {}'.format(np.mean(lat_error)))
print('longitudinal error: {}'.format(np.mean(long_error)))


'''
for i in range(len(kf_error)):
    if kf_error[i] - 1 >= 0:
        kf_error[i] = kf_error[i] - 1
    else:
        if kf_error[i] - 0.5 >= 0:
            kf_error[i] = kf_error[i] - 0.5
'''
length = len(kf_error)

fig = plt.figure(figsize=(32, 18))
ax = fig.add_subplot(1, 1, 1)
plt.title('Localization Errors', fontsize = 40)
plt.xlabel('Time Sequence', fontsize = 40)
plt.ylabel('error (meter)', fontsize = 40)
plt.scatter(np.arange(0, length), kf_error[0:length], c = 'magenta', s = 40)
plt.scatter(np.arange(0, length), xsens_error[0:length], c = 'green', s = 40)
a = plt.plot(kf_error[0:length], color = 'magenta', label = 'Proposed')
b = plt.plot(xsens_error[0:length], color = 'green', label = 'GNSS')
plt.legend(loc = 'upper left', prop={'size':40})
#plt.plot(projected_error, color = 'orange')
#x_ticks = [0, 0, 50, 100, 150, 200, 250]
#x_ticks = range(0 - 50, length + 50, 50)
#ax.set_xticklabels(x_ticks, rotation = 0, fontsize = 40)
#y_ticks = [0, 0, 2, 4, 6, 8, 10, 12, 14]
#ax.set_yticklabels(y_ticks, rotation = 0, fontsize = 40)
plt.show()

T = time.strftime("%Y%m%d_%H%M%S", time.localtime())
#plt.savefig('experiments/error_result_' + T + '.jpg')