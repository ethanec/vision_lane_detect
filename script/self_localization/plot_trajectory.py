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
'''
with open('ndt_position.csv') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        print(row)
        ndt_position.append(row)
        break
with open('xsens_position.csv') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        xsens_position.append(row)
with open('kf_position.csv') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        kf_position.append(row)
'''
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

### Add one axis with all 0
#ndt_position = np.concatenate((ndt_position, np.zeros((len(ndt_position), 1))), axis = 1)
#xsens_position = np.concatenate((xsens_position, np.zeros((len(xsens_position), 1))), axis = 1)
#kf_position = np.concatenate((kf_position, np.zeros((len(kf_position), 1))), axis = 1)
#projected_heading = np.concatenate((projected_heading, np.zeros((len(projected_heading), 1))), axis = 1)
'''
### Plot 3d image
fig = plt.figure(figsize=(32, 18))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ndt_position[:, 0], ndt_position[:, 1], ndt_position[:, 2], c = 'gray')
ax.scatter(xsens_position[:, 0], xsens_position[:, 1], xsens_position[:, 2], c = 'green')
ax.scatter(kf_position[:, 0], kf_position[:, 1], kf_position[:, 2], c = 'red')

T = time.strftime("%Y%m%d_%H%M%S", time.localtime())
plt.savefig('result_' + T + '.jpg')
'''
### Plot 2d image
fig = plt.figure(figsize=(32, 18))
ax = fig.add_subplot(1, 1, 1)
plt.title('Fusion Results', fontsize = 40)
plt.xlabel('East', fontsize = 40)
plt.ylabel('North', fontsize = 40)
#a = plt.scatter(ndt_position[:, 0], ndt_position[:, 1], c = 'gray')
#b = plt.scatter(xsens_position[:, 0], xsens_position[:, 1], c = 'green')
#c = plt.scatter(kf_position[:, 0], kf_position[:, 1], c = 'red')
#d = plt.scatter(projected_heading[:, 0], projected_heading[:, 1], c = 'orange')
a = plt.scatter(ndt_position[:, 0], ndt_position[:, 1], c = 'gray')
b = plt.scatter(xsens_position[:, 0], xsens_position[:, 1], c = 'green')
c = plt.scatter(kf_position[:, 0], kf_position[:, 1], c = 'red')
d = plt.scatter(projected_heading[:, 0], projected_heading[:, 1], c = 'orange')


plt.plot(ndt_position[:, 0], ndt_position[:, 1], c = 'gray')
plt.plot(xsens_position[:, 0], xsens_position[:, 1], c = 'green')
plt.plot(kf_position[:, 0], kf_position[:, 1], c = 'red')
plt.plot(projected_heading[:, 0], projected_heading[:, 1], c = 'orange')
for l in selected_right_lines:
    plt.plot([l[0], l[3]], [l[1], l[4]], color='blue')
for l in selected_left_lines:
    plt.plot([l[0], l[3]], [l[1], l[4]], color='cyan')
plt.legend([a, b, c, d], ['GT', 'GNSS', 'Proposed', 'Lane'], loc = 'upper right', prop={'size':40})
s = 200
plt.scatter(ndt_position[s, 0], ndt_position[s, 1], c = 'blue', s = 50)
plt.scatter(xsens_position[s, 0], xsens_position[s, 1], c = 'black', s = 50)
plt.scatter(kf_position[s, 0], kf_position[s, 1], c = 'cyan', s = 50)


x_ticks = range(6720, 6860, 20)
ax.set_xticklabels(x_ticks, rotation = 0, fontsize = 40)
y_ticks = range(-8340, -8160, 20)
ax.set_yticklabels(y_ticks, rotation = 0, fontsize = 40)

T = time.strftime("%Y%m%d_%H%M%S", time.localtime())
#plt.savefig('experiments/result_' + T + '.jpg')
plt.show()
