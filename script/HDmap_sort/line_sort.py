#!/usr/bin/env python

import rospy
import message_filters
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension

from math import *
import matplotlib.pyplot as plt
import numpy as np
import random
import csv
import pandas as pd

from vector_map_msgs.msg import PointArray
from vector_map_msgs.msg import LineArray
from vector_map_msgs.msg import NodeArray
from vector_map_msgs.msg import LaneArray
from vector_map_msgs.msg import WhiteLineArray
from vector_map_msgs.msg import DTLaneArray


rospy.init_node('hd_line_sort_node')

#def point_callback(data):
#    print(data)
#rospy.Subscriber(v_map_point_topic_name, PointArray, point_callback)
#rospy.spin()

v_map_point_topic_name = '/vector_map_info/point'
v_map_line_topic_name = '/vector_map_info/line'
v_map_node_topic_name = '/vector_map_info/node'
v_map_lane_topic_name = '/vector_map_info/lane'
v_map_whiteline_topic_name = '/vector_map_info/white_line'
v_map_dtlane_topic_name = '/vector_map_info/dtlane'

point_data = rospy.wait_for_message(v_map_point_topic_name, PointArray, timeout=None)
line_data = rospy.wait_for_message(v_map_line_topic_name, LineArray, timeout=None)
node_data = rospy.wait_for_message(v_map_node_topic_name, NodeArray, timeout=None)
lane_data = rospy.wait_for_message(v_map_lane_topic_name, LaneArray, timeout=None)
whiteline_data = rospy.wait_for_message(v_map_whiteline_topic_name, WhiteLineArray, timeout=None)
dtlane_data = rospy.wait_for_message(v_map_dtlane_topic_name, DTLaneArray, timeout=None)

point_data_pid = []
node_data_nid = []
for p in point_data.data:
    point_data_pid.append(p.pid)
for n in node_data.data:
    node_data_nid.append(n.nid)

#print(point_data.data[0])
wl_col = ['bwlid', 'fwlid', 'blid', 'flid', 'blnid', 'flnid', 'bpid', 'fpid', \
       'bx', 'by', 'bh', 'fx', 'fy', 'fh', 'width', 'color', 'type']
whiteline_df = pd.DataFrame(columns = wl_col)
tmp_df = pd.DataFrame(columns = wl_col)
group = []
for wl in whiteline_data.data:
    if line_data.data[wl.lid - 1].flid != 0:
        group.append(wl)
    else:
        group.append(wl)
        tmp_df['bwlid'] = [group[0].id]
        tmp_df['fwlid'] = [group[-1].id]
        tmp_df['blid'] = [group[0].lid]
        tmp_df['flid'] = [group[-1].lid]
        tmp_df['blnid'] = [group[0].linkid]
        tmp_df['flnid'] = [group[-1].linkid]
        tmp_df['bpid'] = [line_data.data[group[0].lid - 1].bpid]
        tmp_df['fpid'] = [line_data.data[group[-1].lid - 1].fpid]
        i = np.where(np.array(point_data_pid) == tmp_df['bpid'][0])
        j = np.where(np.array(point_data_pid) == tmp_df['fpid'][0])
        tmp_df['bx'] = [point_data.data[i[0][0]].ly]
        tmp_df['by'] = [point_data.data[i[0][0]].bx]
        tmp_df['bh'] = [point_data.data[i[0][0]].h]
        tmp_df['fx'] = [point_data.data[j[0][0]].ly]
        tmp_df['fy'] = [point_data.data[j[0][0]].bx]
        tmp_df['fh'] = [point_data.data[j[0][0]].h]
        tmp_df['width'] = [group[0].width]
        tmp_df['color'] = [group[0].color]
        tmp_df['type'] = [group[0].type]
        whiteline_df = pd.concat([whiteline_df, tmp_df])
        group = []
        #break
print(whiteline_df)
whiteline_df.to_csv('whiteline_df.csv', index = 0)

ln_col = ['blnid', 'flnid', 'bdid', 'fdid', 'bpid', 'fpid', 'bx', 'by', 'bh', 'fx', 'fy', 'fh', 'bx_1', 'by_1', 'bh_1',\
          'lcnt', 'lno', 'lanetype', 'limitvel', 'lanecfgfg', 'blid1', 'blid2', 'blid3', 'blid4', \
          'flid1', 'flid2', 'flid3', 'flid4']
lane_df = pd.DataFrame(columns = ln_col)
tmp_df = pd.DataFrame(columns = ln_col)
group = []
for ln_idx, ln in enumerate(lane_data.data):
    if ln.flid != ln.lnid + 1 or ln.flid2 != 0 or ln.flid3 != 0 or ln.flid4 != 0 or (ln_idx + 1 == len(lane_data.data)):
        group.append(ln)
        tmp_df['blnid'] = [group[0].lnid]
        tmp_df['flnid'] = [group[-1].lnid]
        tmp_df['bdid'] = [group[0].did]
        tmp_df['fdid'] = [group[-1].did]
        p = np.where(np.array(node_data_nid) == group[0].bnid)
        q = np.where(np.array(node_data_nid) == group[-1].fnid)
        r = np.where(np.array(node_data_nid) == group[1].bnid)
        tmp_df['bpid'] = [node_data.data[p[0][0]].pid]
        tmp_df['fpid'] = [node_data.data[q[0][0]].pid]
        i = np.where(np.array(point_data_pid) == tmp_df['bpid'][0])
        j = np.where(np.array(point_data_pid) == tmp_df['fpid'][0])
        k = np.where(np.array(point_data_pid) == node_data.data[r[0][0]].pid)
        tmp_df['bx'] = [point_data.data[i[0][0]].ly]
        tmp_df['by'] = [point_data.data[i[0][0]].bx]
        tmp_df['bh'] = [point_data.data[i[0][0]].h]
        tmp_df['fx'] = [point_data.data[j[0][0]].ly]
        tmp_df['fy'] = [point_data.data[j[0][0]].bx]
        tmp_df['fh'] = [point_data.data[j[0][0]].h]
        tmp_df['bx_1'] = [point_data.data[k[0][0]].ly]
        tmp_df['by_1'] = [point_data.data[k[0][0]].bx]
        tmp_df['bh_1'] = [point_data.data[k[0][0]].h]
        tmp_df['lcnt'] = [group[0].lcnt]
        tmp_df['lno'] = [group[0].lno]
        tmp_df['lanetype'] = [group[0].lanetype]
        tmp_df['limitvel'] = [group[0].limitvel]
        tmp_df['lanecfgfg'] = [group[0].lanecfgfg]
        tmp_df['blid1'] = [group[0].blid]
        tmp_df['blid2'] = [group[0].blid2]
        tmp_df['blid3'] = [group[0].blid3]
        tmp_df['blid4'] = [group[0].blid4]
        tmp_df['flid1'] = [group[-1].flid]
        tmp_df['flid2'] = [group[-1].flid2]
        tmp_df['flid3'] = [group[-1].flid3]
        tmp_df['flid4'] = [group[-1].flid4]
        lane_df = pd.concat([lane_df, tmp_df])
        group = []
        #break
    else:
        group.append(ln)
print(lane_df)
lane_df.to_csv('lane_df.csv', index = 0)
