import rospy
from vector_map_msgs.msg import PointArray
from vector_map_msgs.msg import LineArray
from vector_map_msgs.msg import NodeArray
from vector_map_msgs.msg import LaneArray
from vector_map_msgs.msg import WhiteLineArray          
from vector_map_msgs.msg import DTLaneArray

rospy.init_node('line_sort_test_node')

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

# print(point_data.data[0])

# for p in point_data.data:
#     print(p.bx)
group = []
for ln_idx, ln in enumerate(lane_data.data):
    # group.append(ln)
    print(ln[0])
