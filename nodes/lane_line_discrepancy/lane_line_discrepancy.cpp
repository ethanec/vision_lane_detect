#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <math.h>
#include <list>
#include "utils.h"
#include <string>
#include <csignal>

#include <fstream>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>
#include <visualization_msgs/Marker.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseStamped.h>

#include <ros/console.h>
#include <tf/transform_datatypes.h>
#include <std_msgs/Bool.h>
#include <visualization_msgs/MarkerArray.h>
//#include <vector_map/vector_map.h>
//#include <map_file/get_file.h>
#include <sys/stat.h>

#include "ros/ros.h"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <tf/tf.h>
#include <tf/time_cache.h>
#include <tf/transform_listener.h>

#include <vector_map_msgs/PointArray.h>
#include <vector_map_msgs/LineArray.h>
#include <vector_map_msgs/NodeArray.h>
#include <vector_map_msgs/LaneArray.h>

using namespace sensor_msgs;
using namespace geometry_msgs;
using namespace message_filters;

static ros::Publisher lane_line;

void lane_line_discrepancy_callback(const Image &img) {
    cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
    lane_line.publish(cv_image->toImageMsg());
}

int main(int argc, char *argv[]) {
    //std::cout << "Hello World!" << std::endl;
    //signal(SIGINT, sig_handler);
    
    ros::init(argc, argv, "lane_line_discrepancy");
    ros::NodeHandle n;

    std::string laneNet_topic_name = "/laneNet_image";

    ros::Subscriber subscriber = n.subscribe(laneNet_topic_name, 1, lane_line_discrepancy_callback);
  
    lane_line = n.advertise<Image>("lane_line_discrepancy_image", 1);
    
    ros::spin();

    return 0;

}