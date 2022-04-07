#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <math.h>
#include <list>
#include "utils.h"
#include <string>
#include <csignal>
#include <map>
#include <fstream>
#include <iterator>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>
#include <visualization_msgs/Marker.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Imu.h>
#include <dbw_mkz_msgs/SteeringReport.h>

#include <ros/console.h>
#include <tf/transform_datatypes.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Int32MultiArray.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>
#include <visualization_msgs/MarkerArray.h>
//#include <vector_map/vector_map.h>
//#include <map_file/get_file.h>
#include <sys/stat.h>

#include <vision_lane_detect/Int32MultiArray_H.h>
#include <vision_lane_detect/Float32MultiArray_H.h>

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
#include <vector_map_msgs/WhiteLineArray.h>

#define PI 3.14159265
#define GO 0.1

using namespace sensor_msgs; // define all msgs under namespace
using namespace geometry_msgs;
using namespace message_filters;
using namespace dbw_mkz_msgs;

//static volatile int keepRunning = 1;
/*
float cameraMat[9] = {9.8253595454109904e+02, 0., 4.8832572821891949e+02, \
                        0., 9.7925117255173427e+02, 3.0329591096350060e+02, \
                        0., 0., 1. };
*/
//float cameraMat[9] = {2.0808733974132961e+03, 0., 8.9252440162446635e+02, 0., 2.0734840157739636e+03, 6.5658574940950098e+02, 0., 0., 1. };
float cameraMat[9] = {8.2976148355956389e+02, 0., 3.9741419142658657e+02, 0., 8.2676716862764215e+02, 2.6665497653419578e+02, 0., 0., 1.};
float cameraExtrinsicMat[16] = {0.0786098732349, -0.0270551753262, 0.99653826084, 1.9227830929028441, \
                                -0.996855485844, 0.00787538826984, 0.078848707043, 1.1042030956136471, \
                                -0.00998139132316, -0.999602919037, -0.0263510166708, 3.5798012968786721e-01, \
                                0., 0., 0., 1.};

float distCoeff[5] = {-2.6104330125762382e-01, -4.5664387654405406e-02, -5.6132057565511703e-04, 5.8439981517010653e-03, 3.7747672592173037e-01};
/*
float cameraMat[9] = {1.2851714246138999e+03, 0., 4.8575906309844117e+02, \
                        0., 1.2557353564487087e+03, 2.9935654509060248e+02, \
                        0., 0., 1.};
*/
int imageCoord1[2] = {0, 0};
int imageCoord2[2] = {0, 0};
int estimate_range = 15;
float measure_range = 7.5;
float lateral_measure_range = 3;
bool first_callback = true;
float last_last_car_position[3] = {0, 0, 0};
float last_car_position[3] = {0, 0, 0};
float present_car_position[3] = {0, 0, 0};
float imu_car_position[3] = {0, 0, 0};
float ne_vector[3] = {0, 0, 0};
float last_ne_vector[3] = {0, 0, 0};
float unit_ne_vector[3] = {0, 0, 0};
float unit_lane_vector[3] = {0, 0, 0};
float genji_lane_position[3] = {0, 0, 0};
float estimate_point[3] = {0, 0, 0}; // (6xxx, -8xxx)
float long_coefficients[3] = {0, 0, 0};
float lat_coefficients[3] = {0, 0, 0};
float ratio_to_one_meter = 0;
float measure_distance = 0, cell_w = 0, cell_l = 0;
float farest_distance = 0;
float X = 0;
float Y = 0;
float Z = 0;
float X_xsens = 0;
float Y_xsens = 0;
float Z_xsens = 0;

int lane_idx = 0, last_lane_idx = 0;
int seg_lane_idx = 0, last_seg_lane_idx = 0, last_last_seg_lane_idx = 0; 
int r_whiteline_idx = 0, l_whiteline_idx = 0, last_r_whiteline_idx = 0, last_l_whiteline_idx = 0;
float last_lane_vec[3] = {0, 0, 0};
float closest_distance = 0;
int trigger = 0;
int theta = 45;
int repeat = 0;
int iteration = 0;

class HDLanePipeline {
private:
    double t_tmp;
    double last_t_tmp;
    float duration, last_duration;
    float x_s, x_s_, y_s, y_s_;
    float x_v, x_v_, y_v, y_v_;
    float V;

    ros::NodeHandle n;
    ros::Publisher image_lane;
    ros::Publisher fuse_image_lane;
    ros::Publisher world_line_points;
    ros::Publisher image_line_points;
    ros::Publisher world_lane_points;
    
    //tf::TransformListener *listeners_ptr1, *listeners_ptr2;
    PoseStamped point_pose_in_map, point_pose_in_camera, point_pose_in_xsens, lane_point_in_map;

    std::map<int, vector_map_msgs::Point> Point_Map;
    std::map<int, vector_map_msgs::Line> Line_Map;
    std::map<int, vector_map_msgs::Node> Node_Map;
    std::map<int, vector_map_msgs::Lane> Lane_Map;
    std::map<int, vector_map_msgs::WhiteLine> WhiteLine_Map;

    std::vector<vector_map_msgs::Point> selected_points;
    std::vector<vector_map_msgs::Node> selected_nodes;
    std::vector<vector_map_msgs::Lane> selected_lanes;
    std::vector<vector_map_msgs::Line> selected_lines;
    std::vector<vector_map_msgs::WhiteLine> selected_whitelines;
    std::vector< std::vector<int> > selected_nodes_of_lanes;

    std::vector< std::vector<int> > selected_points_of_nodes;
    std::vector< std::vector< std::vector<float> > > points_to_transform;
    std::vector< std::vector< std::vector<float> > > points_for_whitelines;

    std::vector<int> selected_lines_of_whitelines;
    std::vector< std::vector<int> > selected_points_of_lines;

    //std_msgs::Float32MultiArray world_points_array;
    //std_msgs::Int32MultiArray image_points_array;

    // Custom Message
    vision_lane_detect::Float32MultiArray_H world_points_array;
    vision_lane_detect::Int32MultiArray_H image_points_array;

    std::vector<int> next_lanes;
    std::vector<std::vector<float> > imu_csv;
    std::vector<std::vector<float> > lane_csv;
    std::vector<std::vector<float> > whiteline_csv;
    
    
    struct Lane_Data_Sorted {
        int blnid, flnid, bdid, fdid, bpid, fpid, lcnt, lno, lanetype, limitvel, lanecfgfg;
        int blid1, blid2, blid3, blid4, flid1, flid2, flid3, flid4;
        float bx, by, bh, fx, fy, fh, bx_1, by_1, bh_1;
    };
    struct WhiteLine_Data_Sorted {
        int bwlid, fwlid, blid, flid, blnid, flnid, bpid, fpid, color, type;
        float bx, by, bh, fx, fy, fh;
        float width;
    };
    
    std::vector<Lane_Data_Sorted> lane_data_sorted;
    std::vector<WhiteLine_Data_Sorted> whiteline_data_sorted;
    std::vector<int> toDelete;

    struct Points_Coords{
        float bx, by, bh, fx, fy, fh;
    };
    std::map<int, Points_Coords> lane_points_coords;
    std::map<int, Points_Coords> whiteline_points_coords;
    
    void point_Callback(const vector_map_msgs::PointArray msg_array) {
        for (int i = 0; i < msg_array.data.size(); ++i) {
            Point_Map.insert(std::pair<int, vector_map_msgs::Point>(msg_array.data[i].pid, msg_array.data[i]));
        }
    }
    
    void line_Callback(const vector_map_msgs::LineArray msg_array) {
        for (int i = 0; i < msg_array.data.size(); ++i) {
            Line_Map.insert(std::pair<int, vector_map_msgs::Line>(msg_array.data[i].lid, msg_array.data[i]));
        }
    }

    void node_Callback(const vector_map_msgs::NodeArray msg_array) {
        for (int i = 0; i < msg_array.data.size(); ++i) {
            Node_Map.insert(std::pair<int, vector_map_msgs::Node>(msg_array.data[i].nid, msg_array.data[i]));
        }
    }
    
    void lane_Callback(const vector_map_msgs::LaneArray msg_array) {
        for (int i = 0; i < msg_array.data.size(); ++i) {
            Lane_Map.insert(std::pair<int, vector_map_msgs::Lane>(msg_array.data[i].lnid, msg_array.data[i]));
        }
        for (std::map<int, vector_map_msgs::Lane>::iterator it = Lane_Map.begin(); it != Lane_Map.end(); ++it) {
            //int bn = (it->second).bnid;
            //int bp = (Node_Map.find(bn)->second).pid;
            Points_Coords pc;
            pc.bx = (Point_Map.find( (Node_Map.find( (it->second).bnid )->second).pid )->second).ly;
            pc.by = (Point_Map.find( (Node_Map.find( (it->second).bnid )->second).pid )->second).bx;
            pc.bh = (Point_Map.find( (Node_Map.find( (it->second).bnid )->second).pid )->second).h;
            pc.fx = (Point_Map.find( (Node_Map.find( (it->second).fnid )->second).pid )->second).ly;
            pc.fy = (Point_Map.find( (Node_Map.find( (it->second).fnid )->second).pid )->second).bx;
            pc.fh = (Point_Map.find( (Node_Map.find( (it->second).fnid )->second).pid )->second).h;
            lane_points_coords.insert( std::pair<int, Points_Coords>( (it->second).lnid, pc ) );
        }
    }
    
    void whiteline_Callback(const vector_map_msgs::WhiteLineArray msg_array) {
        //int n = 0;
        for (int i = 0; i < msg_array.data.size(); ++i) {
            WhiteLine_Map.insert(std::pair<int, vector_map_msgs::WhiteLine>(msg_array.data[i].id, msg_array.data[i])); // key may be linkid
            //if (WhiteLine_Map.find(msg_array.data[i].lnid * 100)) {
            //    n = n + 1;
            //} else {
            //    n = 0;
            //}
            //WhiteLine_Map.insert(std::pair<int, vector_map_msgs::WhiteLine>(msg_array.data[i].lnid * 100 + n, msg_array.data[i]));
        }
        for (std::map<int, vector_map_msgs::WhiteLine>::iterator it = WhiteLine_Map.begin(); it != WhiteLine_Map.end(); ++it) {
            //int bn = (it->second).bnid;
            //int bp = (Node_Map.find(bn)->second).pid;
            Points_Coords pc;
            pc.bx = (Point_Map.find( (Line_Map.find( (it->second).lid )->second).bpid )->second).ly;
            pc.by = (Point_Map.find( (Line_Map.find( (it->second).lid )->second).bpid )->second).bx;
            pc.bh = (Point_Map.find( (Line_Map.find( (it->second).lid )->second).bpid )->second).h;
            pc.fx = (Point_Map.find( (Line_Map.find( (it->second).lid )->second).fpid )->second).ly;
            pc.fy = (Point_Map.find( (Line_Map.find( (it->second).lid )->second).fpid )->second).bx;
            pc.fh = (Point_Map.find( (Line_Map.find( (it->second).lid )->second).fpid )->second).h;
            whiteline_points_coords.insert( std::pair<int, Points_Coords>( (it->second).id, pc ) );
        }
    }

    float Distance(float *point, std::vector<float> line) {
        float d = std::abs(point[0] * line[0] + point[1] * line[1] + line[2]) / sqrt(pow(line[0], 2) + pow(line[1], 2));
        return d;
    }

    void HDPipeline_Callback(const ImageConstPtr &img, const PoseStampedConstPtr &pos, const SteeringReportConstPtr &sr) {
        try {
            //printf("(x, y, z): (%f, %f, %f)\n", pos->pose.position.x, pos->pose.position.y, pos->pose.position.z);
            cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
            //cv_bridge::CvImagePtr cv_laneNet_image = cv_bridge::toCvCopy(laneNet, sensor_msgs::image_encodings::BGR8);
            
            IplImage frame = cv_image->image;

            if (cv_image->image.empty()) {
                printf("Empty /image_raw Image!");
            }
            //if (cv_laneNet_image->image.empty()) {
            //    printf("Empty /laneNet_image Image!");
            //}

            world_points_array.data.clear();
            image_points_array.data.clear();
            //std::cout << frame.width << ", " << frame.height << std::endl;

            if (first_callback) {
                first_callback = false;
                
                t_tmp = pos->header.stamp.toSec();
                last_t_tmp = t_tmp;
                /*
                x_v_ = 0.0;
                y_v_ = 0.0;
                duration = t_tmp - last_t_tmp;
                last_duration = t_tmp - last_t_tmp;
                */
                last_car_position[0] = pos->pose.position.x; // Baselink to Localizer
                last_car_position[1] = pos->pose.position.y;
                last_car_position[2] = pos->pose.position.z;
                last_last_car_position[0] = pos->pose.position.x; 
                last_last_car_position[1] = pos->pose.position.y;
                last_last_car_position[2] = pos->pose.position.z;
            } else {
                
                t_tmp = pos->header.stamp.toSec();
                duration = t_tmp - last_t_tmp;
                last_t_tmp = t_tmp;
                
                //std::cout << "Duration: " << duration << std::endl;
                
                present_car_position[0] = pos->pose.position.x; // Baselink to Localizer
                present_car_position[1] = pos->pose.position.y;
                present_car_position[2] = pos->pose.position.z;
                //---------------------------------- Estimate front point
                ne_vector[0] = present_car_position[0] - last_car_position[0];
                ne_vector[1] = present_car_position[1] - last_car_position[1];
                ne_vector[2] = present_car_position[2] - last_car_position[2];
                ratio_to_one_meter = 1 / sqrt(pow(ne_vector[0], 2) + pow(ne_vector[1], 2) + pow(ne_vector[2], 2));
                unit_ne_vector[0] = ne_vector[0] * ratio_to_one_meter;
                unit_ne_vector[1] = ne_vector[1] * ratio_to_one_meter;
                unit_ne_vector[2] = ne_vector[2] * ratio_to_one_meter;
                estimate_point[0] = present_car_position[0] + unit_ne_vector[0] * estimate_range;
                estimate_point[1] = present_car_position[1] + unit_ne_vector[1] * estimate_range;
                estimate_point[2] = present_car_position[2] + unit_ne_vector[2] * estimate_range;
                long_coefficients[0] = unit_ne_vector[1];
                long_coefficients[1] = -1 * unit_ne_vector[0];
                long_coefficients[2] = unit_ne_vector[0] * estimate_point[1] - unit_ne_vector[1] * estimate_point[0];
                lat_coefficients[0] = unit_ne_vector[0];
                lat_coefficients[1] = unit_ne_vector[1];
                lat_coefficients[2] = -1 * unit_ne_vector[0] * present_car_position[0] - unit_ne_vector[1] * present_car_position[1];

                if (sqrt(pow(ne_vector[0], 2) + pow(ne_vector[1], 2) + pow(ne_vector[2], 2)) > 0.2) {
                    iteration++;
                    std::cout << "ITERATION: " << iteration << std::endl;
                    //////////////////
                    // Japan HD Map //
                    //////////////////
                    std::vector<float> Horiz_Line;
                    std::vector<float> Verti_Line;
                    std::vector<float> distances;
                    std::vector<int> distance_idx;
                    Horiz_Line.push_back(ne_vector[0]);
                    Horiz_Line.push_back(ne_vector[1]);
                    Horiz_Line.push_back(-1 * ne_vector[0] * present_car_position[0] - 1 * ne_vector[1] * present_car_position[1]);
                    Verti_Line.push_back(ne_vector[1]);
                    Verti_Line.push_back(-1 * ne_vector[0]);
                    Verti_Line.push_back(ne_vector[0] * present_car_position[1] - ne_vector[1] * present_car_position[0]);
                    last_lane_vec[0] = lane_points_coords.find(last_seg_lane_idx)->second.fx - lane_points_coords.find(last_seg_lane_idx)->second.bx;
                    last_lane_vec[1] = lane_points_coords.find(last_seg_lane_idx)->second.fy - lane_points_coords.find(last_seg_lane_idx)->second.by;
                    for (std::map<int, Points_Coords>::iterator it = lane_points_coords.begin(); it != lane_points_coords.end(); ++it) {    
                        float *lane_center = new float[2];
                        lane_center[0] = 0.5 * (it->second.fx + it->second.bx);
                        lane_center[1] = 0.5 * (it->second.fy + it->second.by);
                        if (Distance(lane_center, Horiz_Line) <= 0.5 && \
                            /*Distance(lane_center, Verti_Line) <= 5 && \*/
                            (ne_vector[0] * (it->second.fx - it->second.bx) + \
                            ne_vector[1] * (it->second.fy - it->second.by)) / \
                            (sqrt(pow(ne_vector[0], 2) + pow(ne_vector[1], 2)) * \
                            sqrt(pow(it->second.fx - it->second.bx, 2) + pow(it->second.fy - it->second.by, 2))) > \
                            cos(theta * PI / 180)) {

                            //float p1_x = last_car_position[0], p1_y = last_car_position[1];
                            //float p2_x = present_car_position[0], p2_y = present_car_position[1];
                            //float v1_x = 0.5 * (it->second.fx + it->second.bx) - p1_x, v1_y = 0.5 * (it->second.fy + it->second.by) - p1_y;
                            //float v2_x = 0.5 * (it->second.fx + it->second.bx) - p2_x, v2_y = 0.5 * (it->second.fy + it->second.by) - p2_y;
                            float p1_x = it->second.bx, p1_y = it->second.by;
                            float p2_x = it->second.fx, p2_y = it->second.fy;
                            float v1_x = present_car_position[0] - p1_x, v1_y = present_car_position[1] - p1_y;
                            float v2_x = present_car_position[0] - p2_x, v2_y = present_car_position[1] - p2_y;
                            if ((p2_x - p1_x) * v1_x + (p2_y - p1_y) * v1_y < 0) {
                                distances.push_back( sqrt(pow(v1_x, 2) + pow(v1_y, 2)) );
                                distance_idx.push_back( it->first );
                            } else if ((p2_x - p1_x) * v2_x + (p2_y - p1_y) * v2_y > 0) {
                                distances.push_back( sqrt(pow(v2_x, 2) + pow(v2_y, 2)) );
                                distance_idx.push_back( it->first );
                            } else {
                                std::vector<float> Line;
                                Line.push_back(p2_y - p1_y);
                                Line.push_back(-1 * (p2_x - p1_x));
                                Line.push_back(-1 * (p2_y - p1_y) * p1_x - (-1 * (p2_x - p1_x)) * p1_y);
                                distances.push_back(Distance(present_car_position, Line));
                                distance_idx.push_back( it->first );
                                Line.clear();
                            }
                        }
                        delete [] lane_center;
                    }
                    if (distance_idx.size() > 0) {
                        seg_lane_idx = distance_idx[std::min_element(distances.begin(), distances.end()) - distances.begin()];
                    }
                    
                    for (int i = 0; i < lane_data_sorted.size(); ++i) {
                        if (seg_lane_idx <= lane_data_sorted[i].flnid && seg_lane_idx >= lane_data_sorted[i].blnid) {
                            lane_idx = i;
                        }
                    }

                    Horiz_Line.clear();
                    distances.clear();
                    distance_idx.clear();
                    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
                    std::cout << "Segment lane index: " << seg_lane_idx << std::endl;
                    std::cout << "Last Lane index: " << last_lane_idx << std::endl;
                    std::cout << "Lane index: " << lane_idx << std::endl;
                    //---------------------------------- At intersection
                    if (repeat < 2) {
                        float p_x = 0.5 * (lane_points_coords.find( seg_lane_idx )->second.bx + lane_points_coords.find( seg_lane_idx )->second.fx);
                        float p_y = 0.5 * (lane_points_coords.find( seg_lane_idx )->second.by + lane_points_coords.find( seg_lane_idx )->second.fy);
                        float p_x_ = 0.5 * (lane_points_coords.find( last_seg_lane_idx )->second.bx + lane_points_coords.find( last_seg_lane_idx )->second.fx);
                        float p_y_ = 0.5 * (lane_points_coords.find( last_seg_lane_idx )->second.by + lane_points_coords.find( last_seg_lane_idx )->second.fy);
                        float p_x__ = 0.5 * (lane_points_coords.find( last_last_seg_lane_idx )->second.bx + \
                                             lane_points_coords.find( last_last_seg_lane_idx )->second.fx);
                        float p_y__ = 0.5 * (lane_points_coords.find( last_last_seg_lane_idx )->second.by + \
                                             lane_points_coords.find( last_last_seg_lane_idx )->second.fy);
                        unit_lane_vector[0] = (p_x_ - p_x__) / sqrt(pow(p_x_ - p_x__, 2) + pow(p_y_ - p_y__, 2));
                        unit_lane_vector[1] = (p_y_ - p_y__) / sqrt(pow(p_x_ - p_x__, 2) + pow(p_y_ - p_y__, 2));
                        if (lane_idx != last_lane_idx && last_lane_idx != 0 && sqrt(pow(p_x - p_x_, 2) + pow(p_y - p_y_, 2)) < 3) {
                            std::cout << "===============================================" << std::endl;
                            std::cout << (unit_ne_vector[0] * (lane_data_sorted[last_lane_idx].fx - lane_data_sorted[last_lane_idx].bx) + \
                                          unit_ne_vector[1] * (lane_data_sorted[last_lane_idx].fy - lane_data_sorted[last_lane_idx].by)) / \
                                         sqrt(pow(lane_data_sorted[last_lane_idx].fx - lane_data_sorted[last_lane_idx].bx, 2) + \
                                              pow(lane_data_sorted[last_lane_idx].fy - lane_data_sorted[last_lane_idx].by, 2)) << std::endl;
                            if ((lane_data_sorted[last_lane_idx].fx - lane_data_sorted[last_lane_idx].bx) * \
                                (present_car_position[0] - lane_data_sorted[last_lane_idx].fx) + \
                                (lane_data_sorted[last_lane_idx].fy - lane_data_sorted[last_lane_idx].by) * \
                                (present_car_position[1] - lane_data_sorted[last_lane_idx].fy) <= 0 && \
                                (lane_data_sorted[last_lane_idx].fx - lane_data_sorted[last_lane_idx].bx) * \
                                (present_car_position[0] - lane_data_sorted[last_lane_idx].bx) + \
                                (lane_data_sorted[last_lane_idx].fy - lane_data_sorted[last_lane_idx].by) * \
                                (present_car_position[1] - lane_data_sorted[last_lane_idx].by) >= 0 && \
                                (unit_ne_vector[0] * (lane_points_coords.find( last_seg_lane_idx )->second.fx - \
                                 lane_points_coords.find( last_seg_lane_idx )->second.bx) + \
                                 unit_ne_vector[1] * (lane_points_coords.find( last_seg_lane_idx )->second.fy - \
                                 lane_points_coords.find( last_seg_lane_idx )->second.by)) / \
                                sqrt(pow(lane_points_coords.find( last_seg_lane_idx )->second.fx - \
                                         lane_points_coords.find( last_seg_lane_idx )->second.bx, 2) + \
                                     pow(lane_points_coords.find( last_seg_lane_idx )->second.fy - \
                                         lane_points_coords.find( last_seg_lane_idx )->second.by, 2)) > 0.965) {
                                int a = lane_data_sorted[last_lane_idx].blnid;
                                int b = lane_data_sorted[last_lane_idx].flnid;
                                for (std::map<int, Points_Coords>::iterator it = lane_points_coords.find(a); it != lane_points_coords.find(b); ++it) {
                                    float p1_x = it->second.bx, p1_y = it->second.by;
                                    float p2_x = it->second.fx, p2_y = it->second.fy;
                                    float v1_x = present_car_position[0] - p1_x, v1_y = present_car_position[1] - p1_y;
                                    float v2_x = present_car_position[0] - p2_x, v2_y = present_car_position[1] - p2_y;
                                    if ((p2_x - p1_x) * v1_x + (p2_y - p1_y) * v1_y < 0) {
                                        distances.push_back( sqrt(pow(v1_x, 2) + pow(v1_y, 2)) );
                                        distance_idx.push_back( it->first );
                                    } else if ((p2_x - p1_x) * v2_x + (p2_y - p1_y) * v2_y > 0) {
                                        distances.push_back( sqrt(pow(v2_x, 2) + pow(v2_y, 2)) );
                                        distance_idx.push_back( it->first );
                                    } else {
                                        std::vector<float> Line;
                                        Line.push_back(p2_y - p1_y);
                                        Line.push_back(-1 * (p2_x - p1_x));
                                        Line.push_back(-1 * (p2_y - p1_y) * p1_x - (-1 * (p2_x - p1_x)) * p1_y);
                                        distances.push_back(Distance(present_car_position, Line));
                                        distance_idx.push_back( it->first );
                                        Line.clear();
                                    }
                                }
                                seg_lane_idx = distance_idx[std::min_element(distances.begin(), distances.end()) - distances.begin()];
                                lane_idx = last_lane_idx;
                                distances.clear();
                                distance_idx.clear();
                            } else {
                                std::vector<float> dot;
                                std::vector<int> dot_idx;
                                dot_idx.push_back(lane_data_sorted[last_lane_idx].flid1);
                                dot_idx.push_back(lane_data_sorted[last_lane_idx].flid2);
                                dot_idx.push_back(lane_data_sorted[last_lane_idx].flid3);
                                dot_idx.push_back(lane_data_sorted[last_lane_idx].flid4);
                                std::map<int, Points_Coords>::iterator it1;
                                std::map<int, Points_Coords>::iterator it2;
                                int l;
                                for (int i = 0; i < dot_idx.size(); ++i) {
                                    if (dot_idx[i] != 0) {
                                        for (int j = 0; j < lane_data_sorted.size(); ++j) {
                                            if (dot_idx[i] == lane_data_sorted[j].blnid) {
                                                //std::cout << "Lane Length: " << lane_data_sorted[j].flnid - lane_data_sorted[j].blnid + 1 << std::endl;
                                                l = (lane_data_sorted[j].flnid - lane_data_sorted[j].blnid + 1 < 20) ? \
                                                     lane_data_sorted[j].flnid - lane_data_sorted[j].blnid + 1 : 20;
                                            }
                                        }
                                        it1 = lane_points_coords.find( dot_idx[i] );
                                        it2 = lane_points_coords.find( dot_idx[i] + l - 1 );
                                        dot.push_back((unit_ne_vector[0] * (it2->second.fx - it1->second.bx) + \
                                                       unit_ne_vector[1] * (it2->second.fy - it1->second.by)) / \
                                                      sqrt(pow(it2->second.fx - it1->second.bx, 2) + pow(it2->second.fy - it1->second.by, 2)));
                                        //std::cout << "Inner Product: " << dot[i] << std::endl;
                                    }
                                }
                                for (int i = 0; i < lane_data_sorted.size(); ++i) {
                                    if (lane_data_sorted[i].blnid == dot_idx[std::max_element(dot.begin(), dot.end()) - dot.begin()] && \
                                        dot[std::max_element(dot.begin(), dot.end()) - dot.begin()] > 0.707) {
                                        lane_idx = i;
                                    }
                                }
                                int a = lane_data_sorted[lane_idx].blnid;
                                int b = lane_data_sorted[lane_idx].flnid;
                                for (std::map<int, Points_Coords>::iterator it = lane_points_coords.find(a); it != lane_points_coords.find(b); ++it) {
                                    float p1_x = it->second.bx, p1_y = it->second.by;
                                    float p2_x = it->second.fx, p2_y = it->second.fy;
                                    float v1_x = present_car_position[0] - p1_x, v1_y = present_car_position[1] - p1_y;
                                    float v2_x = present_car_position[0] - p2_x, v2_y = present_car_position[1] - p2_y;
                                    if ((p2_x - p1_x) * v1_x + (p2_y - p1_y) * v1_y < 0) {
                                        distances.push_back( sqrt(pow(v1_x, 2) + pow(v1_y, 2)) );
                                        distance_idx.push_back( it->first );
                                    } else if ((p2_x - p1_x) * v2_x + (p2_y - p1_y) * v2_y > 0) {
                                        distances.push_back( sqrt(pow(v2_x, 2) + pow(v2_y, 2)) );
                                        distance_idx.push_back( it->first );
                                    } else {
                                        std::vector<float> Line;
                                        Line.push_back(p2_y - p1_y);
                                        Line.push_back(-1 * (p2_x - p1_x));
                                        Line.push_back(-1 * (p2_y - p1_y) * p1_x - (-1 * (p2_x - p1_x)) * p1_y);
                                        distances.push_back(Distance(present_car_position, Line));
                                        distance_idx.push_back( it->first );
                                        Line.clear();
                                    }
                                }
                                //std::cout << "===============================================" << std::endl;
                                seg_lane_idx = distance_idx[std::min_element(distances.begin(), distances.end()) - distances.begin()];
                                
                                dot.clear();
                                dot_idx.clear();
                                distances.clear();
                                distance_idx.clear();
                            }
                        } else if (sqrt(pow(p_x - p_x_, 2) + pow(p_y - p_y_, 2)) >= 3 && last_lane_idx != 0 && lane_idx != 142) {
                            std::cout << "###############################################" << std::endl;
                            genji_lane_position[0] = p_x_ + V * unit_lane_vector[0] * duration;
                            genji_lane_position[1] = p_y_ + V * unit_lane_vector[1] * duration;
                            for (int k = lane_data_sorted[last_lane_idx].blnid; k <= lane_data_sorted[last_lane_idx].flnid; ++k) {
                                float p1_x = lane_points_coords.find( k )->second.bx, p1_y = lane_points_coords.find( k )->second.by;
                                float p2_x = lane_points_coords.find( k )->second.fx, p2_y = lane_points_coords.find( k )->second.fy;
                                float v1_x = genji_lane_position[0] - p1_x, v1_y = genji_lane_position[1] - p1_y;
                                float v2_x = genji_lane_position[0] - p2_x, v2_y = genji_lane_position[1] - p2_y;
                                if ((p2_x - p1_x) * v1_x + (p2_y - p1_y) * v1_y < 0) {
                                    distances.push_back( sqrt(pow(v1_x, 2) + pow(v1_y, 2)) );
                                    distance_idx.push_back( lane_points_coords.find( k )->first );
                                } else if ((p2_x - p1_x) * v2_x + (p2_y - p1_y) * v2_y > 0) {
                                    distances.push_back( sqrt(pow(v2_x, 2) + pow(v2_y, 2)) );
                                    distance_idx.push_back( lane_points_coords.find( k )->first );
                                } else {
                                    std::vector<float> Line;
                                    Line.push_back(p2_y - p1_y);
                                    Line.push_back(-1 * (p2_x - p1_x));
                                    Line.push_back(-1 * (p2_y - p1_y) * p1_x - (-1 * (p2_x - p1_x)) * p1_y);
                                    distances.push_back(Distance(genji_lane_position, Line));
                                    distance_idx.push_back( lane_points_coords.find( k )->first );
                                    Line.clear();
                                }
                                distances.push_back(sqrt(pow(genji_lane_position[0] - p_x, 2) + pow(genji_lane_position[1] - p_y, 2)));
                                distance_idx.push_back(k);
                            }
                            seg_lane_idx = distance_idx[std::min_element(distances.begin(), distances.end()) - distances.begin()];
                            lane_idx = last_lane_idx;
                            
                            distances.clear();
                            distance_idx.clear();
                        }
                        
                        if (seg_lane_idx == last_seg_lane_idx) {
                            repeat++;
                        }
                    } else {
                        repeat = 0;
                    }
                    
                    std::cout << "Segment lane index: " << seg_lane_idx << std::endl;
                    std::cout << "Last Lane index: " << last_lane_idx << std::endl;
                    std::cout << "Lane index: " << lane_idx << std::endl;
                    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
                    
                    //---------------------------------- Save data for drawing
                    std::vector<float> a_lane;
                    a_lane.push_back(0.5 * (lane_points_coords.find( seg_lane_idx )->second.bx + lane_points_coords.find( seg_lane_idx )->second.fx));
                    a_lane.push_back(0.5 * (lane_points_coords.find( seg_lane_idx )->second.by + lane_points_coords.find( seg_lane_idx )->second.fy));
                    a_lane.push_back(0.5 * (lane_points_coords.find( seg_lane_idx )->second.bh + lane_points_coords.find( seg_lane_idx )->second.fh));
                    lane_csv.push_back(a_lane);
                    //printf("%f, %f, %f\n", a_lane[0], a_lane[1], a_lane[2]);
                    a_lane.clear();
                    
                    selected_lanes.push_back(Lane_Map.find(seg_lane_idx)->second);
                    lane_point_in_map.header.frame_id = pos->header.frame_id;
                    lane_point_in_map.header.stamp = pos->header.stamp;
                    lane_point_in_map.pose.position.x = 0.5 * (lane_points_coords.find( seg_lane_idx )->second.bx + lane_points_coords.find( seg_lane_idx )->second.fx);
                    lane_point_in_map.pose.position.y = 0.5 * (lane_points_coords.find( seg_lane_idx )->second.by + lane_points_coords.find( seg_lane_idx )->second.fy);
                    lane_point_in_map.pose.position.z = 0.5 * (lane_points_coords.find( seg_lane_idx )->second.bh + lane_points_coords.find( seg_lane_idx )->second.fh);
                    lane_point_in_map.pose.orientation.x = 0;
                    lane_point_in_map.pose.orientation.y = 0;
                    lane_point_in_map.pose.orientation.z = 0;
                    lane_point_in_map.pose.orientation.w = 1;
                    
                    
                    //---------------------------------- Transform Lane for Visualization
                    /*
                    for (int k = lane_data_sorted[lane_idx].blnid; k <= lane_data_sorted[lane_idx].flnid; ++k) {
                        selected_lanes.push_back(Lane_Map.find(k)->second);
                    }
                    
                    points_to_transform.resize(selected_lanes.size());
                    for (int n = 0; n < selected_lanes.size(); ++n) {
                        points_to_transform[n].resize(2);
                        points_to_transform[n][0].push_back( (lane_points_coords.find(selected_lanes[n].lnid))->second.bx );
                        points_to_transform[n][0].push_back( (lane_points_coords.find(selected_lanes[n].lnid))->second.by );
                        points_to_transform[n][0].push_back( (lane_points_coords.find(selected_lanes[n].lnid))->second.bh );
                        points_to_transform[n][1].push_back( (lane_points_coords.find(selected_lanes[n].lnid))->second.fx );
                        points_to_transform[n][1].push_back( (lane_points_coords.find(selected_lanes[n].lnid))->second.fy );
                        points_to_transform[n][1].push_back( (lane_points_coords.find(selected_lanes[n].lnid))->second.fh );
                    }
                    */
                    //---------------------------------- White Line Segments Selection
                    
                    std::vector<float> r_distances;
                    std::vector<int> r_distance_idx;
                    std::vector<float> l_distances;
                    std::vector<int> l_distance_idx;
                    float h_a = lane_points_coords.find( seg_lane_idx )->second.fx - lane_points_coords.find( seg_lane_idx )->second.bx;
                    float h_b = lane_points_coords.find( seg_lane_idx )->second.fy - lane_points_coords.find( seg_lane_idx )->second.by;
                    float lane_x = 0.5 * (lane_points_coords.find( seg_lane_idx )->second.bx + lane_points_coords.find( seg_lane_idx )->second.fx);
                    float lane_y = 0.5 * (lane_points_coords.find( seg_lane_idx )->second.by + lane_points_coords.find( seg_lane_idx )->second.fy);
                    Horiz_Line.push_back(h_a);
                    Horiz_Line.push_back(h_b);
                    Horiz_Line.push_back(-1 * h_a * lane_x - 1 * h_b * lane_y);
                    for (std::map<int, Points_Coords>::iterator it = whiteline_points_coords.begin(); it != whiteline_points_coords.end(); ++it) {
                        float *point = new float[2];
                        point[0] = 0.5 * (it->second.fx + it->second.bx);
                        point[1] = 0.5 * (it->second.fy + it->second.by);
                        if (Distance(point, Horiz_Line) <= 1 && \
                            (h_a * (it->second.fx - it->second.bx) + \
                            h_b * (it->second.fy - it->second.by)) / \
                            (sqrt(pow(h_a, 2) + pow(h_b, 2)) * \
                            sqrt(pow(it->second.fx - it->second.bx, 2) + pow(it->second.fy - it->second.by, 2))) > \
                            cos(5 * PI / 180)) {

                            float p1_x = it->second.bx, p1_y = it->second.by;
                            float p2_x = it->second.fx, p2_y = it->second.fy;
                            float v1_x = lane_x - p1_x, v1_y = lane_y - p1_y;
                            float v2_x = lane_x - p2_x, v2_y = lane_y - p2_y;
                            if ((p2_x - p1_x) * v1_x + (p2_y - p1_y) * v1_y < 0) {
                                if (h_b * (0.5 * (p2_x + p1_x) - lane_x) - \
                                    h_a * (0.5 * (p2_y + p1_y) - lane_y) > 0) {
                                
                                    r_distances.push_back( sqrt(pow(v1_x, 2) + pow(v1_y, 2)) );
                                    r_distance_idx.push_back( it->first );
                                } else {
                                    l_distances.push_back( sqrt(pow(v1_x, 2) + pow(v1_y, 2)) );
                                    l_distance_idx.push_back( it->first );
                                }
                            } else if ((p2_x - p1_x) * v2_x + (p2_y - p1_y) * v2_y > 0) {
                                if (h_b * (0.5 * (p2_x + p1_x) - lane_x) - \
                                    h_a * (0.5 * (p2_y + p1_y) - lane_y) > 0) {

                                    r_distances.push_back( sqrt(pow(v1_x, 2) + pow(v1_y, 2)) );
                                    r_distance_idx.push_back( it->first );
                                } else {
                                    l_distances.push_back( sqrt(pow(v1_x, 2) + pow(v1_y, 2)) );
                                    l_distance_idx.push_back( it->first );
                                }
                            } else {
                                float *vehicle = new float[2];
                                vehicle[0] = lane_x;
                                vehicle[1] = lane_y;
                                std::vector<float> Line;
                                Line.push_back(p2_y - p1_y);
                                Line.push_back(-1 * (p2_x - p1_x));
                                Line.push_back(-1 * (p2_y - p1_y) * p1_x - (-1 * (p2_x - p1_x)) * p1_y);
                                if (h_b * (0.5 * (p2_x + p1_x) - lane_x) - \
                                    h_a * (0.5 * (p2_y + p1_y) - lane_y) > 0) {

                                    r_distances.push_back(Distance(vehicle, Line));
                                    r_distance_idx.push_back( it->first );
                                } else {
                                    l_distances.push_back(Distance(vehicle, Line));
                                    l_distance_idx.push_back( it->first );
                                }
                                delete [] vehicle;
                                Line.clear();
                            }
                        }
                        delete [] point;
                    }
                    /*
                    //---------------------------------- White Lines Selection
                    int blnid = lane_data_sorted[lane_idx].blnid;
                    int flnid = lane_data_sorted[lane_idx].flnid;
                    std::vector<float> r_distances;
                    std::vector<int> r_distance_idx;
                    std::vector<float> l_distances;
                    std::vector<int> l_distance_idx;
                    for (int i = 0; i < whiteline_data_sorted.size(); ++i) {
                        
                        //if ( (whiteline_data_sorted[i].blnid - blnid) * (flnid - whiteline_data_sorted[i].blnid) >= 0 ||
                        //     (whiteline_data_sorted[i].flnid - blnid) * (flnid - whiteline_data_sorted[i].flnid) >= 0 ) {
                        if ((whiteline_data_sorted[i].blnid >= blnid && whiteline_data_sorted[i].blnid <= flnid) || \
                            (whiteline_data_sorted[i].flnid >= blnid && whiteline_data_sorted[i].flnid <= flnid)) {

                            for (int j = whiteline_data_sorted[i].bwlid; j <= whiteline_data_sorted[i].fwlid; ++j) {
                                selected_whitelines.push_back(WhiteLine_Map.find(j)->second);

                                
                                float p1_x = whiteline_points_coords.find(j)->second.bx, p1_y = whiteline_points_coords.find(j)->second.by;
                                float p2_x = whiteline_points_coords.find(j)->second.fx, p2_y = whiteline_points_coords.find(j)->second.fy;
                                float v1_x = present_car_position[0] - p1_x, v1_y = present_car_position[1] - p1_y;
                                float v2_x = present_car_position[0] - p2_x, v2_y = present_car_position[1] - p2_y;
                                if ((p2_x - p1_x) * v1_x + (p2_y - p1_y) * v1_y < 0) {
                                    if (ne_vector[1] * (0.5 * (p2_x + p1_x) - present_car_position[0]) - \
                                        ne_vector[0] * (0.5 * (p2_y + p1_y) - present_car_position[1]) > 0) {
                                    
                                        r_distances.push_back( sqrt(pow(v1_x, 2) + pow(v1_y, 2)) );
                                        r_distance_idx.push_back( whiteline_points_coords.find(j)->first );
                                    } else {
                                        l_distances.push_back( sqrt(pow(v1_x, 2) + pow(v1_y, 2)) );
                                        l_distance_idx.push_back( whiteline_points_coords.find(j)->first );
                                    }
                                } else if ((p2_x - p1_x) * v2_x + (p2_y - p1_y) * v2_y > 0) {
                                    if (ne_vector[1] * (0.5 * (p2_x + p1_x) - present_car_position[0]) - \
                                        ne_vector[0] * (0.5 * (p2_y + p1_y) - present_car_position[1]) > 0) {

                                        r_distances.push_back( sqrt(pow(v1_x, 2) + pow(v1_y, 2)) );
                                        r_distance_idx.push_back( whiteline_points_coords.find(j)->first );
                                    } else {
                                        l_distances.push_back( sqrt(pow(v1_x, 2) + pow(v1_y, 2)) );
                                        l_distance_idx.push_back( whiteline_points_coords.find(j)->first );
                                    }
                                } else {
                                    std::vector<float> Line;
                                    Line.push_back(p2_y - p1_y);
                                    Line.push_back(-1 * (p2_x - p1_x));
                                    Line.push_back(-1 * (p2_y - p1_y) * p1_x - (-1 * (p2_x - p1_x)) * p1_y);
                                    if (ne_vector[1] * (0.5 * (p2_x + p1_x) - present_car_position[0]) - \
                                        ne_vector[0] * (0.5 * (p2_y + p1_y) - present_car_position[1]) > 0) {

                                        r_distances.push_back(Distance(present_car_position, Line));
                                        r_distance_idx.push_back( whiteline_points_coords.find(j)->first );
                                    } else {
                                        l_distances.push_back(Distance(present_car_position, Line));
                                        l_distance_idx.push_back( whiteline_points_coords.find(j)->first );
                                    }
                                    Line.clear();
                                }
                                
                            }
                        }
                    }
                    std::cout << r_distance_idx.size() << std::endl;
                    std::cout << l_distance_idx.size() << std::endl;
                    */
                    if (r_distance_idx.size() > 0) {
                        r_whiteline_idx = r_distance_idx[std::min_element(r_distances.begin(), r_distances.end()) - r_distances.begin()];
                    } else {
                        r_whiteline_idx = last_r_whiteline_idx;
                    }
                    if (l_distance_idx.size() > 0) {
                        l_whiteline_idx = l_distance_idx[std::min_element(l_distances.begin(), l_distances.end()) - l_distances.begin()];
                    } else {
                        l_whiteline_idx = last_l_whiteline_idx;
                    }

                    points_for_whitelines.resize(2);
                    points_for_whitelines[0].resize(2);
                    points_for_whitelines[1].resize(2);
                    for (int n = 0; n < points_for_whitelines.size(); ++n) {
                        int s = (n == 0) ? r_whiteline_idx : l_whiteline_idx;
                        points_for_whitelines[n][0].push_back(whiteline_points_coords.find(s)->second.bx);
                        points_for_whitelines[n][0].push_back(whiteline_points_coords.find(s)->second.by);
                        points_for_whitelines[n][0].push_back(whiteline_points_coords.find(s)->second.bh);
                        points_for_whitelines[n][1].push_back(whiteline_points_coords.find(s)->second.fx);
                        points_for_whitelines[n][1].push_back(whiteline_points_coords.find(s)->second.fy);
                        points_for_whitelines[n][1].push_back(whiteline_points_coords.find(s)->second.fh);
                    }
                    
                    //---------------------------------- Save data for drawing
                    std::vector<float> a_whiteline;
                    a_whiteline.push_back(points_for_whitelines[0][0][0]);
                    a_whiteline.push_back(points_for_whitelines[0][0][1]);
                    a_whiteline.push_back(points_for_whitelines[0][0][2]);
                    a_whiteline.push_back(points_for_whitelines[0][1][0]);
                    a_whiteline.push_back(points_for_whitelines[0][1][1]);
                    a_whiteline.push_back(points_for_whitelines[0][1][2]);
                    whiteline_csv.push_back(a_whiteline);
                    a_whiteline.clear();
                    a_whiteline.push_back(points_for_whitelines[1][0][0]);
                    a_whiteline.push_back(points_for_whitelines[1][0][1]);
                    a_whiteline.push_back(points_for_whitelines[1][0][2]);
                    a_whiteline.push_back(points_for_whitelines[1][1][0]);
                    a_whiteline.push_back(points_for_whitelines[1][1][1]);
                    a_whiteline.push_back(points_for_whitelines[1][1][2]);
                    whiteline_csv.push_back(a_whiteline);
                    a_whiteline.clear();
                    
                    Horiz_Line.clear();
                    r_distances.clear();
                    r_distance_idx.clear();
                    l_distances.clear();
                    l_distance_idx.clear();
                    
                    //---------------------------------- Transform White Lines for Visualization
                    /*
                    points_to_transform.resize(selected_whitelines.size());
                    for (int n = 0; n < selected_whitelines.size(); ++n) {
                        points_to_transform[n].resize(2);
                        points_to_transform[n][0].push_back( whiteline_points_coords.find(selected_whitelines[n].id)->second.bx );
                        points_to_transform[n][0].push_back( whiteline_points_coords.find(selected_whitelines[n].id)->second.by );
                        points_to_transform[n][0].push_back( whiteline_points_coords.find(selected_whitelines[n].id)->second.bh );
                        points_to_transform[n][1].push_back( whiteline_points_coords.find(selected_whitelines[n].id)->second.fx );
                        points_to_transform[n][1].push_back( whiteline_points_coords.find(selected_whitelines[n].id)->second.fy );
                        points_to_transform[n][1].push_back( whiteline_points_coords.find(selected_whitelines[n].id)->second.fh );
                    }
                    */
                    
                    last_last_car_position[0] = last_car_position[0];
                    last_last_car_position[1] = last_car_position[1];
                    last_last_car_position[2] = last_car_position[2];
                    last_car_position[0] = present_car_position[0];
                    last_car_position[1] = present_car_position[1];
                    last_car_position[2] = present_car_position[2];
                    last_ne_vector[0] = ne_vector[0];
                    last_ne_vector[1] = ne_vector[1];
                    last_last_seg_lane_idx = last_seg_lane_idx;
                    last_seg_lane_idx = seg_lane_idx;
                    last_lane_idx = lane_idx;
                    last_r_whiteline_idx = r_whiteline_idx;
                    last_l_whiteline_idx = l_whiteline_idx;

                    if (points_for_whitelines.size() > 0) {
                        // Save data to world_points_array
                        world_points_array.header.frame_id = pos->header.frame_id;
                        world_points_array.header.stamp = pos->header.stamp;
                        world_points_array.layout.dim[0].label = "lane";
                        world_points_array.layout.dim[1].label = "point";
                        world_points_array.layout.dim[2].label = "coord";
                        world_points_array.layout.dim[0].size = points_for_whitelines.size();
                        world_points_array.layout.dim[1].size = points_for_whitelines[0].size(); // 2 points
                        world_points_array.layout.dim[2].size = points_for_whitelines[0][0].size(); // 3 coordinates in world
                        world_points_array.layout.dim[0].stride = points_for_whitelines.size() * points_for_whitelines[0].size() * points_for_whitelines[0][0].size();
                        world_points_array.layout.dim[1].stride = points_for_whitelines[0].size() * points_for_whitelines[0][0].size();
                        world_points_array.layout.dim[2].stride = points_for_whitelines[0][0].size();
                        world_points_array.data.clear();
                        world_points_array.data.resize(points_for_whitelines.size() * points_for_whitelines[0].size() * points_for_whitelines[0][0].size());

                        for (int o = 0; o < points_for_whitelines.size(); ++o) {
                            for (int oo = 0; oo < points_for_whitelines[o].size(); ++oo) {
                                if (points_for_whitelines[o][oo].size() > 0) {
                                    world_points_array.data[o * 2 * 3 + oo * 3 + 0] = points_for_whitelines[o][oo][0];
                                    world_points_array.data[o * 2 * 3 + oo * 3 + 1] = points_for_whitelines[o][oo][1];
                                    world_points_array.data[o * 2 * 3 + oo * 3 + 2] = points_for_whitelines[o][oo][2];
                                }
                            }
                        }
                    }
                    
                    image_lane.publish(cv_image->toImageMsg());
                    //fuse_image_lane.publish(cv_laneNet_image->toImageMsg());
                    world_line_points.publish(world_points_array);// white lines coordinates in world frame
                    image_line_points.publish(image_points_array);// white lines coordinates in image frame
                    world_lane_points.publish(lane_point_in_map);// lanes coordinates in world frame
                }
            }
            V = sr->speed;

            toDelete.clear();
            selected_points.clear();
            selected_nodes.clear();
            selected_lanes.clear();
            selected_nodes_of_lanes.clear();
            selected_points_of_nodes.clear();
            points_to_transform.clear();
            points_for_whitelines.clear();
            selected_lines.clear();
            selected_whitelines.clear();
            selected_lines_of_whitelines.clear();
            selected_points_of_lines.clear();
            next_lanes.clear();

            world_points_array.data.clear();
            image_points_array.data.clear();

            

        } catch (tf::TransformException &ex) {
            ROS_ERROR("%s", ex.what());
            ros::Duration(1.0).sleep();
            //continue;
        }
    }
    
    void read_sorted_lanes() {
        std::fstream file;
        //file.open("/home/mec/Autoware/ros/src/computing/perception/detection/vision_detector/packages/vision_lane_detect/script/HDmap_sort/lane_df.csv");
        file.open("/home/meclab/autoware.ai/src/autoware/core_perception/vision_lane_detect/script/HDmap_sort/lane_df.csv"); //YC
        std::string line;
        int col = 25;
        int i = 0;
        while(getline(file, line)) {
            if (i > 0) {
                std::istringstream templine(line); // string 轉換成 stream
                std::string data;
                Lane_Data_Sorted L;
                int j = 0;
                while (getline( templine, data, ',')) {
                    //std::cout << atof(data.c_str()) << std::endl;
                    if (j == 0) {
                        L.blnid = atoi(data.c_str());
                    } else if (j == 1) {
                        L.flnid = atoi(data.c_str());
                    } else if (j == 2) {
                        L.bdid = atoi(data.c_str());
                    } else if (j == 3) {
                        L.fdid = atoi(data.c_str());
                    } else if (j == 4) {
                        L.bpid = atoi(data.c_str());
                    } else if (j == 5) {
                        L.fpid = atoi(data.c_str());
                    } else if (j == 6) {
                        L.bx = atof(data.c_str());
                    } else if (j == 7) {
                        L.by = atof(data.c_str());
                    } else if (j == 8) {
                        L.bh = atof(data.c_str());
                    } else if (j == 9) {
                        L.fx = atof(data.c_str());
                    } else if (j == 10) {
                        L.fy = atof(data.c_str());
                    } else if (j == 11) {
                        L.fh = atof(data.c_str());
                    } else if (j == 12) {
                        L.bx_1 = atof(data.c_str());
                    } else if (j == 13) {
                        L.by_1 = atof(data.c_str());
                    } else if (j == 14) {
                        L.bh_1 = atof(data.c_str());
                    } else if (j == 15) {
                        L.lcnt = atoi(data.c_str());
                    } else if (j == 16) {
                        L.lno = atoi(data.c_str());
                    } else if (j == 17) {
                        L.lanetype = atoi(data.c_str());
                    } else if (j == 18) {
                        L.limitvel = atoi(data.c_str());
                    } else if (j == 19) {
                        L.lanecfgfg = atoi(data.c_str());
                    } else if (j == 20) {
                        L.blid1 = atoi(data.c_str());
                    } else if (j == 21) {
                        L.blid2 = atoi(data.c_str());
                    } else if (j == 22) {
                        L.blid3 = atoi(data.c_str());
                    } else if (j == 23) {
                        L.blid4 = atoi(data.c_str());
                    } else if (j == 24) {
                        L.flid1 = atoi(data.c_str());
                    } else if (j == 25) {
                        L.flid2 = atoi(data.c_str());
                    } else if (j == 26) {
                        L.flid3 = atoi(data.c_str());
                    } else if (j == 27) {
                        L.flid4 = atoi(data.c_str());
                    }
                    j++;
                }
                lane_data_sorted.push_back(L);
                //std::cout << lane_data_sorted[0].by << std::endl;
                //printf("%lf\n", lane_data_sorted[0].by);
                //break;
            }
            i++;
        }
        file.close();
        std::cout << "Lanes Reading Finished!" << std::endl;
    }
    void read_sorted_whitelines() {
        std::fstream file;
        //file.open("/home/mec/Autoware/ros/src/computing/perception/detection/vision_detector/packages/vision_lane_detect/script/HDmap_sort/whiteline_df.csv");
        file.open("/home/meclab/autoware.ai/src/autoware/core_perception/vision_lane_detect/script/HDmap_sort/whiteline_df.csv"); //YC
        std::string line;
        int col = 17;
        int i = 0;
        while(getline(file, line)) {
            if (i > 0) {
                std::istringstream templine(line); // string 轉換成 stream
                std::string data;
                WhiteLine_Data_Sorted L;
                int j = 0;
                while (getline( templine, data, ',')) {
                    //std::cout << atof(data.c_str()) << std::endl;
                    if (j == 0) {
                        L.bwlid = atoi(data.c_str());
                    } else if (j == 1) {
                        L.fwlid = atoi(data.c_str());
                    } else if (j == 2) {
                        L.blid = atoi(data.c_str());
                    } else if (j == 3) {
                        L.flid = atoi(data.c_str());
                    } else if (j == 4) {
                        L.blnid = atoi(data.c_str());
                    } else if (j == 5) {
                        L.flnid = atoi(data.c_str());
                    } else if (j == 6) {
                        L.bpid = atoi(data.c_str());
                    } else if (j == 7) {
                        L.fpid = atoi(data.c_str());
                    } else if (j == 8) {
                        L.bx = atof(data.c_str());
                    } else if (j == 9) {
                        L.by = atof(data.c_str());
                    } else if (j == 10) {
                        L.bh = atof(data.c_str());
                    } else if (j == 11) {
                        L.fx = atof(data.c_str());
                    } else if (j == 12) {
                        L.fy = atof(data.c_str());
                    } else if (j == 13) {
                        L.fh = atof(data.c_str());
                    } else if (j == 14) {
                        L.width = atof(data.c_str());
                    } else if (j == 15) {
                        L.color = atoi(data.c_str());
                    } else if (j == 16) {
                        L.type = atoi(data.c_str());
                    }
                    j++;
                }
                whiteline_data_sorted.push_back(L);
                //std::cout << lane_data_sorted[0].by << std::endl;
                //printf("%lf\n", lane_data_sorted[0].by);
                //break;
            }
            i++;
        }
        file.close();
        std::cout << "WhiteLines Reading Finished!" << std::endl;
        //printf("%lf\n", whiteline_data_sorted[20].by);
    }
    
public:
    
    HDLanePipeline() {  // erase all vectors

    }

    ~HDLanePipeline() {
        imu_csv.clear();
        lane_csv.clear();
        whiteline_csv.clear();
        //---------------------------------- Clear Data
        //std::cout << "Clear!" << std::endl;
        Point_Map.clear();
        Line_Map.clear();
        Node_Map.clear();
        Lane_Map.clear();
        WhiteLine_Map.clear();
        selected_points.clear();
        selected_nodes.clear();
        selected_lanes.clear();

        lane_data_sorted.clear();
        whiteline_data_sorted.clear();

        lane_points_coords.clear();
        whiteline_points_coords.clear();
    }
    
    void Run() {
        std::string v_map_point_topic_name = "/vector_map_info/point";
        std::string v_map_line_topic_name = "/vector_map_info/line";
        std::string v_map_node_topic_name = "/vector_map_info/node";
        std::string v_map_lane_topic_name = "/vector_map_info/lane";
        std::string v_map_whiteline_topic_name = "/vector_map_info/white_line";

        //std::string image_topic_name = "/image_raw";
        std::string image_topic_name = "/gmsl_camera/port_0/cam_0/image_raw";
        //std::string car_position_topic_name = "/ndt_pose";
        std::string car_position_topic_name = "/xsens/gps/pose";
        std::string imu_data_topic_name = "/xsens/imu/data";
        std::string steering_report_topic_name = "/vehicle/steering_report";
        //std::string laneNet_topic_name = "/laneNet_image";
        std::string laneNet_topic_name = "/erfNet_image";

        image_lane = n.advertise<Image>("HD_lane_topic", 1);// Image or autoware_msgs::ImageLaneObjects
        //fuse_image_lane = n.advertise<Image>("fuse_lane_topic", 1);// Image or autoware_msgs::ImageLaneObjects
        //world_line_points = n.advertise<std_msgs::Float32MultiArray>("world_line_points_topic", 1);
        //image_line_points = n.advertise<std_msgs::Int32MultiArray>("image_line_points_topic", 1);
        //world_lane_points = n.advertise<PoseStamped>("world_lane_points_topic", 1);

        // Custom Message
        world_line_points = n.advertise<vision_lane_detect::Float32MultiArray_H>("world_line_points_topic", 1);
        image_line_points = n.advertise<vision_lane_detect::Int32MultiArray_H>("image_line_points_topic", 1);
        world_lane_points = n.advertise<PoseStamped>("world_lane_points_topic", 1);

        world_points_array.layout.dim.push_back(std_msgs::MultiArrayDimension());
        world_points_array.layout.dim.push_back(std_msgs::MultiArrayDimension());
        world_points_array.layout.dim.push_back(std_msgs::MultiArrayDimension());
        image_points_array.layout.dim.push_back(std_msgs::MultiArrayDimension());
        image_points_array.layout.dim.push_back(std_msgs::MultiArrayDimension());
        image_points_array.layout.dim.push_back(std_msgs::MultiArrayDimension());

        ros::Subscriber point_sub = n.subscribe(v_map_point_topic_name, 1000, &HDLanePipeline::point_Callback, this);
        ros::Subscriber line_sub = n.subscribe(v_map_line_topic_name, 1000, &HDLanePipeline::line_Callback, this);
        ros::Subscriber node_sub = n.subscribe(v_map_node_topic_name, 1000, &HDLanePipeline::node_Callback, this);
        ros::Subscriber lane_sub = n.subscribe(v_map_lane_topic_name, 1000, &HDLanePipeline::lane_Callback, this);
        ros::Subscriber whiteline_sub = n.subscribe(v_map_whiteline_topic_name, 1000, &HDLanePipeline::whiteline_Callback, this);
        
        read_sorted_lanes();
        read_sorted_whitelines();
        
        // loading hd map and sorted hd map

        //---------------------------------- Time Synchronizer
        
        message_filters::Subscriber<Image> image_sub(n, image_topic_name, 1);
        message_filters::Subscriber<Image> laneNet_sub(n, laneNet_topic_name, 1);
        message_filters::Subscriber<PoseStamped> car_position_sub(n, car_position_topic_name, 1);
        message_filters::Subscriber<SteeringReport> steering_report_sub(n, steering_report_topic_name, 1);
        
        typedef sync_policies::ApproximateTime<Image, PoseStamped, SteeringReport> MySyncPolicy;
        // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
        Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image_sub, car_position_sub, steering_report_sub);
        sync.registerCallback(boost::bind(&HDLanePipeline::HDPipeline_Callback, this, _1, _2, _3));

        ros::spin();

    }
};

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "HD_lane");
    HDLanePipeline app;

    app.Run();

    return 0;

}