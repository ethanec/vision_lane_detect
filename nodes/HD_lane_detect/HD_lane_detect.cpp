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
#include <std_msgs/Int32MultiArray.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>
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
#include <vector_map_msgs/WhiteLineArray.h>

#define PI 3.14159265
#define GO 0.1

using namespace sensor_msgs;
using namespace geometry_msgs;
using namespace message_filters;

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
float last_car_position[3] = {0, 0, 0};
float present_car_position[3] = {0, 0, 0};
float ne_vector[3] = {0, 0, 0};
float unit_ne_vector[3] = {0, 0, 0};
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
float k1 = distCoeff[0];
float k2 = distCoeff[1];
float p1 = distCoeff[2];
float p2 = distCoeff[3];
float k3 = distCoeff[4];
float q_rad = 0;
float q_s = 0;
float q_w = 0;
float q_a = 0;
float q_b = 0;
float q_c = 0;
float map_to_body[12] = {0};

int lane_idx = 0, last_lane_idx = 0;
float closest_distance = 0;
int trigger = 0;



class HDLanePipeline {
private:
    ros::NodeHandle n;
    ros::Publisher image_lane;
    ros::Publisher fuse_image_lane;
    ros::Publisher world_line_points;
    ros::Publisher image_line_points;
    ros::Publisher world_lane_points;
    
    tf::TransformListener *listeners_ptr1, *listeners_ptr2;
    PoseStamped point_pose_in_map, point_pose_in_camera, point_pose_in_xsens, lane_point_in_map;

    std::vector<vector_map_msgs::Point> Point_Map;
    std::vector<vector_map_msgs::Line> Line_Map;
    std::vector<vector_map_msgs::Node> Node_Map;
    std::vector<vector_map_msgs::Lane> Lane_Map;
    std::vector<vector_map_msgs::WhiteLine> WhiteLine_Map;

    std::vector<vector_map_msgs::Point> selected_points;\
    std::vector<vector_map_msgs::Node> selected_nodes;
    std::vector<vector_map_msgs::Lane> selected_lanes;
    std::vector<vector_map_msgs::Line> selected_lines;
    std::vector<vector_map_msgs::WhiteLine> selected_whitelines;
    std::vector< std::vector<int> > selected_nodes_of_lanes;
    std::vector< std::vector<int> > selected_points_of_nodes;
    std::vector< std::vector< std::vector<float> > > points_to_transform;

    std::vector<int> selected_lines_of_whitelines;
    std::vector< std::vector<int> > selected_points_of_lines;

    std_msgs::Float32MultiArray world_points_array;
    std_msgs::Int32MultiArray image_points_array;
    
    struct Lane_Data {
        int LnID, BPID, FPID;
        
        float lane_vec[2];
        float cos_with_vehicle_heading, distance_to_vehicle;
    };
    struct Lane_Data_Sorted {
        int blnid, flnid, bdid, fdid, bpid, fpid, lcnt, lno, lanetype, limitvel, lanecfgfg;
        int blid1, blid2, blid3, blid4, flid1, flid2, flid3, flid4;
        double bx, by, bh, fx, fy, fh, bx_1, by_1, bh_1;
    };
    struct WhiteLine_Data_Sorted {
        int bwlid, fwlid, blid, flid, blnid, flnid, bpid, fpid, color, type;
        double bx, by, bh, fx, fy, fh;
        float width;
    };
    std::vector<Lane_Data> selected_lane_data;
    std::vector<Lane_Data_Sorted> lane_data_sorted;
    std::vector<WhiteLine_Data_Sorted> whiteline_data_sorted;
    std::vector<int> toDelete;
    std::vector<std::vector<std::vector<double> > > lane_points_coords;
    std::vector<std::vector<std::vector<double> > > whiteline_points_coords;
    
    void point_Callback(const vector_map_msgs::PointArray msg_array) {
        //std::cout << msg_array.data[0] << std::endl;
        Point_Map.resize(msg_array.data.size());
        Point_Map.assign(msg_array.data.begin(), msg_array.data.end());
        //std::cout << Point_Map[0] << std::endl;
    }

    void line_Callback(const vector_map_msgs::LineArray msg_array) {
        //std::cout << msg_array.data[0] << std::endl;
        Line_Map.resize(msg_array.data.size());
        Line_Map.assign(msg_array.data.begin(), msg_array.data.end());
    }

    void node_Callback(const vector_map_msgs::NodeArray msg_array) {
        //std::cout << msg_array.data[0] << std::endl;
        Node_Map.resize(msg_array.data.size());
        Node_Map.assign(msg_array.data.begin(), msg_array.data.end());
        //std::cout << Node_Map[0] << std::endl;
    }
    
    void lane_Callback(const vector_map_msgs::LaneArray msg_array) {
        //std::cout << msg_array.data[0] << std::endl;
        Lane_Map.resize(msg_array.data.size());
        Lane_Map.assign(msg_array.data.begin(), msg_array.data.end());
        /*
        std::cout << "Loading Lane Data..." << std::endl;
        lane_points_coords.resize(Lane_Map.size());
        for (int n = 0; n < Lane_Map.size(); ++n) {
            lane_points_coords[n].resize(2);
        }

        for (int i = 0; i < Lane_Map.size(); ++i) {
            for (int j = 0; j < Node_Map.size(); ++j) {
                if (Lane_Map[i].bnid == Node_Map[j].nid) {
                    for (int k = 0; k < Point_Map.size(); ++k) {
                        if (Node_Map[j].pid == Point_Map[k].pid) {
                            lane_points_coords[i][0].push_back(Point_Map[k].ly);
                            lane_points_coords[i][0].push_back(Point_Map[k].bx);
                            lane_points_coords[i][0].push_back(Point_Map[k].h);
                        }
                        if (lane_points_coords[i][0].size() == 3) {
                            break;
                        }
                    }
                }
                if (Lane_Map[i].fnid == Node_Map[j].nid) {
                    for (int k = 0; k < Point_Map.size(); ++k) {
                        if (Node_Map[j].pid == Point_Map[k].pid) {
                            lane_points_coords[i][1].push_back(Point_Map[k].ly);
                            lane_points_coords[i][1].push_back(Point_Map[k].bx);
                            lane_points_coords[i][1].push_back(Point_Map[k].h);
                        }
                        if (lane_points_coords[i][1].size() == 3) {
                            break;
                        }
                    }
                }
            }
        }
        std::cout << "Loading Finished!" << std::endl;
        */
    }
    
    void whiteline_Callback(const vector_map_msgs::WhiteLineArray msg_array) {
        WhiteLine_Map.resize(msg_array.data.size());
        WhiteLine_Map.assign(msg_array.data.begin(), msg_array.data.end());
        /*
        std::cout << "Loading Line Data..." << std::endl;
        whiteline_points_coords.resize(WhiteLine_Map.size());
        for (int n = 0; n < WhiteLine_Map.size(); ++n) {
            whiteline_points_coords[n].resize(2);
        }
        for (int i = 0; i < WhiteLine_Map.size(); ++i) {
            for (int j = 0; j < Line_Map.size(); ++j) {
                if (WhiteLine_Map[i].lid == Line_Map[j].lid) {
                    for (int k = 0; k < Point_Map.size(); ++k) {
                        
                        if (Line_Map[j].bpid == Point_Map[k].pid) {
                            whiteline_points_coords[i][0].push_back(Point_Map[k].ly);
                            whiteline_points_coords[i][0].push_back(Point_Map[k].bx);
                            whiteline_points_coords[i][0].push_back(Point_Map[k].h);
                        }
                        if (Line_Map[j].fpid == Point_Map[k].pid) {
                            whiteline_points_coords[i][1].push_back(Point_Map[k].ly);
                            whiteline_points_coords[i][1].push_back(Point_Map[k].bx);
                            whiteline_points_coords[i][1].push_back(Point_Map[k].h);
                        }
                        if (whiteline_points_coords[i][0].size() == 3 && whiteline_points_coords[i][1].size() == 3) {
                            break;
                        }
                    }
                }
            }
        }
        std::cout << "Loading Finished!" << std::endl;
        */
    }

    int x_Radial_Distortion(int x, int y) {
        int r = sqrt(pow(x, 2) + pow(y, 2));
        return int(x * (1 + k1 * pow(r, 2) + k2 * pow(r, 4) + k3 * pow(r, 6)));
    }

    int y_Radial_Distortion(int x, int y) {
        int r = sqrt(pow(x, 2) + pow(y, 2));
        return int(y * (1 + k1 * pow(r, 2) + k2 * pow(r, 4) + k3 * pow(r, 6)));
    }

    int x_Tangential_Distortion(int x, int y) {
        int r = sqrt(pow(x, 2) + pow(y, 2));
        return int(x + (2 * p1 * x * y + p2 * (pow(r, 2) + 2 * pow(x, 2))));
    }

    int y_Tangential_Distortion(int x, int y) {
        int r = sqrt(pow(x, 2) + pow(y, 2));
        return int(y + (2 * p2 * x * y + p1 * (pow(r, 2) + 2 * pow(y, 2))));
    }

    void lanes_selection(float *present_car_position) {
        //int min_distance = 0, min_lnid = 0
        for (int j = 0; j < selected_points.size(); ++j) {
            for (int jj = 0; jj < Node_Map.size(); ++jj) {
                if (Node_Map[jj].pid == selected_points[j].pid) {
                    selected_nodes.push_back(Node_Map[jj]);
                    break;
                }
            }
        }

        for (int k = 0; k < selected_nodes.size(); ++k) {
            for (int kk = 0; kk < Lane_Map.size(); ++kk) {
                if (Lane_Map[kk].bnid == selected_nodes[k].nid || Lane_Map[kk].fnid == selected_nodes[k].nid) {
                    selected_lanes.push_back(Lane_Map[kk]);
                    //break;
                }
            }
        }

        selected_nodes_of_lanes.resize(selected_lanes.size());
        for (int l = 0; l < selected_lanes.size(); ++l) {
            selected_nodes_of_lanes[l].push_back(selected_lanes[l].bnid);
            selected_nodes_of_lanes[l].push_back(selected_lanes[l].fnid);
        }

        selected_points_of_nodes.resize(selected_nodes_of_lanes.size());
        for (int m = 0; m < selected_nodes_of_lanes.size(); ++m) {
            selected_points_of_nodes[m].push_back(selected_nodes_of_lanes[m][0]);
            selected_points_of_nodes[m].push_back(selected_nodes_of_lanes[m][1]);
        }

        if (selected_points_of_nodes.size() > 0) {
            

            // Resize the points_to_transform
            
            points_to_transform.resize(selected_points_of_nodes.size());
            for (int n = 0; n < selected_points_of_nodes.size(); ++n) {
                points_to_transform[n].resize(selected_points_of_nodes[n].size());
            }
            
            for (int n = 0; n < selected_points_of_nodes.size(); ++n) {
                for (int nn = 0; nn < selected_points_of_nodes[n].size(); ++nn) {
                    for (int nnn = 0; nnn < Point_Map.size(); ++nnn) {
                        if (Point_Map[nnn].pid == selected_points_of_nodes[n][nn]) {
                            points_to_transform[n][nn].push_back(Point_Map[nnn].ly);
                            points_to_transform[n][nn].push_back(Point_Map[nnn].bx);
                            points_to_transform[n][nn].push_back(Point_Map[nnn].h);
                            //min_distance = () ? () : ();
                        } 
                    }
                }
            }
            
            //std::cout << ">>>>>>>>>>" << std::endl;
        }
    }
    
    void whitelines_selection() {
        for (int i = 0; i < selected_points.size(); ++i) {
            for (int ii = 0; ii < Line_Map.size(); ++ii) {
                if (Line_Map[ii].bpid == selected_points[i].pid || Line_Map[ii].fpid == selected_points[i].pid) {
                    selected_lines.push_back(Line_Map[ii]);
                    //break;
                }
            }
        }
        
        for (int j = 0; j < selected_lines.size(); ++j) {
            for (int jj = 0; jj < WhiteLine_Map.size(); ++jj) {
                if (WhiteLine_Map[jj].lid == selected_lines[j].lid) {
                    selected_whitelines.push_back(WhiteLine_Map[jj]);
                    //break;
                }
            }
        }
        
        //selected_lines_of_whitelines.resize(selected_whitelines.size());
        for (int k = 0; k < selected_whitelines.size(); ++k) {
            selected_lines_of_whitelines.push_back(selected_whitelines[k].lid);
            //std::cout << selected_whitelines[k].lid << std::endl;
        }
        
        selected_points_of_lines.resize(selected_lines_of_whitelines.size());
        for (int l = 0; l < selected_lines_of_whitelines.size(); ++l) {
            selected_points_of_lines[l].resize(2);
            for (int ll = 0; ll < selected_lines.size(); ++ll) {
                if (selected_lines[ll].lid == selected_lines_of_whitelines[l]) {
                    selected_points_of_lines[l][0] = selected_lines[ll].bpid;
                    selected_points_of_lines[l][1] = selected_lines[ll].fpid;
                }
            }
        }
        
        if (selected_points_of_lines.size() > 0) {

            // Resize the points_to_transform
            
            points_to_transform.resize(selected_points_of_lines.size());
            for (int m = 0; m < selected_points_of_lines.size(); ++m) {
                points_to_transform[m].resize(selected_points_of_lines[m].size());
            }
                
            for (int n = 0; n < selected_points_of_lines.size(); ++n) {
                for (int nn = 0; nn < selected_points_of_lines[n].size(); ++nn) {
                    for (int nnn = 0; nnn < Point_Map.size(); ++nnn) {
                        if (Point_Map[nnn].pid == selected_points_of_lines[n][nn]) {
                            points_to_transform[n][nn].push_back(Point_Map[nnn].ly);
                            points_to_transform[n][nn].push_back(Point_Map[nnn].bx);
                            points_to_transform[n][nn].push_back(Point_Map[nnn].h);
                        } 
                    }
                }
            }
        }
    }

    void ThinkTron_HD_Map(float *present_car_position, float *ne_vector, float *long_coefficients, float *lat_coefficients) {
        //---------------------------------- Extract the most fitted lane for sampling lines' data
        //std::cout << ">>>>>>>>>>" << std::endl;
        selected_lane_data.clear();
        toDelete.clear();
        for (int i = 0; i < Point_Map.size(); ++i) {
            cell_w = abs(Point_Map[i].ly * long_coefficients[0] + Point_Map[i].bx * long_coefficients[1] + long_coefficients[2]);
            cell_l = abs(Point_Map[i].ly * lat_coefficients[0] + Point_Map[i].bx * lat_coefficients[1] + lat_coefficients[2]);
            if (cell_w <= 5 && cell_l <= 1.5) {
                for (int j = 0; j < Lane_Map.size(); ++j) {
                    //if ((Point_Map[i].pid - 1 < Node_Map.size()) && (Lane_Map[j].bnid == Node_Map.at(Point_Map[i].pid - 1).nid || Lane_Map[j].fnid == Node_Map.at(Point_Map[i].pid - 1).nid)) {
                    if (Point_Map[i].pid - 1 < Node_Map.size() && Lane_Map[j].bnid == Node_Map.at(Point_Map[i].pid - 1).nid) {
                        //std::cout << Lane_Map[j].lnid << std::endl;
                        Lane_Data L;
                        L.LnID = Lane_Map[j].lnid;
                        L.BPID = Lane_Map[j].bnid;
                        L.FPID = Lane_Map[j].fnid;
                        selected_lane_data.push_back(L);
                        break;
                    }
                }
            }
        }
        //---------------------------------- Calculates the lane vector, its cos similarity with car's heading and its distance to the ego vehicle.
        for (int i = 0; i < selected_lane_data.size(); ++i) {
            selected_lane_data[i].lane_vec[0] = Point_Map.at(selected_lane_data[i].FPID - 1).ly - Point_Map.at(selected_lane_data[i].BPID - 1).ly;
            selected_lane_data[i].lane_vec[1] = Point_Map.at(selected_lane_data[i].FPID - 1).bx - Point_Map.at(selected_lane_data[i].BPID - 1).bx;
            selected_lane_data[i].cos_with_vehicle_heading = (ne_vector[0] * selected_lane_data[i].lane_vec[0] + ne_vector[1] * selected_lane_data[i].lane_vec[1]) \
                / (sqrt(pow(ne_vector[0], 2) + pow(ne_vector[1], 2)) * sqrt(pow(selected_lane_data[i].lane_vec[0], 2) + pow(selected_lane_data[i].lane_vec[1], 2)));
            selected_lane_data[i].distance_to_vehicle = abs(present_car_position[0] * selected_lane_data[i].lane_vec[1] \
                - present_car_position[1] * selected_lane_data[i].lane_vec[0] \
                + (selected_lane_data[i].lane_vec[0] * Point_Map.at(selected_lane_data[i].BPID - 1).bx \
                - selected_lane_data[i].lane_vec[1] * Point_Map.at(selected_lane_data[i].BPID - 1).ly)) \
                / sqrt(pow(selected_lane_data[i].lane_vec[1], 2) + pow(selected_lane_data[i].lane_vec[0], 2));
            if (selected_lane_data[i].cos_with_vehicle_heading < cos(45 * PI / 180)) {
                toDelete.push_back(i);
            }
        }
        //---------------------------------- Delete those lanes whose direction with vehicle's heading is larger than cos(15 degree).
        for (int del = 0; del < toDelete.size(); ++del) {
            for (std::vector<Lane_Data>::iterator iter = selected_lane_data.begin(); iter != selected_lane_data.end();) {
                if (iter == selected_lane_data.begin() + toDelete[del]) {
                    iter = selected_lane_data.erase(iter);
                } else {
                    iter++;
                }
            }
        }
    }

    void HDPipeline_Callback(const ImageConstPtr &img, const PoseStampedConstPtr &pos) {
    //void HDPipeline_Callback(const ImageConstPtr &img, const ImageConstPtr &laneNet, const PoseStampedConstPtr &pos) {
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
                last_car_position[0] = pos->pose.position.x; // Baselink to Localizer
                last_car_position[1] = pos->pose.position.y;
                last_car_position[2] = pos->pose.position.z;
            } else {
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

                last_car_position[0] = present_car_position[0];
                last_car_position[1] = present_car_position[1];
                last_car_position[2] = present_car_position[2];

                //std::cout << "displacement: " << sqrt( pow(ne_vector[0], 2) + pow(ne_vector[1], 2) ) << std::endl;


                
                //////////////////////
                // ThinkTron HD Map //
                //////////////////////
                ThinkTron_HD_Map(present_car_position, ne_vector, long_coefficients, lat_coefficients);
                
                //---------------------------------- If there is at least one lane for reference, choose the closest one and use its end point to sample lines' data.
                //---------------------------------- If not, use the raw GPS data to sample lines' data.
                //selected_lane_data.clear();
                
                int selected_lane_idx = 0;
                if (selected_lane_data.size() > 0) {
                    //std::cout << "--------------------------------" << std::endl;
                    for (int i = 0; i < selected_lane_data.size(); ++i) {
                        //std::cout << "cos: " << selected_lane_data[i].cos_with_vehicle_heading << std::endl;
                        //std::cout << "lane vector: " << selected_lane_data[i].lane_vec[0] << ", " << selected_lane_data[i].lane_vec[1] << std::endl;
                        //std::cout << "car vector: " << ne_vector[0] << ", " << ne_vector[1] << std::endl;
                        if (i == 0) {
                            selected_lane_idx = i;
                            closest_distance = selected_lane_data[i].distance_to_vehicle;
                        } else {
                            if (selected_lane_data[i].distance_to_vehicle < closest_distance) {
                                selected_lane_idx = i;
                                closest_distance = selected_lane_data[i].distance_to_vehicle;
                            }
                        }
                    }
                    //std::cout << "--------------------------------" << std::endl;
                    //std::cout << selected_lane_data[selected_lane_idx].LnID << std::endl;
                    for (int i = 0; i < Point_Map.size(); ++i) {
                        //---------------------------------- Sample those points which are located in the estimated range
                        measure_distance = abs(Point_Map[i].ly * long_coefficients[0] + Point_Map[i].bx * long_coefficients[1] + long_coefficients[2]);
                        farest_distance = sqrt(pow(Point_Map[i].ly - Point_Map.at(selected_lane_data[selected_lane_idx].FPID - 1).ly, 2) \
                                                + pow(Point_Map[i].bx - Point_Map.at(selected_lane_data[selected_lane_idx].FPID - 1).bx, 2));
                        if (measure_distance <= lateral_measure_range \
                        && farest_distance <= 30 \
                        && ((Point_Map[i].ly - Point_Map.at(selected_lane_data[selected_lane_idx].FPID - 1).ly) * unit_ne_vector[0] \
                        + (Point_Map[i].bx - Point_Map.at(selected_lane_data[selected_lane_idx].FPID - 1).bx) * unit_ne_vector[1]) >= 0) {
                        //if (measure_distance <= lateral_measure_range && farest_distance <= 20) {
                            selected_points.push_back(Point_Map[i]);
                        }
                    }
                    lane_point_in_map.header.frame_id = pos->header.frame_id;
                    lane_point_in_map.header.stamp = pos->header.stamp;
                    lane_point_in_map.pose.position.x = Point_Map.at(selected_lane_data[selected_lane_idx].FPID - 1).ly;
                    lane_point_in_map.pose.position.y = Point_Map.at(selected_lane_data[selected_lane_idx].FPID - 1).bx;
                    lane_point_in_map.pose.position.z = pos->pose.position.z;
                    lane_point_in_map.pose.orientation.x = 0;
                    lane_point_in_map.pose.orientation.y = 0;
                    lane_point_in_map.pose.orientation.z = 0;
                    lane_point_in_map.pose.orientation.w = 1;
                } else {
                    std::cout << "No selected lane!" << std::endl;
                    for (int i = 0; i < Point_Map.size(); ++i) {
                        //std::cout << Point_Map[i].bx << ", " << Point_Map[i].ly << std::endl; (-8xxx, 6xxx)
                        //---------------------------------- Sample those points which are located in the estimated range
                        
                        measure_distance = abs(Point_Map[i].ly * long_coefficients[0] + Point_Map[i].bx * long_coefficients[1] + long_coefficients[2]);
                        farest_distance = sqrt(pow(Point_Map[i].ly - present_car_position[0], 2) \
                                                + pow(Point_Map[i].bx - present_car_position[1], 2));
                        if (measure_distance <= lateral_measure_range \
                        && farest_distance <= 30 \
                        && ((Point_Map[i].ly - present_car_position[0]) * unit_ne_vector[0] \
                        + (Point_Map[i].bx - present_car_position[1]) * unit_ne_vector[1]) >= 0) {
                        //if (measure_distance <= lateral_measure_range && farest_distance <= 20) {
                            selected_points.push_back(Point_Map[i]);
                        }
                    }
                    lane_point_in_map.header.frame_id = pos->header.frame_id;
                    lane_point_in_map.header.stamp = pos->header.stamp;
                    lane_point_in_map.pose.position.x = pos->pose.position.x;
                    lane_point_in_map.pose.position.y = pos->pose.position.y;
                    lane_point_in_map.pose.position.z = pos->pose.position.z;
                    lane_point_in_map.pose.orientation.x = 0;
                    lane_point_in_map.pose.orientation.y = 0;
                    lane_point_in_map.pose.orientation.z = 0;
                    lane_point_in_map.pose.orientation.w = 1;
                }
                
                
                //lanes_selection(present_car_position);
                whitelines_selection();
                if (points_to_transform.size() > 0) {
                
                    // Save data to world_points_array
                    world_points_array.layout.dim[0].label = "lane";
                    world_points_array.layout.dim[1].label = "point";
                    world_points_array.layout.dim[2].label = "coord";
                    world_points_array.layout.dim[0].size = points_to_transform.size();
                    world_points_array.layout.dim[1].size = points_to_transform[0].size(); // 2 points
                    world_points_array.layout.dim[2].size = points_to_transform[0][0].size(); // 3 coordinates in world
                    world_points_array.layout.dim[0].stride = points_to_transform.size() * points_to_transform[0].size() * points_to_transform[0][0].size();
                    world_points_array.layout.dim[1].stride = points_to_transform[0].size() * points_to_transform[0][0].size();
                    world_points_array.layout.dim[2].stride = points_to_transform[0][0].size();
                    world_points_array.data.clear();
                    world_points_array.data.resize(points_to_transform.size() * points_to_transform[0].size() * points_to_transform[0][0].size());
                    // Save data to image_points_array
                    image_points_array.layout.dim[0].label = "lane";
                    image_points_array.layout.dim[1].label = "point";
                    image_points_array.layout.dim[2].label = "coord";
                    image_points_array.layout.dim[0].size = points_to_transform.size();
                    image_points_array.layout.dim[1].size = points_to_transform[0].size(); // 2 points
                    image_points_array.layout.dim[2].size = 2; // 2 coordinates in image
                    image_points_array.layout.dim[0].stride = points_to_transform.size() * points_to_transform[0].size() * 2;
                    image_points_array.layout.dim[1].stride = points_to_transform[0].size() * 2;
                    image_points_array.layout.dim[2].stride = 2;
                    image_points_array.data.clear();
                    image_points_array.data.resize(points_to_transform.size() * points_to_transform[0].size() * 2);
                
                    listeners_ptr1->waitForTransform("map", "camera", pos->header.stamp, ros::Duration(10.0) );
                    //listeners_ptr2->waitForTransform("map", "xsens/GPS", pos->header.stamp, ros::Duration(10.0) );
                    //std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
                
                    for (int o = 0; o < points_to_transform.size(); ++o) {
                        for (int oo = 0; oo < points_to_transform[o].size(); ++oo) {
                            
                            if (points_to_transform[o][oo].size() > 0) {
                                
                                point_pose_in_map.header.frame_id = pos->header.frame_id;
                                point_pose_in_map.header.stamp = pos->header.stamp;
                                point_pose_in_map.pose.position.x = points_to_transform[o][oo][0];
                                point_pose_in_map.pose.position.y = points_to_transform[o][oo][1];
                                point_pose_in_map.pose.position.z = points_to_transform[o][oo][2];
                                
                                point_pose_in_map.pose.orientation.x = 0;
                                point_pose_in_map.pose.orientation.y = 0;
                                point_pose_in_map.pose.orientation.z = 0;
                                point_pose_in_map.pose.orientation.w = 1;

                                listeners_ptr1->transformPose("camera", point_pose_in_map, point_pose_in_camera);
                                //listeners_ptr2->transformPose("xsens/GPS", point_pose_in_map, point_pose_in_xsens);
                                X = point_pose_in_camera.pose.position.x;
                                Y = point_pose_in_camera.pose.position.y;
                                Z = point_pose_in_camera.pose.position.z;
                                X_xsens = point_pose_in_xsens.pose.position.x;
                                Y_xsens = point_pose_in_xsens.pose.position.y;
                                Z_xsens = point_pose_in_xsens.pose.position.z;
                                //printf("(X, Y, Z): (%f, %f, %f)\n", X_xsens, Y_xsens, Z_xsens);
                                //---------------------------------- Output the array which stores the coordinates referring to xsens
                                //world_points_array.data[o * 2 * 3 + oo * 3 + 0] = X_xsens;
                                //world_points_array.data[o * 2 * 3 + oo * 3 + 1] = Y_xsens;
                                //world_points_array.data[o * 2 * 3 + oo * 3 + 2] = Z_xsens;
                                //---------------------------------- Output the array which stores the coordinates referring to map
                                world_points_array.data[o * 2 * 3 + oo * 3 + 0] = points_to_transform[o][oo][0];
                                world_points_array.data[o * 2 * 3 + oo * 3 + 1] = points_to_transform[o][oo][1];
                                world_points_array.data[o * 2 * 3 + oo * 3 + 2] = points_to_transform[o][oo][2];

                                //printf("(Xc, Yc, Zc): (%f, %f, %f)\n", X, Y, Z);
                                X = X / Z;
                                Y = Y / Z;
                                //printf("(X, Y, Z): (%f, %f, %f)\n", points_to_transform[o][oo][0], \
                                points_to_transform[o][oo][1], points_to_transform[o][oo][2]);//6xxx, -8xxx, -6x
                                //printf("(X, Y): (%f, %f)\n", X, Y);
                                //printf("(U, V): (%f, %f)\n", cameraMat[0] * X + cameraMat[2], cameraMat[4] * Y + cameraMat[5]);
                            
                                // Radial Distortion
                                //X = x_Radial_Distortion(X, Y);
                                //Y = y_Radial_Distortion(X, Y);
                                //std::cout << X << "," << Y << std::endl;
                                
                                // Tangential Distortion
                                //X = x_Tangential_Distortion(X, Y);
                                //Y = y_Tangential_Distortion(X, Y);
                                //std::cout << X << "," << Y << std::endl;

                                if (oo == 0) {
                                    imageCoord1[0] = cameraMat[0] * X + cameraMat[2];
                                    imageCoord1[1] = cameraMat[4] * Y + cameraMat[5];
                                    //imageCoord1[0] = (imageCoord1[0] > frame.width) ? (frame.width) : ((imageCoord1[0] < 0) ? (0) : (imageCoord1[0]));
                                    //imageCoord1[1] = (imageCoord1[1] > frame.height) ? (frame.height) : ((imageCoord1[1] < 0) ? (0) : (imageCoord1[1]));
                                    image_points_array.data[o * 2 * 2 + 0] = (imageCoord1[1] > frame.height / 2) ? (imageCoord1[0]) : (0);
                                    image_points_array.data[o * 2 * 2 + 1] = (imageCoord1[1] > frame.height / 2) ? (imageCoord1[1]) : (0);
                                    
                                } else if (oo == 1) {
                                    imageCoord2[0] = cameraMat[0] * X + cameraMat[2];
                                    imageCoord2[1] = cameraMat[4] * Y + cameraMat[5];
                                    //imageCoord2[0] = (imageCoord2[0] > frame.width) ? (frame.width) : ((imageCoord2[0] < 0) ? (0) : (imageCoord2[0]));
                                    //imageCoord2[1] = (imageCoord2[1] > frame.height) ? (frame.height) : ((imageCoord2[1] < 0) ? (0) : (imageCoord2[1]));
                                    image_points_array.data[o * 2 * 2 + 2] = (imageCoord2[1] > frame.height / 2) ? (imageCoord2[0]) : (0);
                                    image_points_array.data[o * 2 * 2 + 3] = (imageCoord2[1] > frame.height / 2) ? (imageCoord2[1]) : (0);
                                }
                            }
                            //break;
                        }
                        
                        if ((imageCoord1[1] > frame.height / 2) && (imageCoord2[1] > frame.height / 2)) {
                            //---------------------------------- Plot the discrepancy lanes
                            //printf("(U1, V1): (%d, %d)\n", imageCoord1[0], imageCoord1[1]);
                            //printf("(U2, V2): (%d, %d)\n", imageCoord2[0], imageCoord2[1]);
                            cv::line(cv_image->image, cv::Point(imageCoord1[0], imageCoord1[1]), \
                                                        cv::Point(imageCoord2[0], imageCoord2[1]), CV_RGB(255, 255, 0), 3);
                            /*
                            cv::line(cv_laneNet_image->image, cv::Point(imageCoord1[0], imageCoord1[1]), \
                                                        cv::Point(imageCoord2[0], imageCoord2[1]), CV_RGB(255, 255, 0), 3);
                            */
                            //cv::circle(cv_image->image, cv::Point(estimate_point[0], estimate_point[1]), 2, CV_RGB(0,255,0), -1);

                            imageCoord1[0] = 0;
                            imageCoord1[1] = 0;
                            imageCoord2[0] = 0;
                            imageCoord2[1] = 0;
                        } else {
                            continue;
                        }
                        
                    }

                    //std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
                }

                image_lane.publish(cv_image->toImageMsg());
                //fuse_image_lane.publish(cv_laneNet_image->toImageMsg());
                world_line_points.publish(world_points_array);
                image_line_points.publish(image_points_array);
                world_lane_points.publish(lane_point_in_map);
                
            }

            selected_lane_data.clear();
            toDelete.clear();
            selected_points.clear();
            selected_nodes.clear();
            selected_lanes.clear();
            selected_nodes_of_lanes.clear();
            selected_points_of_nodes.clear();
            points_to_transform.clear();
            selected_lines.clear();
            selected_whitelines.clear();
            selected_lines_of_whitelines.clear();
            selected_points_of_lines.clear();

            world_points_array.data.clear();
            image_points_array.data.clear();

        } catch (tf::TransformException &ex) {
            ROS_ERROR("%s", ex.what());
            ros::Duration(1.0).sleep();
            //continue;
        }
    }
    
public:
    
    HDLanePipeline(tf::TransformListener* in_listeners_ptr1, tf::TransformListener* in_listeners_ptr2) {
        listeners_ptr1 = in_listeners_ptr1;
        listeners_ptr2 = in_listeners_ptr2;
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
        //std::string laneNet_topic_name = "/laneNet_image";
        std::string laneNet_topic_name = "/erfNet_image";

        image_lane = n.advertise<Image>("HD_lane_topic", 1);// Image or autoware_msgs::ImageLaneObjects
        //fuse_image_lane = n.advertise<Image>("fuse_lane_topic", 1);// Image or autoware_msgs::ImageLaneObjects
        world_line_points = n.advertise<std_msgs::Float32MultiArray>("world_line_points_topic", 1);
        image_line_points = n.advertise<std_msgs::Int32MultiArray>("image_line_points_topic", 1);
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
        //ros::Subscriber point_sub = n.subscribe(v_map_point_topic_name, 1000, vMap_Callback<vector_map_msgs::PointArray, vector_map_msgs::Point>);
        //ros::Subscriber line_sub = n.subscribe(v_map_line_topic_name, 1000, vMap_Callback<vector_map_msgs::LineArray, vector_map_msgs::Line>);
        
        //read_sorted_lanes();
        //read_sorted_whitelines();

        //---------------------------------- Time Synchronizer
        /*
        image_sub.subscribe(n, image_topic_name, 1);
        car_position_sub.subscribe(n, car_position_topic_name, 1);
        sync_.reset(new sync(MySyncPolicy(10), image_sub, car_position_sub));
        sync_->registerCallback(boost::bind(&HDLanePipeline::HDPipeline_Callback, this, _1, _2));
        */
        
        message_filters::Subscriber<Image> image_sub(n, image_topic_name, 1);
        message_filters::Subscriber<Image> laneNet_sub(n, laneNet_topic_name, 1);
        message_filters::Subscriber<PoseStamped> car_position_sub(n, car_position_topic_name, 1);
        /*
        typedef sync_policies::ApproximateTime<Image, Image, PoseStamped> MySyncPolicy;
        // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
        Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image_sub, laneNet_sub, car_position_sub);
        sync.registerCallback(boost::bind(&HDLanePipeline::HDPipeline_Callback, this, _1, _2, _3));
        */
        typedef sync_policies::ApproximateTime<Image, PoseStamped> MySyncPolicy;
        // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
        Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image_sub, car_position_sub);
        sync.registerCallback(boost::bind(&HDLanePipeline::HDPipeline_Callback, this, _1, _2));

        ros::spin();

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
};


/*
template <class T, class U>
void vMap_Callback(T msg_array) {

    //std::string typeID = typeid(msg_array).name();

    if (typeid(msg_array) == typeid(vector_map_msgs::PointArray)) {
        std::cout << "point" << std::endl;
        Point_Map.reserve(msg_array.data.size());
        //for(int i = 0; i < msg_array.data.size(); ++i)
        //    Point_Map.push_back(msg_array.data[i]);
        //Point_Map.assign(msg_array.data.begin(), msg_array.data.end());
        //std::cout << Point_Map.size() << std::endl;
    } else if (typeid(msg_array) == typeid(vector_map_msgs::LineArray)) {
        std::cout << "line" << std::endl;
        
    } else {
        std::cout << "Shot the fuck up" << std::endl;
    }
    //U msg = msg_array.data[0];
    //std::cout << msg << std::endl;
    
    const int array_size = msg_array.data.size();
    //std::cout << msg_array.data.size() << std::endl;
	// print all the remaining numbers
    
}
*/

int main(int argc, char *argv[]) {
    //std::cout << "Hello World!" << std::endl;
    ros::init(argc, argv, "HD_lane");
    tf::TransformListener listeners1, listeners2;
    HDLanePipeline app(&listeners1, &listeners2);
    

    app.Run();

    return 0;

}