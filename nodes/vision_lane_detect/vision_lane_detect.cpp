/*
 * Copyright 2015-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
  lane detection program using OpenCV
  (originally developed by Google)
  cut vehicle tracking frmo original program
  reference：https://code.google.com/p/opencv-lane-vehicle-track/source/browse/

  customized for demo_with_webcam
 */
#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <math.h>
#include <list>
#include <ctime>
#include "utils.h"
#include "line_utils.h"
#include "kalman_filter.h"
#include "particle_filter.h"
#include <string>

#include <std_msgs/Int32MultiArray.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>

#include <vision_lane_detect/Int32MultiArray_H.h>
#include <vision_lane_detect/Float32MultiArray_H.h>

/*#include "switch_release.h"*/

#if !defined(USE_POSIX_SHARED_MEMORY)
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include "autoware_msgs/ImageLaneObjects.h"
#endif

#if defined(USE_POSIX_SHARED_MEMORY)
/* global variables and function to use shared memory */
static unsigned char    *shrd_ptr;
static int              *shrd_ptr_height, *shrd_ptr_width;
extern void      attach_ShareMem(void);
extern IplImage *getImage_fromSHM(void);
extern void      setImage_toSHM(IplImage *result);
extern void      detach_ShareMem(void);
#endif

#ifndef RELEASE
#define SHOW_DETAIL // if this macro is valid, grayscale/edge/half images are displayed
#endif

//static ros::Publisher image_lane_objects;
static ros::Publisher image_lane;

ros::Publisher image_line_points;
//std_msgs::Int32MultiArray image_points_array;
vision_lane_detect::Int32MultiArray_H image_points_array;


Line line_lt(10);
Line line_rt(10);
KalmanFilter bev_kf("BEV");
ParticleFilter bev_pf("BEV");
int processed_frames = 0;

// clip portion of the image
static void crop(IplImage *src, IplImage *dst, CvRect rect)
{
  cvSetImageROI(src, rect);     // clip "rect" from "src" image
  cvCopy(src, dst);             // copy clipped portion to "dst"
  cvResetImageROI(src);         // reset cliping of "src"
}

struct Lane {
  Lane(){}
  Lane(CvPoint a, CvPoint b, float angle, float kl, float bl)
    : p0(a), p1(b), angle(angle), votes(0),visited(false),found(false),k(kl),b(bl) { }
  CvPoint p0, p1;
  float angle;
  int votes;
  bool visited, found;
  float k, b;
};

struct Status {
  Status(): reset(true),lost(0){}
  ExpMovingAverage k, b;
  bool reset;
  int lost;
};

#define GREEN  CV_RGB(0, 255, 0)
#define RED    CV_RGB(255, 0, 0)
#define BLUE   CV_RGB(0, 0, 255)
#define PURPLE CV_RGB(255, 0, 255)

Status laneR, laneL;

enum {
  SCAN_STEP           = 5,      // in pixels
  LINE_REJECT_DEGREES = 10,     // in degrees
  BW_TRESHOLD         = 250,    // edge response strength to recognize for 'WHITE'
  BORDERX             = 10,     // px, skip this much from left & right borders
  MAX_RESPONSE_DIST   = 5,      // px

  CANNY_MIN_TRESHOLD = 1,       // edge detector minimum hysteresis threshold
  CANNY_MAX_TRESHOLD = 100,     // edge detector maximum hysteresis threshold

  HOUGH_TRESHOLD        = 50,   // line approval vote threshold
  HOUGH_MIN_LINE_LENGTH = 50,   // remove lines shorter than this treshold
  HOUGH_MAX_LINE_GAP    = 100   // join lines to one with smaller than this gaps
};


#define K_VARY_FACTOR 0.2f
#define B_VARY_FACTOR 20
#define MAX_LOST_FRAMES 30

static void process_image_common(IplImage *frame)
{

  std::clock_t t_start = std::clock();
  CvFont font;
  cvInitFont(&font, CV_FONT_VECTOR0, 0.25f, 0.25f);

  CvSize video_size;
#if defined(USE_POSIX_SHARED_MEMORY)
  video_size.height = *shrd_ptr_height;
  video_size.width  = *shrd_ptr_width;
#else
  // XXX These parameters should be set ROS parameters
  video_size.height = frame->height;
  video_size.width  = frame->width;
  //std::cout << frame->height << ", " << frame->width << std::endl;
#endif
  CvSize    frame_size = cvSize(video_size.width, video_size.height/2);

  IplImage *temp_frame = cvCreateImage(frame_size, IPL_DEPTH_8U, 3);
  IplImage *half_frame = cvCreateImage(cvSize(video_size.width/2, video_size.height/2), IPL_DEPTH_8U, 3);
  
  CvMemStorage *houghStorage = cvCreateMemStorage(0);

  cvPyrDown(frame, half_frame, CV_GAUSSIAN_5x5); // Reduce the image by 2

  /* we're intersted only in road below horizont - so crop top image portion off */
  crop(frame, temp_frame, cvRect(0, frame_size.height, frame_size.width, frame_size.height));
  ////////////////
  // Edge Lines //
  ////////////////

  // smoothing image more strong than original program
  IplImage *gray  = cvCreateImage(frame_size, IPL_DEPTH_8U, 1);
  IplImage *edges = cvCreateImage(frame_size, IPL_DEPTH_8U, 1);
  cvCvtColor(temp_frame, gray, CV_BGR2GRAY); // convert to grayscale
  //cvSmooth(gray, gray, CV_GAUSSIAN, 15, 15);
  cvCanny(gray, edges, CANNY_MIN_TRESHOLD, CANNY_MAX_TRESHOLD);

  /////////////////
  // Median Blur //
  /////////////////
  IplImage *median  = cvCreateImage(frame_size, IPL_DEPTH_8U, 1);
  cvSmooth(gray, median, CV_MEDIAN, 7);
  cvThreshold(median, median, 135, 255, CV_THRESH_BINARY);
  //////////////////
  // Yellow Lines //
  //////////////////
  IplImage *hsv = cvCreateImage(frame_size, IPL_DEPTH_8U, 3);
  cvCvtColor(temp_frame, hsv, CV_BGR2HSV); // convert to hsv scale
  IplImage *yellow_hsv_th = cvCreateImage(frame_size, IPL_DEPTH_8U, 1);
  cvInRangeS(hsv, cvScalar(0, 70, 70), cvScalar(50, 255, 255), yellow_hsv_th);

  /////////////////
  // White Lines //
  /////////////////
  IplImage *binary = cvCreateImage(frame_size, IPL_DEPTH_8U, 1);
  cvCvtColor(temp_frame, binary, CV_BGR2GRAY); 
  cvEqualizeHist(binary, binary);
  cvThreshold(binary, binary, 200, 255, CV_THRESH_BINARY);
  //cvShowImage("Binary1", binary);
  //cvAddWeighted(yellow_hsv_th, 1.0, binary, 1.0, 0.0, binary);
  cvOr(binary, yellow_hsv_th, binary);
  cvAnd(binary, median, binary);
  
  ///////////////////
  // Find Contours //
  ///////////////////
  IplImage* binary_clone = cvCloneImage(binary);
  IplImage* img_out = cvCreateImage(frame_size, IPL_DEPTH_8U, 3);
  cvCvtColor(binary, img_out, CV_GRAY2BGR);
  CvSeq* contour = NULL;
  CvMemStorage* contourStorage = cvCreateMemStorage(0);
  
  CvContourScanner scanner = NULL;     
  scanner = cvStartFindContours(binary_clone, contourStorage, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, cvPoint(0,0)); 
  //CvRect rect;
  CvScalar black_mask = cvScalar(0, 0, 0);
  //CvPoint pt1;
  //CvPoint pt2;
  //double area = 0;
  CvBox2D ellipse;
  while (contour = cvFindNextContour(scanner)) {
    if (contour->total >= 5) {    
      ellipse = cvFitEllipse2(contour);
      //std::cout << ellipse.angle << std::endl;
      if (ellipse.angle < 30 || ellipse.angle > 150) {
        cvDrawContours(img_out, contour, black_mask, black_mask, 0, CV_FILLED);
      }
    }
  } 
  //cvShowImage("img_out", img_out);
  cvCvtColor(img_out, binary, CV_BGR2GRAY);

  /////////////////
  // Sobel Lines //
  /////////////////
  
  IplImage *sobel = cvCreateImage(frame_size, IPL_DEPTH_8U, 1);
  cvCvtColor(temp_frame, sobel, CV_BGR2GRAY); 
  cvSobel(binary, sobel, 1, 0, 7);
  cvThreshold(sobel, sobel, 230, 255, CV_THRESH_BINARY);
  cvAnd(binary, sobel, binary);
  //cvShowImage("Sobel", binary);

  ////////////////
  // Morphology //
  ////////////////
  //IplImage *closing = cvCreateImage(frame_size, IPL_DEPTH_8U, 1);
  int filterSize = 3;
  IplConvKernel *convKernel = cvCreateStructuringElementEx(filterSize, filterSize, (filterSize - 1)/2, (filterSize - 1)/2, CV_SHAPE_RECT, NULL);
  cvMorphologyEx(binary, binary, NULL, convKernel, CV_MOP_OPEN);
  
  ////////////////////
  // Birds Eye View //
  ////////////////////
  IplImage* bev = cvCreateImage(frame_size, IPL_DEPTH_8U, 1);
  CvPoint2D32f BEV_src[4];
  BEV_src[0] = cvPoint2D32f(200, 100);
  BEV_src[1] = cvPoint2D32f(600, 100);
  BEV_src[2] = cvPoint2D32f(700, 251);
  BEV_src[3] = cvPoint2D32f(100, 251);
  //CvPoint2D32f BEV_src_[1] = {&BEV_src[0]};
  CvPoint2D32f BEV_dst[4];
  BEV_dst[0] = cvPoint2D32f(100, 0);
  BEV_dst[1] = cvPoint2D32f(700, 0);
  BEV_dst[2] = cvPoint2D32f(700, 251);
  BEV_dst[3] = cvPoint2D32f(100, 251);
  //CvPoint2D32f BEV_dst_[1] = {&BEV_dst[0]};
  CvMat *M = cvCreateMat(3, 3, CV_32FC1);
  CvMat *inv_M = cvCreateMat(3, 3, CV_32FC1);
  cvGetPerspectiveTransform(BEV_src, BEV_dst, M);
  cvGetPerspectiveTransform(BEV_dst, BEV_src, inv_M);
  cvWarpPerspective(binary, bev, M);

  cvThreshold(bev, bev, 1, 255, CV_THRESH_BINARY);
  //cvShowImage("BEV", bev);
  //if (processed_frames == 20) {
  //  cvSaveImage("BEV.png", bev);
  //}

  IplImage *slide_window  = cvCreateImage(frame_size, IPL_DEPTH_8U, 3);
  cvCvtColor(bev, slide_window, CV_GRAY2BGR);

  //std::pair<Line, Line>(line_lt, line_rt) = get_fits_by_sliding_windows(bev, line_lt, line_rt);

  ///////////////////////
  // Interesting Range //
  ///////////////////////
  IplImage* mask = cvCreateImage(frame_size, IPL_DEPTH_8U, 1);
  cvZero(mask);
  CvPoint PointArray[4];
  PointArray[0] = cvPoint(50, 80);
  PointArray[1] = cvPoint(750, 80);
  PointArray[2] = cvPoint(800, 251);
  PointArray[3] = cvPoint(0, 251);
  CvPoint *PointArray_[1] = {&PointArray[0]};
  int PolyVertexNumber[1] = {4};
  cvFillPoly(mask, PointArray_, PolyVertexNumber, 1, cvScalar(255));
  cvAnd(bev, mask, bev);

  //cvShowImage("Binary3", binary);
  int poly_order = 1;
  if (processed_frames > 0 && line_lt.detected && line_rt.detected) {
    get_fits_by_previous_fits(bev, line_lt, line_rt);
  } else {
    get_fits_by_sliding_windows(processed_frames, bev, line_lt, line_rt);
  }

  //get_fits_by_sliding_windows(bev, line_lt, line_rt);

  //get_fits_by_sliding_windows(bev, line_lt, line_rt);
  //cvShowImage("BEV", bev);
  
  IplImage *output  = cvCreateImage(frame_size, IPL_DEPTH_8U, 3);
  if (processed_frames > 0 || (line_lt.detected && line_rt.detected)) {
    //bev_kf.bev_kalman_filter(bev, line_lt, line_rt, poly_order, processed_frames, "pf");
    bev_pf.bev_particle_filter(bev, line_lt, line_rt, poly_order, processed_frames);
    /////////////////
    // Show Output //
    /////////////////
    cvZero(output);
    for (int i = 0; i < (frame->height / 2); i = i + 10) {
      CvPoint ipt_l;
      CvPoint ipt_r;
      ipt_l.y = i;
      ipt_r.y = i;
      ipt_l.x = 0;
      ipt_r.x = 0;
      for (int j = 0; j < poly_order + 1; ++j) {
        ipt_l.x += line_lt.last_fit_pixel.at<double>(j, 0) * pow(i, j);
        ipt_r.x += line_rt.last_fit_pixel.at<double>(j, 0) * pow(i, j);
      }
      cvCircle(output, ipt_l, 5, CvScalar(255, 0, 0), CV_FILLED, CV_AA);
      cvCircle(output, ipt_r, 5, CvScalar(0, 0, 255), CV_FILLED, CV_AA);
    }
    
    cvWarpPerspective(output, output, inv_M);
    cvAddWeighted(temp_frame, 1.0, output, 1, 0.0, output);
    //cvShowImage("Output", output);
    cvSetImageROI(frame, cvRect(0, frame_size.height, frame_size.width, frame_size.height));
    cvCopy(output, frame, NULL);
    cvResetImageROI(frame);
    //cvShowImage("Output", frame);
    //if (processed_frames == 20) {
    //  cvSaveImage("Output.png", frame);
    //}
    
    //crop(frame, temp_frame, cvRect(0, frame_size.height, frame_size.width, frame_size.height));
    /*
    IplImage *output_comp  = cvCreateImage(cvSize(video_size.width, video_size.height / 2 + 1), IPL_DEPTH_8U, 3);
    crop(frame, output_comp, cvRect(0, 0, video_size.width, video_size.height / 2 + 1));
    cv::Mat mat_out_up = cv::cvarrToMat(output_comp);
    cv::Mat mat_out_down = cv::cvarrToMat(output);
    cv::Mat mat_out;
    cv::vconcat(mat_out_up, mat_out_down, mat_out);
    IplImage output_tmp = mat_out;
    output = &output_tmp;
    cvShowImage("Output", output);
    cvReleaseImage(&output_comp);
    mat_out_up.release();
    mat_out_down.release();
    mat_out.release();
    */
    ///////////////////
    // Publish Topic //
    ///////////////////
    
    image_points_array.data.clear();
    image_points_array.data.resize(2 * 2 * 2);

    cv::Mat mat_inv_M = cv::cvarrToMat(inv_M);
    mat_inv_M.convertTo(mat_inv_M, CV_64F);
    //std::cout << mat_inv_M << std::endl;

    cv::Mat src_top_point(3, 1, CV_64F), dst_top_point(3, 1, CV_64F);
    cv::Mat src_bot_point(3, 1, CV_64F), dst_bot_point(3, 1, CV_64F);
    // left line
    src_top_point.at<double>(0, 0) = line_lt.last_fit_pixel.at<double>(0, 0) * pow(0, 0) \
                                  + line_lt.last_fit_pixel.at<double>(1, 0) * pow(0, 1);
    src_top_point.at<double>(1, 0) = 0;
    src_top_point.at<double>(2, 0) = 1;
    dst_top_point = mat_inv_M * src_top_point;
    dst_top_point.at<double>(0, 0) = int(dst_top_point.at<double>(0, 0) / dst_top_point.at<double>(2, 0));
    dst_top_point.at<double>(1, 0) = int(dst_top_point.at<double>(1, 0) / dst_top_point.at<double>(2, 0));
    dst_top_point.at<double>(2, 0) = int(dst_top_point.at<double>(2, 0) / dst_top_point.at<double>(2, 0));

    src_bot_point.at<double>(0, 0) = line_lt.last_fit_pixel.at<double>(0, 0) * pow(frame->height / 2, 0) \
                                  + line_lt.last_fit_pixel.at<double>(1, 0) * pow(frame->height / 2, 1);
    src_bot_point.at<double>(1, 0) = frame->height / 2;
    src_bot_point.at<double>(2, 0) = 1;
    dst_bot_point = mat_inv_M * src_bot_point;
    dst_bot_point.at<double>(0, 0) = int(dst_bot_point.at<double>(0, 0) / dst_bot_point.at<double>(2, 0));
    dst_bot_point.at<double>(1, 0) = int(dst_bot_point.at<double>(1, 0) / dst_bot_point.at<double>(2, 0));
    dst_bot_point.at<double>(2, 0) = int(dst_bot_point.at<double>(2, 0) / dst_bot_point.at<double>(2, 0));

    image_points_array.data[0] = dst_top_point.at<double>(0, 0);
    image_points_array.data[1] = dst_top_point.at<double>(1, 0) + int(frame->height / 2);
    image_points_array.data[2] = dst_bot_point.at<double>(0, 0);
    image_points_array.data[3] = dst_bot_point.at<double>(1, 0) + int(frame->height / 2);
    // right line
    src_top_point.at<double>(0, 0) = line_rt.last_fit_pixel.at<double>(0, 0) * pow(0, 0) \
                                  + line_rt.last_fit_pixel.at<double>(1, 0) * pow(0, 1);
    src_top_point.at<double>(1, 0) = 0;
    src_top_point.at<double>(2, 0) = 1;
    dst_top_point = mat_inv_M * src_top_point;
    dst_top_point.at<double>(0, 0) = int(dst_top_point.at<double>(0, 0) / dst_top_point.at<double>(2, 0));
    dst_top_point.at<double>(1, 0) = int(dst_top_point.at<double>(1, 0) / dst_top_point.at<double>(2, 0));
    dst_top_point.at<double>(2, 0) = int(dst_top_point.at<double>(2, 0) / dst_top_point.at<double>(2, 0));
    
    src_bot_point.at<double>(0, 0) = line_rt.last_fit_pixel.at<double>(0, 0) * pow(frame->height / 2, 0) \
                                  + line_rt.last_fit_pixel.at<double>(1, 0) * pow(frame->height / 2, 1);
    src_bot_point.at<double>(1, 0) = frame->height / 2;
    src_bot_point.at<double>(2, 0) = 1;
    dst_bot_point = mat_inv_M * src_bot_point;
    dst_bot_point.at<double>(0, 0) = int(dst_bot_point.at<double>(0, 0) / dst_bot_point.at<double>(2, 0));
    dst_bot_point.at<double>(1, 0) = int(dst_bot_point.at<double>(1, 0) / dst_bot_point.at<double>(2, 0));
    dst_bot_point.at<double>(2, 0) = int(dst_bot_point.at<double>(2, 0) / dst_bot_point.at<double>(2, 0));
    
    image_points_array.data[4] = dst_top_point.at<double>(0, 0);
    image_points_array.data[5] = dst_top_point.at<double>(1, 0) + int(frame->height / 2);
    image_points_array.data[6] = dst_bot_point.at<double>(0, 0);
    image_points_array.data[7] = dst_bot_point.at<double>(1, 0) + int(frame->height / 2);

    image_line_points.publish(image_points_array);

    mat_inv_M.release();
    src_top_point.release();
    dst_top_point.release();
    src_bot_point.release();
    dst_bot_point.release();

    processed_frames = processed_frames + 1;
  }

#ifdef SHOW_DETAIL
  /* show middle line */
  //cvLine(temp_frame, cvPoint(frame_size.width/2, 0), cvPoint(frame_size.width/2, frame_size.height), CV_RGB(255, 255, 0), 1);

  
  //cvShowImage("Color", temp_frame);
  //cvShowImage("temp_frame", temp_frame);
  //cvShowImage("frame", frame);
#endif

#if defined(USE_POSIX_SHARED_MEMORY)
  setImage_toSHM(frame);
#endif

#ifdef SHOW_DETAIL
  // cvMoveWindow("Gray", 0, 0);
  // cvMoveWindow("Edges", 0, frame_size.height+25);
  // cvMoveWindow("Color", 0, 2*(frame_size.height+25));
#endif
  //cv_binary.release();

  image_points_array.data.clear();
  
  cvReleaseMemStorage(&houghStorage);
  cvReleaseMemStorage(&contourStorage);
  cvReleaseImage(&temp_frame);
  cvReleaseImage(&half_frame);
  cvReleaseImage(&gray);
  cvReleaseImage(&edges);
  cvReleaseImage(&hsv);
  cvReleaseImage(&yellow_hsv_th);
  
  cvReleaseImage(&binary);
  
  cvReleaseImage(&mask);
  cvReleaseImage(&bev);
  cvReleaseImage(&slide_window);
  cvReleaseImage(&output);
  
  cvReleaseMat(&M);
  cvReleaseMat(&inv_M);
  cvReleaseImage(&median);
  cvReleaseImage(&binary_clone);
  cvReleaseImage(&img_out);

  double duration = ( std::clock() - t_start ) / (double) CLOCKS_PER_SEC;
  //std::cout << "Time taken for inference is " << duration << " ms." << std::endl;
  std::cout << "FPS: " << 1 / duration << std::endl;
}

#if !defined(USE_POSIX_SHARED_MEMORY)
static void lane_cannyhough_callback(const sensor_msgs::Image& image_source)
{
  image_points_array.header.stamp = image_source.header.stamp;
  image_points_array.header.frame_id = image_source.header.frame_id;
  cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(image_source, sensor_msgs::image_encodings::BGR8);
  IplImage frame = cv_image->image;
  process_image_common(&frame);

  image_lane.publish(cv_image->toImageMsg());

  cvWaitKey(2);
}
#endif

int main(int argc, char *argv[])
{
#if defined(USE_POSIX_SHARED_MEMORY)
  attach_ShareMem();

  while (1)
    {
      CvMemStorage *houghStorage = cvCreateMemStorage(0);
      IplImage *frame = getImage_fromSHM();
      process_image_common(frame);
      cvReleaseImage(frame);
    }

  detach_ShareMem();
#else
  ros::init(argc, argv, "line_ocv");
  ros::NodeHandle n;
  ros::NodeHandle private_nh("~");
  std::string image_topic_name;
  private_nh.param<std::string>("image_raw_topic", image_topic_name, "/gmsl_camera/port_0/cam_0/image_raw");
  //private_nh.param<std::string>("image_raw_topic", image_topic_name, "/image_raw");
  ROS_INFO("Setting image topic to %s", image_topic_name.c_str());

  image_lane = n.advertise<sensor_msgs::Image>("opencv_lane_topic", 1);
  image_line_points = n.advertise<vision_lane_detect::Int32MultiArray_H>("opencv_image_line_points_topic", 1);
  image_points_array.layout.dim.push_back(std_msgs::MultiArrayDimension());
  image_points_array.layout.dim.push_back(std_msgs::MultiArrayDimension());
  image_points_array.layout.dim.push_back(std_msgs::MultiArrayDimension());
  image_points_array.layout.dim[0].label = "line";
  image_points_array.layout.dim[1].label = "point";
  image_points_array.layout.dim[2].label = "coord";
  image_points_array.layout.dim[0].size = 2; // 2 lines
  image_points_array.layout.dim[1].size = 2; // 2 points
  image_points_array.layout.dim[2].size = 2; // 2 coordinates in image
  image_points_array.layout.dim[0].stride = 2 * 2 * 2;
  image_points_array.layout.dim[1].stride = 2 * 2;
  image_points_array.layout.dim[2].stride = 2;

  ros::Subscriber subscriber = n.subscribe(image_topic_name, 1, lane_cannyhough_callback);
  


  ros::spin();
#endif

  return 0;
}

