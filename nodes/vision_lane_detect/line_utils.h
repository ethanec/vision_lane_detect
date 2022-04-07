#ifndef _LINEUTILS_H_
#define _LINEUTILS_H_

#include <stdio.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <math.h>
#include <list>
#include <string>


class Line {
public:
    Line (int buffer_len) : detected(false)
    {

    }
    ~Line()
    {
        all_point.clear();
        last_fit_pixel.release();
        last_fit_meter.release();
        recent_fits_pixel.clear();
        recent_fits_meter.clear();
    }
    inline void update_line(cv::Mat new_fit_pixel, cv::Mat new_fit_meter, bool d, bool clear_buffer = false) {
        detected = d;
        if (clear_buffer) {
            recent_fits_pixel.clear();
            recent_fits_meter.clear();
        }
        last_fit_pixel = new_fit_pixel;
        last_fit_meter = new_fit_meter;
        if (recent_fits_pixel.size() < 10) {
            recent_fits_pixel.push_back(last_fit_pixel);
            recent_fits_meter.push_back(last_fit_meter);
        } else {
            recent_fits_pixel.pop_front();
            recent_fits_meter.pop_front();
            recent_fits_pixel.push_back(last_fit_pixel);
            recent_fits_meter.push_back(last_fit_meter);
        }
    }

    inline cv::Mat polyfit(double x_ratio, double y_ratio, int order) {

        int size = all_point.size();
        int x_num = order + 1;

        cv::Mat mat_u(size, x_num, CV_64F);
	    cv::Mat mat_y(size, 1, CV_64F);

        for (int i = 0; i < mat_u.rows; ++i) {
            for (int j = 0; j < mat_u.cols; ++j) {
                mat_u.at<double>(i, j) = pow(all_point[i].y * y_ratio, j);
            }
        }
        for (int i = 0; i < mat_y.rows; ++i) {
            mat_y.at<double>(i, 0) = all_point[i].x * x_ratio;
        }
        cv::Mat mat_k(x_num, 1, CV_64F);
	    mat_k = (mat_u.t() * mat_u).inv() * mat_u.t() * mat_y;

        return mat_k;
    }
    
    std::vector<cv::Point> all_point;
    cv::Mat last_fit_pixel;
    cv::Mat last_fit_meter;
    //double line_slope = 0.0;
    int position_top = 0;
    bool detected;
    //std::vector<cv::Mat> recent_fits_pixel;
    //std::vector<cv::Mat> recent_fits_meter;
    std::deque<cv::Mat> recent_fits_pixel;
    std::deque<cv::Mat> recent_fits_meter;
    
};


//std::pair<Line, Line> get_fits_by_sliding_windows(IplImage *frame, Line line_lt, Line line_rt) {
void get_fits_by_sliding_windows(int processed_frames, IplImage *frame, Line &line_lt, Line &line_rt) {
    std::cout << "Detecting.................................................." << std::endl;
    ///////////////
    // Histogram //
    ///////////////
    CvSize frame_size = cvSize(frame->width, frame->height);
    cv::Mat bev_mat = cv::cvarrToMat(frame);
    bev_mat.convertTo(bev_mat, CV_32FC1); // Notice about data type
    cv::Mat Histogram(1, frame->width, CV_32FC1);
    cv::reduce(bev_mat, Histogram, 0, CV_REDUCE_SUM);
    //std::cout << Histogram << std::endl;
    int midpoint = frame->width / 2;
    double l, r;
    cv::Point base_l, base_r;
    cv::minMaxLoc(Histogram(cv::Rect(0, 0, midpoint, 1)), NULL, &l, NULL, &base_l);
    cv::minMaxLoc(Histogram(cv::Rect(midpoint, 0, midpoint, 1)), NULL, &r, NULL, &base_r);
    //std::cout << base_l.x << ", " << base_r.x + midpoint << std::endl;
    /////////////////////
    // Sliding Windows //
    /////////////////////
    int left_x_current = base_l.x, right_x_current = base_r.x + midpoint;
    float window_height = (frame->height) / 9; // height / number of windows
    std::vector<cv::Point> nz;
    bev_mat.convertTo(bev_mat, CV_8UC1);
    cv::findNonZero(bev_mat, nz);
    int margin = 50;  // width of the windows +/- margin
    int minpix = 50;  // minimum number of pixels found to recenter window
    //Create empty lists to receive left and right lane pixel indices
    std::vector<std::vector<cv::Point> > left_lane_points;
    std::vector<std::vector<cv::Point> > right_lane_points;

    IplImage *slide_window  = cvCreateImage(frame_size, IPL_DEPTH_8U, 3);
    cvCvtColor(frame, slide_window, CV_GRAY2BGR);
    float win_y_low = 0, win_y_high = 0, win_xleft_low = 0, win_xleft_high = 0, win_xright_low = 0, win_xright_high = 0;
    for (int i = 0; i < 9; ++i) {
        std::vector<cv::Point> good_left;
        std::vector<cv::Point> good_right;
        win_y_low = (frame->height) - (i + 1) * window_height;
        win_y_high = (frame->height) - i * window_height;
        win_xleft_low = left_x_current - margin;
        win_xleft_high = left_x_current + margin;
        win_xright_low = right_x_current - margin;
        win_xright_high = right_x_current + margin;
        cvRectangle(slide_window, CvPoint(win_xleft_low, win_y_low), CvPoint(win_xleft_high, win_y_high), CvScalar(255, 0, 0), 2);
        cvRectangle(slide_window, CvPoint(win_xright_low, win_y_low), CvPoint(win_xright_high, win_y_high), CvScalar(0, 0, 255), 2);
        for (int j = 0; j < nz.size(); ++j) {
            if (nz[j].x >= win_xleft_low && nz[j].x < win_xleft_high && nz[j].y >= win_y_low && nz[j].y < win_y_high) {
                good_left.push_back(nz[j]);
            }
            if (nz[j].x >= win_xright_low && nz[j].x < win_xright_high && nz[j].y >= win_y_low && nz[j].y < win_y_high) {
                good_right.push_back(nz[j]);
            }
        }
        left_lane_points.push_back(good_left);
        right_lane_points.push_back(good_right);
        // If you found > minpix pixels, recenter next window on their mean position
        int sum = 0;
        if (good_left.size() > minpix) {
            sum = 0;
            for (int j = 0; j < good_left.size(); ++j) {
                sum = sum + good_left[j].x;
            }
            left_x_current = int(sum / good_left.size());
        }
        if (good_right.size() > minpix) {
            sum = 0;
            for (int j = 0; j < good_right.size(); ++j) {
                sum = sum + good_right[j].x;
            }
            right_x_current = int(sum / good_right.size());
        }
        good_left.clear();
        good_right.clear();
    }
    //cvShowImage("Slide_Window", slide_window);
    //if (processed_frames == 20) {
    //    cvSaveImage("slide_window1.png", slide_window);
    //}
    
    //////////////////
    // Update Lines //
    //////////////////
    line_lt.all_point.clear();
    line_rt.all_point.clear();
    for (int i = 0; i < left_lane_points.size(); ++i) {
        line_lt.all_point.insert(line_lt.all_point.end(), left_lane_points[i].begin(), left_lane_points[i].end());
    }
    for (int i = 0; i < right_lane_points.size(); ++i) {
        line_rt.all_point.insert(line_rt.all_point.end(), right_lane_points[i].begin(), right_lane_points[i].end());
    }
    //std::cout << line_lt.all_point.size() << ", " << line_rt.all_point.size() << std::endl;
    bool detected = true;
    int poly_order = 1;
    cv::Mat left_fit_pixel;
    cv::Mat left_fit_meter;
    cv::Mat right_fit_pixel;
    cv::Mat right_fit_meter;
    
    //if (line_lt.all_point.empty()) {
    //if (line_lt.all_point.size() < 800 || (line_lt.position_top != 0 && abs(line_lt.position_top - line_lt.polyfit(1, 1, poly_order).at<double>(0, 0)) > 30)) {
    //std::cout << line_lt.all_point.size() << ", " << line_rt.all_point.size() << std::endl;
    if (line_lt.all_point.size() < 800) {
        left_fit_pixel = line_lt.last_fit_pixel;
        left_fit_meter = line_lt.last_fit_meter;
        detected = false;
    } else {
        left_fit_pixel = line_lt.polyfit(1, 1, poly_order);
        left_fit_meter = line_lt.polyfit(3.7 / 700, 30 / 720, poly_order);
    }
    //if (line_rt.all_point.empty()) {
    //if (line_rt.all_point.size() < 800 || (line_rt.position_top != 0 && abs(line_rt.position_top - line_rt.polyfit(1, 1, poly_order).at<double>(0, 0)) > 30)) {
    if (line_rt.all_point.size() < 800) {
        right_fit_pixel = line_rt.last_fit_pixel;
        right_fit_meter = line_rt.last_fit_meter;
        detected = false;
    } else {
        right_fit_pixel = line_rt.polyfit(1, 1, poly_order);
        right_fit_meter = line_rt.polyfit(3.7 / 700, 30 / 720, poly_order);
    }
    line_lt.update_line(left_fit_pixel, left_fit_meter, detected);
    line_rt.update_line(right_fit_pixel, right_fit_meter, detected);
    /////////////////////
    // Draw Poly Lines //
    /////////////////////
    /*
    for (int i = 0; i < (frame->height); i = i + 10) {
        CvPoint ipt_l;
        CvPoint ipt_r;
        ipt_l.y = i;
        ipt_r.y = i;
        ipt_l.x = 0;
        ipt_r.x = 0;
        for (int j = 0; j < poly_order + 1; ++j) {
            ipt_l.x += left_fit_pixel.at<double>(j, 0) * pow(i, j);
            ipt_r.x += right_fit_pixel.at<double>(j, 0) * pow(i, j);
        }
        cvCircle(slide_window, ipt_l, 5, CvScalar(255, 0, 0), CV_FILLED, CV_AA);
        cvCircle(slide_window, ipt_r, 5, CvScalar(0, 0, 255), CV_FILLED, CV_AA);
    }
    //cvShowImage("Detection Output", slide_window);
    */
    cvReleaseImage(&slide_window);
    nz.clear();
    left_lane_points.clear();
    right_lane_points.clear();

    bev_mat.release();
    Histogram.release();
    left_fit_pixel.release();
    left_fit_meter.release();
    right_fit_pixel.release();
    right_fit_meter.release();
    //return std::make_pair(line_lt, line_rt);
}

void get_fits_by_previous_fits(IplImage *frame, Line &line_lt, Line &line_rt) {
    std::cout << "Tracking..................................................." << std::endl;
    CvSize frame_size = cvSize(frame->width, frame->height);
    cv::Mat bev_mat = cv::cvarrToMat(frame);

    cv::Mat left_fit_pixel = line_lt.last_fit_pixel;
    cv::Mat right_fit_pixel = line_rt.last_fit_pixel;
    cv::Mat left_fit_meter;
    cv::Mat right_fit_meter;
    cv::Mat new_left_fit_pixel;
    cv::Mat new_right_fit_pixel;

    std::vector<cv::Point> nz;
    cv::findNonZero(bev_mat, nz);
    bool detected = false;
    int poly_order = 1;
    int margin = 100;
    int midpoint = frame->width / 2;
    line_lt.all_point.clear();
    line_rt.all_point.clear();
    int thresh_l1 = 0, thresh_l2 = 0, thresh_r1 = 0, thresh_r2 = 0;
    for (int i = 0; i < nz.size(); ++i) {
        
        for (int j = 0; j < poly_order + 1; ++j) {
            thresh_l1 += left_fit_pixel.at<double>(j, 0) * pow(nz[i].y, j);
            thresh_l2 += left_fit_pixel.at<double>(j, 0) * pow(nz[i].y, j);
            thresh_r1 += right_fit_pixel.at<double>(j, 0) * pow(nz[i].y, j);
            thresh_r2 += right_fit_pixel.at<double>(j, 0) * pow(nz[i].y, j);
        }
        thresh_l1 = thresh_l1 - margin;
        thresh_l2 = thresh_l2 + margin;
        thresh_r1 = thresh_r1 - margin;
        thresh_r2 = thresh_r2 + margin;

        if ((nz[i].x < midpoint) && (nz[i].x > thresh_l1) && (nz[i].x < thresh_l2)) {
            line_lt.all_point.push_back(nz[i]);
        }

        if ((nz[i].x > midpoint) && (nz[i].x > thresh_r1) && (nz[i].x < thresh_r2)) {
            line_rt.all_point.push_back(nz[i]);
        }

        thresh_l1 = 0;
        thresh_l2 = 0;
        thresh_r1 = 0;
        thresh_r2 = 0;
    }
    //std::cout << line_lt.all_point.size() << ", " << line_rt.all_point.size() << std::endl;
    
    ///////////////left_fit_pixel.at<double>(0, 0);
    //////////////////
    // Update Lines //
    //////////////////
    /*
    detected = true;
    if (line_lt.all_point.empty()) {
        left_fit_pixel = line_lt.last_fit_pixel;
        left_fit_meter = line_lt.last_fit_meter;
        detected = false;
    } else {
        left_fit_pixel = line_lt.polyfit(1, 1, poly_order);
        left_fit_meter = line_lt.polyfit(3.7 / 700, 30 / 720, poly_order);
    }
    if (line_rt.all_point.empty()) {
        right_fit_pixel = line_rt.last_fit_pixel;
        right_fit_meter = line_rt.last_fit_meter;
        detected = false;
    } else {
        right_fit_pixel = line_rt.polyfit(1, 1, poly_order);
        right_fit_meter = line_rt.polyfit(3.7 / 700, 30 / 720, poly_order);
    }
    */

    
    if ((!line_lt.all_point.empty()) && (!line_rt.all_point.empty())) {
        detected = true;
        new_left_fit_pixel = line_lt.polyfit(1, 1, poly_order);
        new_right_fit_pixel = line_rt.polyfit(1, 1, poly_order);
        if (line_lt.all_point.size() < 800 || abs(new_left_fit_pixel.at<double>(0, 0) - left_fit_pixel.at<double>(0, 0)) > 20) {
            left_fit_pixel = line_lt.last_fit_pixel;
            left_fit_meter = line_lt.last_fit_meter;
            detected = false;
        } else {
            left_fit_pixel = new_left_fit_pixel;
            left_fit_meter = line_lt.polyfit(3.7 / 700, 30 / 720, poly_order);
        }
        if (line_rt.all_point.size() < 800|| abs(new_right_fit_pixel.at<double>(0, 0) - right_fit_pixel.at<double>(0, 0)) > 20) {
            right_fit_pixel = line_rt.last_fit_pixel;
            right_fit_meter = line_rt.last_fit_meter;
            detected = false;
        } else {
            right_fit_pixel = new_right_fit_pixel;
            right_fit_meter = line_rt.polyfit(3.7 / 700, 30 / 720, poly_order);
        }
        line_lt.position_top = left_fit_pixel.at<double>(0, 0);
        line_rt.position_top = right_fit_pixel.at<double>(0, 0);
    }
    

    line_lt.update_line(left_fit_pixel, left_fit_meter, detected);
    line_rt.update_line(right_fit_pixel, right_fit_meter, detected);

    IplImage *output  = cvCreateImage(frame_size, IPL_DEPTH_8U, 3);
    cvCvtColor(frame, output, CV_GRAY2BGR);
    for (int i = 0; i < (frame->height); i = i + 10) {
        CvPoint ipt_l;
        CvPoint ipt_r;
        ipt_l.y = i;
        ipt_r.y = i;
        ipt_l.x = 0;
        ipt_r.x = 0;
        for (int j = 0; j < poly_order + 1; ++j) {
            ipt_l.x += left_fit_pixel.at<double>(j, 0) * pow(i, j);
            ipt_r.x += right_fit_pixel.at<double>(j, 0) * pow(i, j);
        }
        cvCircle(output, ipt_l, 5, CvScalar(255, 0, 0), CV_FILLED, CV_AA);
        cvCircle(output, ipt_r, 5, CvScalar(0, 0, 255), CV_FILLED, CV_AA);
    }
    //cvShowImage("Tracking Output", output);
    
    
    cvReleaseImage(&output);
    nz.clear();

    bev_mat.release();
    left_fit_pixel.release();
    left_fit_meter.release();
    new_left_fit_pixel.release();
    right_fit_pixel.release();
    right_fit_meter.release();
    new_right_fit_pixel.release();
}

#endif