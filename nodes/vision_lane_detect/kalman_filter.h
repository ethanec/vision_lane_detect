#ifndef _KALMANFILTER_H_
#define _KALMANFILTER_H_

#include <stdio.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <math.h>
#include <list>
#include <string>
#include "line_utils.h"

class KalmanFilter {
public:
    KalmanFilter (std::string m)
    {
        method = m;

        mat_x.create(8, 1, CV_64F);
        _mat_x.create(8, 1, CV_64F);
        __mat_x.create(8, 1, CV_64F);
        mat_x_.create(8, 1, CV_64F);
        mat_p_.create(8, 8, CV_64F);
        Kg.create(8, 8, CV_64F);
        new_mat_x.create(8, 1, CV_64F);
        new_mat_p.create(8, 8, CV_64F);
        F = cv::Mat::eye(8, 8, CV_64F) * 2;
        //F = cv::Mat::eye(8, 8, CV_64F);
        H = cv::Mat::eye(8, 8, CV_64F);
        P = cv::Mat::eye(8, 8, CV_64F);
        //Q = cv::Mat::eye(8, 8, CV_64F);
        Q = cv::Mat::eye(8, 8, CV_64F) * 1.5;
        R = cv::Mat::eye(8, 8, CV_64F) * 2.5;
        I = cv::Mat::eye(8, 8, CV_64F);
    }

    ~KalmanFilter()
    {
        mat_x.release();
        _mat_x.release();
        __mat_x.release();
        mat_x_.release();
        mat_p_.release();
        Kg.release();
        new_mat_x.release();
        new_mat_p.release();
        F.release();
        H.release();
        P.release();
        Q.release();
        R.release();
        I.release();
    }

    void bev_kalman_filter(IplImage *frame, Line &line_lt, Line &line_rt, int poly_order, int processed_frames, std::string detection_mode) {
        CvSize frame_size = cvSize(frame->width, frame->height);
        
        int s = 0;
        if (method == "BEV") {
            /////////////////////////////
            // Define 8 Control Points //
            /////////////////////////////
            for (int i = 0; i < (frame->height); i = i + int(frame->height / 3)) {
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
                mat_x.at<double>(s * 2, 0) = ipt_l.x;
                mat_x.at<double>(s * 2 + 1, 0) = ipt_r.x;
                s++;
            }

            if (processed_frames > 1) {
                ///////////////////
                // Kalman Filter //
                ///////////////////
                mat_x_ = F * (_mat_x - 0.5 * __mat_x);
                //mat_x_ = F * _mat_x;
                mat_p_ = F * P * F.t() + Q;
                
                for (int i = 0; i < 4; ++i) {
                    if (abs(mat_x.at<double>(i * 2, 0) - mat_x_.at<double>(i * 2, 0)) > 30 && detection_mode == "pf") {
                        mat_x.at<double>(i * 2, 0) = mat_x_.at<double>(i * 2, 0);
                        R.at<double>(i * 2, i * 2) = 1;
                    }
                    if (abs(mat_x.at<double>(i * 2 + 1, 0) - mat_x_.at<double>(i * 2 + 1, 0)) > 30 && detection_mode == "pf") {
                        mat_x.at<double>(i * 2 + 1, 0) = mat_x_.at<double>(i * 2 + 1, 0);
                        R.at<double>(i * 2 + 1, i * 2 + 1) = 1;
                    }

                    /*
                    mat_x.at<double>(i * 2, 0) = (abs(mat_x.at<double>(i * 2, 0) - mat_x_.at<double>(i * 2, 0)) < 100) \
                                                    ? mat_x.at<double>(i * 2, 0) : mat_x_.at<double>(i * 2, 0);
                    mat_x.at<double>(i * 2 + 1, 0) = (abs(mat_x.at<double>(i * 2 + 1, 0) - mat_x_.at<double>(i * 2 + 1, 0)) < 80) \
                                                        ? mat_x.at<double>(i * 2 + 1, 0) : mat_x_.at<double>(i * 2 + 1, 0);

                    R.at<double>(i * 2, i * 2) = 1;
                    R.at<double>(i * 2 + 1, i * 2 + 1) = 1;
                    */
                }
                
                Kg = mat_p_ * H.t() * ((H * mat_p_ * H.t() + R).inv());

                new_mat_x = mat_x_ + Kg * (mat_x - H * mat_x_);
                new_mat_p = (I - Kg * H) * mat_p_;
                
                //std::cout << "Predicted state: " << mat_x_ << std::endl;
                //std::cout << "Predicted covariance: " << mat_p_ << std::endl;
                //std::cout << "Measurement: " << mat_x << std::endl;
                //std::cout << "Kalman gain: " << Kg << std::endl;
                //std::cout << "Updated state: " << new_mat_x << std::endl;
                //std::cout << "Updated covariance: " << new_mat_p << std::endl;
                
                ///////////////////////
                // Update Polynomial //
                ///////////////////////
                line_lt.all_point.clear();
                line_rt.all_point.clear();
                for (int i = 0; i < 4; ++i) {
                    line_lt.all_point.push_back(cv::Point(int(new_mat_x.at<double>(i * 2, 0)), int(frame->height / 3) * i));
                    line_rt.all_point.push_back(cv::Point(int(new_mat_x.at<double>(i * 2 + 1, 0)), int(frame->height / 3) * i));
                }
                line_lt.last_fit_pixel = line_lt.polyfit(1, 1, 1);
                line_rt.last_fit_pixel = line_rt.polyfit(1, 1, 1);


                //_mat_x.copyTo(__mat_x);
                //new_mat_x.copyTo(_mat_x);
                //new_mat_p.copyTo(P);
                __mat_x = _mat_x;
                _mat_x = new_mat_x;
                P = new_mat_p;
            } else {
                _mat_x.copyTo(__mat_x);
                mat_x.copyTo(_mat_x);
            }
            
            
        }
    }
private:
    cv::Mat mat_x;// observed state
    cv::Mat _mat_x;// previous state
    cv::Mat __mat_x;// previous previous state
    cv::Mat mat_x_;// predicted state
    cv::Mat mat_p_;// predicted covariance
    cv::Mat Kg;// Kalman gain
    cv::Mat new_mat_x;// updated state
    cv::Mat new_mat_p;// updated covariance
    cv::Mat F;// transition matrix
    cv::Mat H;// observation matrix
    cv::Mat P;// updated covariance
    cv::Mat Q;// noise covariance
    cv::Mat R;// measurement covariance
    cv::Mat I;

    std::string method;
};
#endif