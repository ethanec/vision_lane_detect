#ifndef _PARTICLEFILTER_H_
#define _PARTICLEFILTER_H_

#include <stdio.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <math.h>
#include <list>
#include <string>
#include "line_utils.h"
#include <numeric>

#define PI 3.14159265

class ParticleFilter {
public:
    ParticleFilter (std::string m)
    {
        method = m;
        polar_lt.create(2, 1, CV_64F);
        polar_rt.create(2, 1, CV_64F);
        polar.create(2, 1, CV_64F);
        predicted_line_lt.create(2, 1, CV_64F);
        predicted_line_rt.create(2, 1, CV_64F);
        predicted_line.create(2, 1, CV_64F);
        particle_num = 50;
        //particle.create(4, 1, CV_64F);
        //weight.create(2, 2, CV_64F);
        //weight_sum.create(2, 2, CV_64F);
        //particles.resize(particle_num, particle);
        //particles_weights.resize(particle_num, weight);
        //particles_weights_sum.resize(particle_num, weight_sum);
        particles.resize(particle_num);
        particles_weights.resize(particle_num);
        particles_weights_sum.resize(particle_num);
        x_predicted.create(4, 1, CV_64F);
        F = cv::Mat::eye(4, 4, CV_64F);
        w.create(4, 1, CV_64F);

        rho_0 = 0.0;
        rho_1 = 0.0;
        rho_2 = 0.0;
        theta_0 = 0.0; 
        theta_1 = 0.0; 
        theta_2 = 0.0;
    }

    ~ParticleFilter()
    {
        polar_lt.release();
        polar_rt.release();
        polar.release();
        predicted_line_lt.release();
        predicted_line_rt.release();
        predicted_line.release();
        
        //particle.release();
        //weight.release();
        //weight_sum.release();
        particles.clear();
        particles_weights.clear();
        particles_weights_sum.clear();
        to_delete.clear();
        x_predicted.release();
        F.release();
        w.release();

        rho_error.clear();
        theta_error.clear();
    }

    cv::Mat linear2polar(cv::Mat linear_line) {
        polar.at<double>(1, 0) = atan(-1 * linear_line.at<double>(1, 0)) * 180 / PI; // theta
        polar.at<double>(0, 0) = linear_line.at<double>(0, 0) * cos(polar.at<double>(1, 0) * PI / 180); // rho
        return polar;
    }

    cv::Mat polar2linear(cv::Mat polar_line) {
        predicted_line.at<double>(1, 0) = -1 * tan(polar_line.at<double>(1, 0) * PI / 180);
        predicted_line.at<double>(0, 0) = polar_line.at<double>(0, 0) / cos(polar_line.at<double>(1, 0) * PI / 180);
        return predicted_line;
    }

    double normal_distribution(double mean, double std) {
        double u = rand() / (double)RAND_MAX;
        double v = rand() / (double)RAND_MAX;
        double x = sqrt(-2 * log(u)) * cos(2 * M_PI * v) * std + mean;
        return x;
    }
    
    inline cv::Mat distance_transform(IplImage *frame) {
        IplImage* src = cvCloneImage(frame);
        IplImage* dst = cvCreateImage (cvGetSize(src), IPL_DEPTH_32F, 1);
        cvNot(src, src);
        cvDistTransform (src, dst, 2, 5);
        IplImage* result = cvCreateImage (cvGetSize(src), IPL_DEPTH_8U, 1);
        cvConvertScale(dst, result);
        
        cv::Mat result_mat = cv::cvarrToMat(result);
        //cv::Mat tmp_m, tmp_sd;
        //result_mat.convertTo(result_mat, CV_64F, 1 / 255.0);
        //cv::meanStdDev(result_mat, tmp_m, tmp_sd);
        //R_d = tmp_sd.at<double>(0, 0);

        cvReleaseImage(&src);
        cvReleaseImage(&dst);
        cvReleaseImage(&result);
        //tmp_m.release();
        //tmp_sd.release();
        
        return result_mat;
    }

    void bev_particle_filter(IplImage *frame, Line &line_lt, Line &line_rt, int poly_order, int processed_frames) {
        CvSize frame_size = cvSize(frame->width, frame->height);
        linear2polar(line_lt.last_fit_pixel).copyTo(polar_lt);
        linear2polar(line_rt.last_fit_pixel).copyTo(polar_rt);

        IplImage *c3 = cvCreateImage(frame_size, IPL_DEPTH_8U, 3);
        cvCvtColor(frame, c3, CV_GRAY2BGR);

        /*
        // mean and stddev of rho & theta //
        if (processed_frames < 1002) {
            
            if (processed_frames == 0) {
                rho_0 = polar_lt.at<double>(0, 0);
                theta_0 = polar_lt.at<double>(1, 0);
            }
            if (processed_frames == 1) {
                rho_1 = polar_lt.at<double>(0, 0);
                theta_1 = polar_lt.at<double>(1, 0);
            }
            if (processed_frames > 1) {
                rho_2 = polar_lt.at<double>(0, 0);
                theta_2 = polar_lt.at<double>(1, 0);

                rho_error.push_back(rho_2 - (2 * rho_1 - rho_0));
                theta_error.push_back(theta_2 - (2 * theta_1 - theta_0));

                rho_0 = rho_1;
                theta_0 = theta_1;
                rho_1 = rho_2;
                theta_1 = theta_2;
            }
        }
        
        if (processed_frames >= 1002) {
            
            double rho_sum = std::accumulate(rho_error.begin(), rho_error.end(), 0.0);
            double rho_mean =  rho_sum / rho_error.size(); //均值
        
            double rho_accum  = 0.0;
            for (int n = 0; n < rho_error.size(); ++n) {
                rho_accum  += (rho_error[n] - rho_mean) * (rho_error[n] - rho_mean);
            }
        
            double rho_stdev = sqrt(rho_accum / (rho_error.size() - 1)); //方差
            std::cout << "Rho Mean: " << rho_mean << " Rho Std: " << rho_stdev << std::endl; // Rho Mean: -0.04762Rho Std: 14.3215

            double theta_sum = std::accumulate(theta_error.begin(), theta_error.end(), 0.0);
            double theta_mean =  theta_sum / theta_error.size(); //均值
        
            double theta_accum  = 0.0;
            for (int n = 0; n < theta_error.size(); ++n) {
                theta_accum  += (theta_error[n] - theta_mean) * (theta_error[n] - theta_mean);
            }
        
            double theta_stdev = sqrt(theta_accum / (theta_error.size() - 1)); //方差
            std::cout << "Theta Mean: " << theta_mean << " Theta Std: " << theta_stdev << std::endl; // Theta Mean: -0.0179372Theta Std: 2.95042
        }
        */
        if (processed_frames == 0) {
            
            for (int n = 0; n < particle_num; ++n) {
                cv::Mat particle(4, 1, CV_64F);
                particle = 0;
                particle.at<double>(0, 0) = normal_distribution(polar_lt.at<double>(0, 0), 24.5017); // 14.3215
                particle.at<double>(1, 0) = normal_distribution(polar_lt.at<double>(1, 0), 2.95042); // 2.95042
                particle.at<double>(2, 0) = normal_distribution(polar_rt.at<double>(0, 0), 24.5017); // 14.3215
                particle.at<double>(3, 0) = normal_distribution(polar_rt.at<double>(1, 0), 2.95042); // 2.95042
                //std::cout << particle << std::endl;
                particle.copyTo(particles[n]);
                particle.release();
            }
            
        } else {
            //cv::Mat s_frame = cv::cvarrToMat(frame);
            //cv::Mat d_frame = distance_transform(frame);

            //CvScalar s_mean, s_StdDev;
            //cvAvgSdv(frame, &s_mean, &s_StdDev);
            //R_bw = s_StdDev.val[0];
            
            //if (processed_frames == 1) {
            //    std::cout << d_frame << std::endl;
            //}
            double l_weights_sum = 0.0, r_weights_sum = 0.0;
            for (int n = 0; n < particle_num; ++n) {
                cv::Mat weight(2, 2, CV_64F);
                weight = 0;
                ////////////////
                // Prediction //
                ////////////////
                for (int i = 0; i < 2; ++i) {
                    w.at<double>(i * 2, 0) = normal_distribution(0, 122.234); // 14.3215, 122.234
                    w.at<double>(i * 2 + 1, 0) = normal_distribution(0, 24.5017); // 2.95042, 24.5017
                }
                x_predicted = F * particles[n] + w;
                /////////////////////////
                // Weights Calculating //
                /////////////////////////
                polar2linear(x_predicted(cv::Rect(0, 0, 1, 2))).copyTo(predicted_line_lt);
                polar2linear(x_predicted(cv::Rect(0, 2, 1, 2))).copyTo(predicted_line_rt);
                /*
                for (int i = 0; i < (frame->height); i = i + 1) {
                    CvPoint ipt_l;
                    CvPoint ipt_r;
                    ipt_l.y = i;
                    ipt_r.y = i;
                    ipt_l.x = 0;
                    ipt_r.x = 0;
                    for (int j = 0; j < poly_order + 1; ++j) {
                        ipt_l.x += predicted_line_lt.at<double>(j, 0) * pow(i, j);
                        ipt_r.x += predicted_line_rt.at<double>(j, 0) * pow(i, j);
                    }
                    cvCircle(c3, ipt_l, 3, CvScalar(255, 0, 0), CV_FILLED, CV_AA);
                    cvCircle(c3, ipt_r, 3, CvScalar(0, 0, 255), CV_FILLED, CV_AA);

                }
                cvShowImage ("c3", c3);
                */
                //weight.at<double>(0, 0) = 1 / sqrt(pow(line_lt.last_fit_pixel.at<double>(0, 0) - predicted_line_lt.at<double>(0, 0), 2) \
                                                 * pow(line_lt.last_fit_pixel.at<double>(1, 0) - predicted_line_lt.at<double>(1, 0), 2));
                //weight.at<double>(1, 1) = 1 / sqrt(pow(line_rt.last_fit_pixel.at<double>(0, 0) - predicted_line_rt.at<double>(0, 0), 2) \
                                                 * pow(line_rt.last_fit_pixel.at<double>(1, 0) - predicted_line_rt.at<double>(1, 0), 2));
                weight.at<double>(0, 0) = (1 + line_lt.last_fit_pixel.at<double>(1, 0) * predicted_line_lt.at<double>(1, 0)) \
                                        / (sqrt(1 + pow(line_lt.last_fit_pixel.at<double>(1, 0), 2)) * sqrt(1 + pow(predicted_line_lt.at<double>(1, 0), 2))) \
                                        / std::abs(line_lt.last_fit_pixel.at<double>(0, 0) - predicted_line_lt.at<double>(0, 0));
                weight.at<double>(1, 1) = (1 + line_rt.last_fit_pixel.at<double>(1, 0) * predicted_line_rt.at<double>(1, 0)) \
                                        / (sqrt(1 + pow(line_rt.last_fit_pixel.at<double>(1, 0), 2)) * sqrt(1 + pow(predicted_line_rt.at<double>(1, 0), 2))) \
                                        / std::abs(line_rt.last_fit_pixel.at<double>(0, 0) - predicted_line_rt.at<double>(0, 0));

                weight.copyTo(particles_weights[n]);
                //std::cout << weight << std::endl;
                l_weights_sum = l_weights_sum + weight.at<double>(0, 0);
                r_weights_sum = r_weights_sum + weight.at<double>(1, 1);

                weight.release();
            }
            ///////////////////
            // Normalization //
            ///////////////////
            for (int n = 0; n < particle_num; ++n) {
                particles_weights[n].at<double>(0, 0) = particles_weights[n].at<double>(0, 0) / l_weights_sum;
                particles_weights[n].at<double>(1, 1) = particles_weights[n].at<double>(1, 1) / r_weights_sum;
                //std::cout << particles_weights[n] << std::endl;
            }
            ////////////////
            // Estimation //
            ////////////////
            cv::Mat estimate_poly(4, 1, CV_64F);
            estimate_poly = 0;
            for (int n = 0; n < particle_num; ++n) {
                cv::Mat expand_weight(4, 4, CV_64F);
                expand_weight = 0;
                expand_weight.at<double>(0, 0) = particles_weights[n].at<double>(0, 0);
                expand_weight.at<double>(1, 1) = particles_weights[n].at<double>(0, 0);
                expand_weight.at<double>(2, 2) = particles_weights[n].at<double>(1, 1);
                expand_weight.at<double>(3, 3) = particles_weights[n].at<double>(1, 1);

                estimate_poly = estimate_poly + expand_weight * particles[n];

                polar2linear(estimate_poly(cv::Rect(0, 0, 1, 2))).copyTo(line_lt.last_fit_pixel);
                polar2linear(estimate_poly(cv::Rect(0, 2, 1, 2))).copyTo(line_rt.last_fit_pixel);

                expand_weight.release();
            }
            estimate_poly.release();
            ////////////////
            // Resampling //
            ////////////////
            int copy_idx = 0;
            double u = std::abs(normal_distribution(0, 1 / double(particle_num) * 0.01)); // 0.01, 0.001
            //std::cout << "u: " << u << std::endl;
            cv::Mat weights_sum(2, 2, CV_64F);
            weights_sum = 0;
            weights_sum.at<double>(0, 0) = particles_weights[0].at<double>(0, 0);
            weights_sum.at<double>(1, 1) = particles_weights[0].at<double>(1, 1);
            weights_sum.copyTo(particles_weights_sum[0]);
            weights_sum.release();
            for (int n = 1; n < particle_num; ++n) {
                cv::Mat weights_sum(2, 2, CV_64F);
                weights_sum = 0;
                weights_sum.at<double>(0, 0) = particles_weights_sum[n - 1].at<double>(0, 0) + particles_weights[n].at<double>(0, 0);
                weights_sum.at<double>(1, 1) = particles_weights_sum[n - 1].at<double>(1, 1) + particles_weights[n].at<double>(1, 1);
                weights_sum.copyTo(particles_weights_sum[n]);
                weights_sum.release();
            }
            for (int n = 0; n < particle_num; ++n) {
                //std::cout << "n: " << n << std::endl;
                //std::cout << particles_weights_sum[n] << std::endl;
                //std::cout << "u: " << u << std::endl;
                //std::cout << "C_I: " << copy_idx << std::endl;
                while ((particles_weights_sum[copy_idx].at<double>(0, 0) < u || particles_weights_sum[copy_idx].at<double>(1, 1) < u) && copy_idx < 50) {
                    to_delete.push_back(copy_idx);
                    copy_idx = copy_idx + 1;
                }
                
                //std::cout << "Delete: " << to_delete.size() << std::endl;
                
                for (int i = 0; i < to_delete.size(); ++i) {
                    if (copy_idx < 50) {
                        particles[ copy_idx ].copyTo(particles[ to_delete[i] ]);
                    }
                }
                u = u + 1 / double(particle_num) * 0.01; // 0.01, 0.001
                to_delete.clear();
            }
            //for (int n = 0; n < particle_num; ++n) {
            //    std::cout << particles[n].at<double>(0, 0) << ", " << particles[n].at<double>(2, 0) << std::endl;
            //}

            //s_frame.release();
            //d_frame.release();
            
        }

        cvReleaseImage(&c3);
    }
private:
    cv::Mat polar_lt, polar_rt;
    cv::Mat polar;
    cv::Mat predicted_line_lt, predicted_line_rt;
    cv::Mat predicted_line;
    int particle_num;
    //cv::Mat particle;
    //cv::Mat weight;
    //cv::Mat weight_sum;
    std::vector<cv::Mat> particles;
    std::vector<cv::Mat> particles_weights;
    std::vector<cv::Mat> particles_weights_sum;
    std::vector<int> to_delete;
    
    cv::Mat x_predicted;
    cv::Mat F;// transition matrix
    cv::Mat w;// process noise
    // measurement noise

    std::string method;

    double rho_0, rho_1, rho_2;
    std::vector<double> rho_error;
    double theta_0, theta_1, theta_2;
    std::vector<double> theta_error;
};
#endif