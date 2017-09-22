//created by linyan.wang @ 2017-6-28

#ifndef ULSEE_LIFT_H
#define ULSEE_LIFT_H

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <math.h>
#include <fstream>
#include "ulsee_poseSDK.h"
#include <set>


class Get_2DPose {
public:
    Get_2DPose() {};

    ~Get_2DPose() {};

    cv::Mat get2DPos(cv::Mat &frame);

private:
    cv::Mat prediction_mul;
    ULSee_Pose pos;

};

#endif// ULSEE_LIFT_H e() {};
