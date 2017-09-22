// created by lampson.song @ 2017-4-18

#ifndef ULSEE_POSEE_H
#define ULSEE_POSE_H

#include <string>
#include <opencv2/core/core.hpp>

class ULSee_Pose
{
	public:
	  ULSee_Pose();
	  ULSee_Pose(std::string input_file);
	  ~ULSee_Pose();

	  void getNosePos(int idx, int &x, int &y);
	  
	  void getRightEyePos(int idx, int &x, int &y);
	  void getLeftEyePos(int idx, int &x, int &y);
	  
	  void getRightEarPos(int idx, int &x, int &y);
	  void getLeftEarPos(int idx, int &x, int &y);
	  
	  void getNeckPos(int idx, int &x, int &y);
	  
	  void getRightElbowPos(int idx, int &x, int &y);
	  void getLeftElbowPos(int idx, int &x, int &y);
	  
	  void getRightShoulderPos(int idx, int &x, int &y);
	  void getLeftShoulderPos(int idx, int &x, int &y);
	  
	  void getRightWristPos(int idx, int &x, int &y);
	  void getLeftWristPos(int idx, int &x, int &y);
	  
	  void getRightHipPos(int idx, int &x, int &y);
	  void getLeftHipPos(int idx, int &x, int &y);
	  
	  void getRightKneePos(int idx, int &x, int &y);
	  void getLeftKneePos(int idx, int &x, int &y);
	  
	  void getRightAnklePos(int idx, int &x, int &y);
	  void getLeftAnklePos(int idx, int &x, int &y);

	  void getDetectedPeopleNumber(int &num);

	  void processFrame(cv::Mat frame);
		

};

#endif// ULSEE_POSEE_H 
