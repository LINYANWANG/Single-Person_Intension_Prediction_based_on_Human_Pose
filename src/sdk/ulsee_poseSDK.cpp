//created by lampson.song @ 2017-4-18
#include "ulsee_poseSDK.h"

#include "../algorithm/rtpose/rtpose.h"
#include <opencv2/imgproc/imgproc.hpp>

#include "ULSeeEncryption.h"

#include <fstream>

RTPose *rtpose;
const auto NUM_PARTS = 18;

int cnt = 0;
double x_scale,y_scale;

std::vector<cv::Point> jointsPos;

static RTPose::RTPoseParameter param = {
	0,
	1280,
	720,
	256,
	144,
//    192,
//    160,
	"pose_deploy_linevec.prototxt",
	"pose_iter_440000.caffemodel",
	"",
	NULL,
	NULL,
};

std::string model_name = "./model/model.bin";

std::string key;

template<typename CharT, typename TraitsT = std::char_traits<CharT> >
class vectorwrapbuf : public std::basic_streambuf<CharT, TraitsT> {
  public:
	    vectorwrapbuf(std::vector<CharT> &vec) 
		{
		  std::streambuf::setg(vec.data(), vec.data(), vec.data() + vec.size());
		}
};

class CheckLicense{
  public: 
	CheckLicense(){};
	static void verify(int &flag)
	{
	    fstream in;
		in.open("./key.txt",std::ios::in);
		
		if(!in.is_open())
	  	{
			std::cout<<"Fail to read key file"<<std::endl;
			exit(1);
		}
		
		// read the first line of key.txt
		getline(in,key);	
		in.close();

	}
};

ULSee_Pose::ULSee_Pose()
{
  	int k_flag=1;
//   	CheckLicense::verify(k_flag); 

	if(k_flag)
	{	  
		ULSeeEncryption ule;
		auto decoderFiles = ule.doDecoder(model_name);

		std::vector<char> a = decoderFiles[param.caffe_proto];
		std::vector<char> b = decoderFiles[param.caffe_model];
		
		vectorwrapbuf<char> databuf(a);
		std::istream is(&databuf);
		param.proto_pointer = &is;

		vectorwrapbuf<char> modelbuf(b);
		std::istream b_is(&modelbuf);
		param.model_pointer = &b_is;
		
		rtpose = new RTPose(param);
	}
	else
	{
		std::cout<<"The license is out of date."<<std::endl;
	}
}


ULSee_Pose::ULSee_Pose(std::string input_file)
{
	
	int k_flag;
   	CheckLicense::verify(k_flag); 

	if(k_flag)
	{	  
		ULSeeEncryption ule;
		auto decoderFiles = ule.doDecoder(model_name);

		std::vector<char> a = decoderFiles[param.caffe_proto];
		std::vector<char> b = decoderFiles[param.caffe_model];
		
		vectorwrapbuf<char> databuf(a);
		std::istream is(&databuf);
		param.proto_pointer = &is;

		vectorwrapbuf<char> modelbuf(b);
		std::istream b_is(&modelbuf);
		param.model_pointer = &b_is;
		
		// set the input file name
  		param.input_file = input_file;
		
		rtpose = new RTPose(param);
	}
	else
	{
		std::cout<<"The license is out of date."<<std::endl;
	}
}

void ULSee_Pose::processFrame(cv::Mat frame)
{
    jointsPos.clear();
	x_scale = double(frame.cols)/1280;
	y_scale = double(frame.rows)/720;
	cv::resize(frame,frame,cv::Size(1280,720));
    rtpose->processFrame(frame,jointsPos,cnt);

//	cv::imshow("in",frame);
//	cv::waitKey(1);
}

ULSee_Pose::~ULSee_Pose()
{
  delete rtpose;
}

void ULSee_Pose::getNosePos(int idx, int &x, int &y)
{
  if(idx<cnt)
  {
	x = jointsPos.at(0+idx*NUM_PARTS).x*x_scale;
	y = jointsPos.at(0+idx*NUM_PARTS).y*y_scale;
  }
  else
  {
  	x = 0;
	y = 0;
  }
}




void ULSee_Pose::getNeckPos(int idx, int &x, int &y)
{
  if(idx<cnt)
  {
	x = jointsPos.at(1+idx*NUM_PARTS).x*x_scale;
	y = jointsPos.at(1+idx*NUM_PARTS).y*y_scale;
  }
  else
  {
  	x = 0;
	y = 0;
  }
}




void ULSee_Pose::getRightHipPos(int idx, int &x, int &y)
{
  if(idx<cnt)
  {
	x = jointsPos.at(8+idx*NUM_PARTS).x*x_scale;
	y = jointsPos.at(8+idx*NUM_PARTS).y*y_scale;
  }
  else
  {
  	x = 0;
	y = 0;
  }
}





void ULSee_Pose::getLeftHipPos(int idx, int &x, int &y)
{
  if(idx<cnt)
  {
	x = jointsPos.at(11+idx*NUM_PARTS).x*x_scale;
	y = jointsPos.at(11+idx*NUM_PARTS).y*y_scale;
  }
  else
  {
  	x = 0;
	y = 0;
  }
}


void ULSee_Pose::getRightShoulderPos(int idx, int &x, int &y)
{
  if(idx<cnt)
  {
	x = jointsPos.at(2+idx*NUM_PARTS).x*x_scale;
	y = jointsPos.at(2+idx*NUM_PARTS).y*y_scale;
  }
  else
  {
  	x = 0;
	y = 0;
  }
}

void ULSee_Pose::getLeftShoulderPos(int idx, int &x, int &y)
{
  if(idx<cnt)
  {
	x = jointsPos.at(5+idx*NUM_PARTS).x*x_scale;
	y = jointsPos.at(5+idx*NUM_PARTS).y*y_scale;
  }
  else
  {
  	x = 0;
	y = 0;
  }
}



void ULSee_Pose::getRightWristPos(int idx, int &x, int &y)
{
  if(idx<cnt)
  {
	x = jointsPos.at(4+idx*NUM_PARTS).x*x_scale;
	y = jointsPos.at(4+idx*NUM_PARTS).y*y_scale;
  }
  else
  {
  	x = 0;
	y = 0;
  }
}

void ULSee_Pose::getLeftWristPos(int idx, int &x, int &y)
{
  if(idx<cnt)
  {
	x = jointsPos.at(7+idx*NUM_PARTS).x*x_scale;
	y = jointsPos.at(7+idx*NUM_PARTS).y*y_scale;
  }
  else
  {
  	x = 0;
	y = 0;
  }
}




void ULSee_Pose::getRightElbowPos(int idx, int &x, int &y)
{
  if(idx<cnt)
  {
	x = jointsPos.at(3+idx*NUM_PARTS).x*x_scale;
	y = jointsPos.at(3+idx*NUM_PARTS).y*y_scale;
  }
  else
  {
  	x = 0;
	y = 0;
  }
}

void ULSee_Pose::getLeftElbowPos(int idx, int &x, int &y)
{
  if(idx<cnt)
  {
	x = jointsPos.at(6+idx*NUM_PARTS).x*x_scale;
	y = jointsPos.at(6+idx*NUM_PARTS).y*y_scale;
  }
  else
  {
  	x = 0;
	y = 0;
  }
}








void ULSee_Pose::getRightKneePos(int idx, int &x, int &y)
{
  if(idx<cnt)
  {
	x = jointsPos.at(9+idx*NUM_PARTS).x*x_scale;
	y = jointsPos.at(9+idx*NUM_PARTS).y*y_scale;
  }
  else
  {
  	x = 0;
	y = 0;
  }
}

void ULSee_Pose::getLeftKneePos(int idx, int &x, int &y)
{
  if(idx<cnt)
  {
	x = jointsPos.at(12+idx*NUM_PARTS).x*x_scale;
	y = jointsPos.at(12+idx*NUM_PARTS).y*y_scale;
  }
  else
  {
  	x = 0;
	y = 0;
  }
}






void ULSee_Pose::getRightAnklePos(int idx, int &x, int &y)
{
  if(idx<cnt)
  {
	x = jointsPos.at(10+idx*NUM_PARTS).x*x_scale;
	y = jointsPos.at(10+idx*NUM_PARTS).y*y_scale;
  }
  else
  {
  	x = 0;
	y = 0;
  }
}

void ULSee_Pose::getLeftAnklePos(int idx, int &x, int &y)
{
  if(idx<cnt)
  {
	x = jointsPos.at(13+idx*NUM_PARTS).x*x_scale;
	y = jointsPos.at(13+idx*NUM_PARTS).y*y_scale;
  }
  else
  {
  	x = 0;
	y = 0;
  }
}





void ULSee_Pose::getRightEyePos(int idx, int &x, int &y)
{
  if(idx<cnt)
  {
	x = jointsPos.at(14+idx*NUM_PARTS).x*x_scale;
	y = jointsPos.at(14+idx*NUM_PARTS).y*y_scale;
  }
  else
  {
  	x = 0;
	y = 0;
  }
}

void ULSee_Pose::getLeftEyePos(int idx, int &x, int &y)
{
  if(idx<cnt)
  {
	x = jointsPos.at(15+idx*NUM_PARTS).x*x_scale;
	y = jointsPos.at(15+idx*NUM_PARTS).y*y_scale;
  }
  else
  {
  	x = 0;
	y = 0;
  }
}





void ULSee_Pose::getRightEarPos(int idx, int &x, int &y)
{
  if(idx<cnt)
  {
	x = jointsPos.at(16+idx*NUM_PARTS).x*x_scale;
	y = jointsPos.at(16+idx*NUM_PARTS).y*y_scale;
  }
  else
  {
  	x = 0;
	y = 0;
  }
}

void ULSee_Pose::getLeftEarPos(int idx, int &x, int &y)
{
  if(idx<cnt)
  {
	x = jointsPos.at(17+idx*NUM_PARTS).x*x_scale;
	y = jointsPos.at(17+idx*NUM_PARTS).y*y_scale;
  }
  else
  {
  	x = 0;
	y = 0;
  }
}

void ULSee_Pose::getDetectedPeopleNumber(int &num)
{
	num = cnt;	
}
