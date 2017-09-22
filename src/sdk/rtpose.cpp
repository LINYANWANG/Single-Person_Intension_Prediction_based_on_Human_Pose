//craeted by lampson.song @ 2017-4-18
 
#include "rtpose.h"

#include "algorithm/rtpose/rtpose.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

#include <gflags/gflags.h>

#include <math.h>
#include "PythonToOCV.h"
#include <fstream>
#include <boost/python.hpp>
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
//#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

using namespace std;
using namespace cv;
namespace py = boost::python;

//DEFINE_int32(mode,0,"testing mode: 0 for camera, 1 for video, 2 for image");
//DEFINE_int32(idx,0,"default camera index is 0");
//DEFINE_string(t_image,"./testData/test.jpeg","test image path and name");
//DEFINE_string(t_video,"./testData/testVideo.avi","test video path and name");

cv::Mat Get_2DPose::get2DPos(cv::Mat &frame){
	int people_num = 0;

    std::cout<<"----start----:"<<std::endl;
		
	pos.processFrame(frame);

	pos.getDetectedPeopleNumber(people_num);
	cv::Mat prediction_mul= cv::Mat::zeros(people_num, 36, CV_64F);
	std::cout << "----people_num----:" << people_num << std::endl;

	for(int idx=0; idx <people_num; idx++)
	{
		int x,y;
		pos.getNosePos(idx,x,y);
                prediction_mul.at<double>(idx, 0) = x;
       	        prediction_mul.at<double>(idx, 1) = y;

		// cv::circle(frame,cv::Point(x,y),1,cv::Scalar(0,0,255),2,8,0);
			
		pos.getNeckPos(idx,x,y);
                prediction_mul.at<double>(idx, 2) = x;
                prediction_mul.at<double>(idx, 3) = y;

		// cv::circle(frame,cv::Point(x,y),1,cv::Scalar(0,0,255),2,8,0);

		pos.getRightHipPos(idx,x,y);
                prediction_mul.at<double>(idx, 4) = x;
                prediction_mul.at<double>(idx, 5) = y;

		// cv::circle(frame,cv::Point(x,y),1,cv::Scalar(0,0,255),2,8,0);

		pos.getLeftHipPos(idx,x,y);
                prediction_mul.at<double>(idx, 6) = x;
                prediction_mul.at<double>(idx, 7) = y;

		// cv::circle(frame,cv::Point(x,y),1,cv::Scalar(0,0,255),2,8,0);
			
		pos.getRightShoulderPos(idx,x,y);
                prediction_mul.at<double>(idx, 8) = x;
                prediction_mul.at<double>(idx, 9) = y;

		// cv::circle(frame,cv::Point(x,y),1,cv::Scalar(0,0,255),2,8,0);
			
		pos.getLeftShoulderPos(idx,x,y);
                prediction_mul.at<double>(idx, 10) = x;
                prediction_mul.at<double>(idx, 11) = y;

		// cv::circle(frame,cv::Point(x,y),1,cv::Scalar(0,0,255),2,8,0);
			
		pos.getRightElbowPos(idx,x,y);
                prediction_mul.at<double>(idx, 12) = x;
                prediction_mul.at<double>(idx, 13) = y;

		// cv::circle(frame,cv::Point(x,y),1,cv::Scalar(0,0,255),2,8,0);
			
		pos.getLeftElbowPos(idx,x,y);
                prediction_mul.at<double>(idx, 14) = x;
                prediction_mul.at<double>(idx, 15) = y;

		// cv::circle(frame,cv::Point(x,y),1,cv::Scalar(0,0,255),2,8,0);
			
		pos.getRightWristPos(idx,x,y);
                prediction_mul.at<double>(idx, 16) = x;
                prediction_mul.at<double>(idx, 17) = y;

		// cv::circle(frame,cv::Point(x,y),1,cv::Scalar(0,0,255),2,8,0);
			
		pos.getLeftWristPos(idx,x,y);
                prediction_mul.at<double>(idx, 18) = x;
                prediction_mul.at<double>(idx, 19) = y;

		// cv::circle(frame,cv::Point(x,y),1,cv::Scalar(0,0,255),2,8,0);
				
		pos.getRightKneePos(idx,x,y);
                prediction_mul.at<double>(idx, 20) = x;
                prediction_mul.at<double>(idx, 21) = y;

		// cv::circle(frame,cv::Point(x,y),1,cv::Scalar(0,0,255),2,8,0);
			
		pos.getLeftKneePos(idx,x,y);
                prediction_mul.at<double>(idx, 22) = x;
                prediction_mul.at<double>(idx, 23) = y;

		// cv::circle(frame,cv::Point(x,y),1,cv::Scalar(0,0,255),2,8,0);

		pos.getRightAnklePos(idx,x,y);
                prediction_mul.at<double>(idx, 24) = x;
                prediction_mul.at<double>(idx, 25) = y;

		// cv::circle(frame,cv::Point(x,y),1,cv::Scalar(0,0,255),2,8,0);
			
		pos.getLeftAnklePos(idx,x,y);
                prediction_mul.at<double>(idx, 26) = x;
                prediction_mul.at<double>(idx, 27) = y;

		// cv::circle(frame,cv::Point(x,y),1,cv::Scalar(0,0,255),2,8,0);
			
		pos.getRightEyePos(idx,x,y);
                prediction_mul.at<double>(idx, 28) = x;
                prediction_mul.at<double>(idx, 29) = y;

		// cv:circle(frame,cv::Point(x,y),1,cv::Scalar(0,0,255),2,8,0);
			
		pos.getLeftEyePos(idx,x,y);
                prediction_mul.at<double>(idx, 30) = x;
                prediction_mul.at<double>(idx, 31) = y;

		// cv::circle(frame,cv::Point(x,y),1,cv::Scalar(0,0,255),2,8,0);
		
		pos.getRightEarPos(idx,x,y);
                prediction_mul.at<double>(idx, 32) = x;
                prediction_mul.at<double>(idx, 33) = y;

		// cv::circle(frame,cv::Point(x,y),1,cv::Scalar(0,0,255),2,8,0);
			
		pos.getLeftEarPos(idx,x,y);
                prediction_mul.at<double>(idx, 34) = x;
                prediction_mul.at<double>(idx, 35) = y;

		// cv::circle(frame,cv::Point(x,y),1,cv::Scalar(0,0,255),2,8,0);
	}
	return prediction_mul;
}

Get_2DPose ldp;

//static void init() {
//    Py_Initialize();
//    import_array();
//}


PyObject *processFrame(PyObject * data) {
    cv::Mat frame;
    frame = pbcvt::fromNDArrayToMat(data);
    cv::Mat newFrame = frame.clone();
    cv::Mat prediction_mul = ldp.get2DPos(newFrame);
    PyObject *result = PyList_New(0);
    for (int i = 0; i < prediction_mul.rows; i++) {
        PyObject *jitem = PyList_New(0);
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,0)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,1)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,2)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,3)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,4)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,5)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,6)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,7)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,8)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,9)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,10)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,11)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,12)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,13)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,14)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,15)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,16)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,17)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,18)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,19)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,20)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,21)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,22)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,23)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,24)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,25)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,26)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,27)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,28)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,29)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,30)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,31)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,32)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,33)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,34)));
        PyList_Append(jitem, PyFloat_FromDouble(prediction_mul.at<double>(i,35)));
	PyList_Append(result, jitem);
    }
    return result;
}

void* extract_pyarray( PyObject* x )
{
    return PyObject_TypeCheck( x, &PyArray_Type ) ? x : 0;
}

BOOST_PYTHON_MODULE( rtpose ){
        // init();
        // This function needs to be included to pass PyObjects as numpy array ( http://mail.python.org/pipermail/cplusplus-sig/2006-September/011021.html )
        boost::python::converter::registry::insert( &extract_pyarray, boost::python::type_id<PyArrayObject>( ) );
        //def fromPython( "detect", &detect )
        py::def("rtpose", processFrame);
}