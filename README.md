# Linyan Intension Prediction based on Human Pose
By Linyan Wang

## Installation
 To run this code, the following packages must be installed:
 1. OpenCV(Version)
 2. tensorflow GPU

## Introduction
 This work is modified based on the project of "Realtime Multi-Person Pose Estimation"[1] proposed by the CMU Lab.

## Contents
 1. Data Preprocess
 2. LSTM-based Training
 3. LSTM-based Testing

<!-- ## Data Proprocessing -->
 <!-- 1. Run ``` -->

## Training & Testing
 1. Run ```bash get_model.sh``` to get the produced model used to estimate human pose.
 2. Run ```cd 3rdparty/caffe; make -j8``` to complie the modified Caffe.
 3. Return and run ```make -j8; cp ./build/rtpose.so .``` to make Makefile and produce "rtpose.so" dynamic library.
 4. Run ```linyan_lstm_train.py``` to train a model used to human intention prediction.
 5. Run ```linyan_lstm_test.py``` to test the result.

## Reference
 [1] Zhe Cao, Tomas Simon, Shih-En Wei, and Yaser Sheikh. Realtime multi-person 2d pose estimation using part affinity fields. In CVPR, 2017.

     


