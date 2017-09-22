#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <mutex>
#include <string>
#include <utility> //std::pair
#include <vector>

#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h> // snprintf
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <gflags/gflags.h>
#include <google/protobuf/text_format.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/cpm/frame.h"
#include "caffe/cpm/layers/imresize_layer.hpp"
#include "caffe/cpm/layers/nms_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "rtpose/modelDescriptor.h"
#include "rtpose/modelDescriptorFactory.h"
#include "rtpose/renderFunctions.h"

// global queues for I/O
struct Global {
  caffe::BlockingQueue<Frame> input_queue;  // have to pop
  caffe::BlockingQueue<Frame> output_queue; // have to pop
  caffe::BlockingQueue<Frame> output_queue_ordered;
  caffe::BlockingQueue<Frame> output_queue_mated;
  std::priority_queue<int, std::vector<int>, std::greater<int>> dropped_index;
  std::vector<std::string> image_list;
  std::mutex mutex;
  int part_to_show;
  bool quit_threads;
  // Parameters
  float nms_threshold;
  int connect_min_subset_cnt;
  float connect_min_subset_score;
  float connect_inter_threshold;
  int connect_inter_min_above_threshold;

  struct UIState {
    UIState()
        : is_fullscreen(0), is_video_paused(0), is_shift_down(0),
          is_googly_eyes(0), current_frame(0), seek_to_frame(-1), fps(0) {}
    bool is_fullscreen;
    bool is_video_paused;
    bool is_shift_down;
    bool is_googly_eyes;
    int current_frame;
    int seek_to_frame;
    double fps;
  };
  UIState uistate;
};

struct NetData {
  caffe::Net<float> *person_net;
  int num_people;
  int nms_max_peaks;
  int nms_num_parts;
  std::unique_ptr<ModelDescriptor> up_model_descriptor;
  float *canvas; // GPU memory
  float *joints; // GPU memory
};

struct ColumnCompare {
  bool operator()(const std::vector<double> &lhs,
                  const std::vector<double> &rhs) const {
    return lhs[2] > rhs[2];
  }
};

class RTPose {
public:
  struct RTPoseParameter {
    int device_id;
    int camera_width;
    int camera_height;
    int net_width;
    int net_height;
    std::string caffe_proto;
    std::string caffe_model;
	std::string input_file;
	std::istream* proto_pointer;
	std::istream* model_pointer;
  };
  RTPose(RTPoseParameter param);
  ~RTPose();

  void processFrame(cv::Mat &img, std::vector<cv::Point> &skeleton, int &cnt);

private:
  void warmup();
  void process_and_pad_image(float *target, cv::Mat oriImg, int tw, int th,
                             bool normalize);
  int connectLimbs(std::vector<std::vector<double>> &subset,
                   std::vector<std::vector<std::vector<double>>> &connection,
                   const float *heatmap_pointer, const float *peaks,
                   int max_peaks, float *joints,
                   ModelDescriptor *model_descriptor);

  Global global_;
  Frame frame_;
  NetData net_data_;
  int device_id_;
  int camera_width_;
  int camera_height_;
  int net_width_;
  int net_height_;
  std::string caffe_proto_;
  std::string caffe_model_;
  std::istream* proto_pointer_;
  std::istream* model_pointer_;

  int g_counter_;
};
