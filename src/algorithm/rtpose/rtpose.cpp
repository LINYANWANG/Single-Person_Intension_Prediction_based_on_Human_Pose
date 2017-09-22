#include "rtpose.h"

static int limb[19 * 2] = {1, 2,  1,  5,  2,  3,  3,  4,  5,  6,  6,  7, 1,
                           8, 8,  9,  9,  10, 1,  11, 11, 12, 12, 13, 1, 0,
                           0, 14, 14, 16, 0,  15, 15, 17, 2,  16, 5,  17};

// Global parameters
const auto MAX_PEOPLE = RENDER_MAX_PEOPLE; // defined in render_functions.hpp
const auto MAX_NUM_PARTS = 70;

static double get_wall_time() {
  struct timeval time;
  if (gettimeofday(&time, NULL)) {
    //  Handle error
    return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * 1e-6;
  // return (double)time.tv_usec;
}

RTPose::RTPose(RTPoseParameter param) {
  device_id_ = param.device_id;
  camera_width_ = param.camera_width;
  camera_height_ = param.camera_height;
  net_width_ = param.net_width;
  net_height_ = param.net_height;
  caffe_proto_ = param.caffe_proto;
  caffe_model_ = param.caffe_model;
  proto_pointer_ = param.proto_pointer;
  model_pointer_ = param.model_pointer;

  g_counter_ = 0;

  frame_.data = new float[3 * net_height_ * net_width_];

  LOG(INFO) << "Camera resolution: " << camera_width_ << "x" << camera_height_;
  LOG(INFO) << "Net resolution: " << net_width_ << "x" << net_height_;

  warmup();
}

RTPose::~RTPose() { delete frame_.data; }

void RTPose::warmup() {
  LOG(INFO) << "Setting GPU " << device_id_;
  int logtostderr = FLAGS_logtostderr;

  caffe::Caffe::SetDevice(device_id_);       // cudaSetDevice(device_id_) inside
  caffe::Caffe::set_mode(caffe::Caffe::GPU); //

  LOG(INFO) << "GPU " << device_id_ << ": copying to person net";
  FLAGS_logtostderr = 0;
  //net_data_.person_net = new caffe::Net<float>(caffe_proto_, caffe::TEST);
  //net_data_.person_net->CopyTrainedLayersFrom(caffe_model_);
  net_data_.person_net = new caffe::Net<float>(proto_pointer_, caffe::TEST);
  net_data_.person_net->CopyTrainedLayersFromMemory(model_pointer_);

  const std::vector<int> shape{{1, 3, net_height_, net_width_}};

  net_data_.person_net->blobs()[0]->Reshape(shape);
  net_data_.person_net->Reshape();
  FLAGS_logtostderr = logtostderr;

  caffe::ImResizeLayer<float> *resize_layer =
      (caffe::ImResizeLayer<float> *)net_data_.person_net
          ->layer_by_name("resize")
          .get();

  resize_layer->SetStartScale(1);
  resize_layer->SetScaleGap(0.5);

  caffe::NmsLayer<float> *nms_layer =
      (caffe::NmsLayer<float> *)net_data_.person_net->layer_by_name("nms")
          .get();
  net_data_.nms_max_peaks = nms_layer->GetMaxPeaks();
  net_data_.nms_num_parts = nms_layer->GetNumParts();
  CHECK_LE(net_data_.nms_num_parts, MAX_NUM_PARTS)
      << "num_parts in NMS layer (" << net_data_.nms_num_parts << ") "
      << "too big ( MAX_NUM_PARTS )";

  if (net_data_.nms_num_parts == 15) {
    ModelDescriptorFactory::createModelDescriptor(
        ModelDescriptorFactory::Type::MPI_15, net_data_.up_model_descriptor);
    global_.nms_threshold = 0.2;
    global_.connect_min_subset_cnt = 3;
    global_.connect_min_subset_score = 0.4;
    global_.connect_inter_threshold = 0.01;
    global_.connect_inter_min_above_threshold = 8;
    LOG(INFO) << "Selecting MPI model.";
  } else if (net_data_.nms_num_parts == 18) {
    ModelDescriptorFactory::createModelDescriptor(
        ModelDescriptorFactory::Type::COCO_18, net_data_.up_model_descriptor);
    global_.nms_threshold = 0.05;
    global_.connect_min_subset_cnt = 5;
    // global_.connect_min_subset_cnt = 3;
    global_.connect_min_subset_score = 0.4;
    global_.connect_inter_threshold = 0.050;
    global_.connect_inter_min_above_threshold = 9;
  } else {
    CHECK(0) << "Unknown number of parts! Couldn't set model";
  }

  // dry run
  LOG(INFO) << "Dry running...";
  net_data_.person_net->Forward();
  LOG(INFO) << "GPU " << device_id_ << " is ready";
}

void RTPose::process_and_pad_image(float *target, cv::Mat oriImg, int tw,
                                   int th, bool normalize) {
  int ow = oriImg.cols;
  int oh = oriImg.rows;
  int offset2_target = tw * th;

  int padw = (tw - ow) / 2;
  int padh = (th - oh) / 2;
  // LOG(ERROR) << " padw " << padw << " padh " << padh;
  CHECK_GE(padw, 0) << "Image too big for target size.";
  CHECK_GE(padh, 0) << "Image too big for target size.";
  // parallel here
  unsigned char *pointer = (unsigned char *)(oriImg.data);

  for (int c = 0; c < 3; c++) {
    for (int y = 0; y < th; y++) {
      int oy = y - padh;
      for (int x = 0; x < tw; x++) {
        int ox = x - padw;
        if (ox >= 0 && ox < ow && oy >= 0 && oy < oh) {
          if (normalize)
            target[c * offset2_target + y * tw + x] =
                float(pointer[(oy * ow + ox) * 3 + c]) / 256.0f - 0.5f;
          else
            target[c * offset2_target + y * tw + x] =
                float(pointer[(oy * ow + ox) * 3 + c]);
        } else {
          target[c * offset2_target + y * tw + x] = 0;
        }
      }
    }
  }
}

int RTPose::connectLimbs(
    std::vector<std::vector<double>> &subset,
    std::vector<std::vector<std::vector<double>>> &connection,
    const float *heatmap_pointer, const float *peaks, int max_peaks,
    float *joints, ModelDescriptor *model_descriptor) {

  const auto num_parts = model_descriptor->get_number_parts();
  const auto limbSeq = model_descriptor->get_limb_sequence();
  const auto mapIdx = model_descriptor->get_map_idx();
  const auto number_limb_seq = model_descriptor->number_limb_sequence();

  int SUBSET_CNT = num_parts + 2;
  int SUBSET_SCORE = num_parts + 1;
  int SUBSET_SIZE = num_parts + 3;

  CHECK((num_parts == 15 && number_limb_seq == 14) ||
        (num_parts == 18 && number_limb_seq == 19))
      << "Wrong connection function for model";

  int peaks_offset = 3 * (max_peaks + 1);
  subset.clear();
  connection.clear();
  for (int k = 0; k < number_limb_seq; k++) {
    const float *map_x =
        heatmap_pointer + mapIdx[2 * k] * net_height_ * net_width_;
    const float *map_y =
        heatmap_pointer + mapIdx[2 * k + 1] * net_height_ * net_width_;

    const float *candA = peaks + limbSeq[2 * k] * peaks_offset;
    const float *candB = peaks + limbSeq[2 * k + 1] * peaks_offset;

    std::vector<std::vector<double>> connection_k;
    int nA = candA[0];
    int nB = candB[0];

    // add parts into the subset in special case
    if (nA == 0 || nB == 0) {
      continue;
    }

    std::vector<std::vector<double>> temp;
    const int num_inter = 10;
    for (int i = 1; i <= nA; i++) {
      for (int j = 1; j <= nB; j++) {
        float s_x = candA[i * 3];
        float s_y = candA[i * 3 + 1];
        float d_x = candB[j * 3] - candA[i * 3];
        float d_y = candB[j * 3 + 1] - candA[i * 3 + 1];
        float norm_vec = sqrt(pow(d_x, 2) + pow(d_y, 2));
        if (norm_vec < 1e-6) {
          continue;
        }
        if (std::isnan(d_x) || std::isnan(d_y)) {
          continue;
        }
        float vec_x = d_x / norm_vec;
        float vec_y = d_y / norm_vec;

        float sum = 0;
        int count = 0;

        for (int lm = 0; lm < num_inter; lm++) {
          int my = round(s_y + lm * d_y / num_inter);
          int mx = round(s_x + lm * d_x / num_inter);
          int idx = my * net_width_ + mx;
          float score = (vec_x * map_x[idx] + vec_y * map_y[idx]);
          if (score > global_.connect_inter_threshold) {
            sum = sum + score;
            count++;
          }
        }
        // float score = sum / count; // + std::min((130/dist-1),0.f)
        if (count >
            global_.connect_inter_min_above_threshold) { // num_inter*0.8) {
                                                         // //thre/2
          // parts score + cpnnection score
          std::vector<double> row_vec(4, 0);
          row_vec[3] =
              sum / count + candA[i * 3 + 2] + candB[j * 3 + 2]; // score_all
          row_vec[2] = sum / count;
          row_vec[0] = i;
          row_vec[1] = j;
          temp.push_back(row_vec);
        }
      }
    }

    //** select the top num connection, assuming that each part occur only once
    // sort rows in descending order based on parts + connection score
    if (temp.size() > 0)
      std::sort(temp.begin(), temp.end(), ColumnCompare());

    int num = std::min(nA, nB);
    int cnt = 0;
    std::vector<int> occurA(nA, 0);
    std::vector<int> occurB(nB, 0);

    for (int row = 0; row < temp.size(); row++) {
      if (cnt == num) {
        break;
      } else {
        int i = int(temp[row][0]);
        int j = int(temp[row][1]);
        float score = temp[row][2];
        if (occurA[i - 1] == 0 && occurB[j - 1] == 0) { // && score> (1+thre)
          std::vector<double> row_vec(3, 0);
          row_vec[0] = limbSeq[2 * k] * peaks_offset + i * 3 + 2;
          row_vec[1] = limbSeq[2 * k + 1] * peaks_offset + j * 3 + 2;
          row_vec[2] = score;
          connection_k.push_back(row_vec);
          cnt = cnt + 1;
          occurA[i - 1] = 1;
          occurB[j - 1] = 1;
        }
      }
    }

    if (k == 0) {
      std::vector<double> row_vec(num_parts + 3, 0);
      for (int i = 0; i < connection_k.size(); i++) {
        double indexA = connection_k[i][0];
        double indexB = connection_k[i][1];
        row_vec[limbSeq[0]] = indexA;
        row_vec[limbSeq[1]] = indexB;
        row_vec[SUBSET_CNT] = 2;
        // add the score of parts and the connection
        row_vec[SUBSET_SCORE] =
            peaks[int(indexA)] + peaks[int(indexB)] + connection_k[i][2];
        subset.push_back(row_vec);
      }
    } else {
      if (connection_k.size() == 0) {
        continue;
      }
      // A is already in the subset, find its connection B
      for (int i = 0; i < connection_k.size(); i++) {
        int num = 0;
        double indexA = connection_k[i][0];
        double indexB = connection_k[i][1];

        for (int j = 0; j < subset.size(); j++) {
          if (subset[j][limbSeq[2 * k]] == indexA) {
            subset[j][limbSeq[2 * k + 1]] = indexB;
            num = num + 1;
            subset[j][SUBSET_CNT] = subset[j][SUBSET_CNT] + 1;
            subset[j][SUBSET_SCORE] = subset[j][SUBSET_SCORE] +
                                      peaks[int(indexB)] + connection_k[i][2];
          }
        }
        // if can not find partA in the subset, create a new subset
        if (num == 0) {
          std::vector<double> row_vec(SUBSET_SIZE, 0);
          row_vec[limbSeq[2 * k]] = indexA;
          row_vec[limbSeq[2 * k + 1]] = indexB;
          row_vec[SUBSET_CNT] = 2;
          row_vec[SUBSET_SCORE] =
              peaks[int(indexA)] + peaks[int(indexB)] + connection_k[i][2];
          subset.push_back(row_vec);
        }
      }
    }
  }
  //** joints by deleting some rows of subset which has few parts occur
  int cnt = 0;
  for (int i = 0; i < subset.size(); i++) {
    if (subset[i][SUBSET_CNT] >= global_.connect_min_subset_cnt &&
        (subset[i][SUBSET_SCORE] / subset[i][SUBSET_CNT]) >
            global_.connect_min_subset_score) {
      for (int j = 0; j < num_parts; j++) {
        int idx = int(subset[i][j]);
        if (idx) {
          joints[cnt * num_parts * 3 + j * 3 + 2] = peaks[idx];
          joints[cnt * num_parts * 3 + j * 3 + 1] =
              peaks[idx - 1] * camera_width_ / (float)net_width_;
          joints[cnt * num_parts * 3 + j * 3] =
              peaks[idx - 2] * camera_height_ / (float)net_height_;
        } else {
          joints[cnt * num_parts * 3 + j * 3 + 2] = 0;
          joints[cnt * num_parts * 3 + j * 3 + 1] = 0;
          joints[cnt * num_parts * 3 + j * 3] = 0;
        }
      }
      cnt++;
      if (cnt == MAX_PEOPLE)
        break;
    }
  }

  return cnt;
}

void RTPose::processFrame(cv::Mat &img, std::vector<cv::Point> &skeleton, int &cnt) { 
  frame_.scale = 1.0;
  frame_.index = g_counter_++;
  frame_.video_frame_number = frame_.index; // no frame_ skip now

  // pad and transform to float
  cv::Mat image;
  cv::Mat M = cv::Mat::eye(2, 3, CV_64F);
  double scale = 0;

  if (img.cols / (double)img.rows > net_width_ / (double)net_height_) {
    scale = net_width_ / (double)img.cols;
  } else {
    scale = net_height_ / (double)img.rows;
  }

  M.at<double>(0, 0) = scale;
  M.at<double>(1, 1) = scale;
  cv::warpAffine(img, image, M, cv::Size(net_width_, net_height_),
                 CV_INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

  // cv::imwrite("test.png", image);
  process_and_pad_image(frame_.data, image, net_width_, net_height_, 1);

  frame_.gpu_fetched_time = get_wall_time();
  int offset = net_width_ * net_height_ * 3;

  std::vector<std::vector<double>> subset;
  std::vector<std::vector<std::vector<double>>> connection;

  const boost::shared_ptr<caffe::Blob<float>> heatmap_blob =
      net_data_.person_net->blob_by_name("resized_map");
  const boost::shared_ptr<caffe::Blob<float>> joints_blob =
      net_data_.person_net->blob_by_name("joints");

  caffe::NmsLayer<float> *nms_layer =
      (caffe::NmsLayer<float> *)net_data_.person_net->layer_by_name("nms")
          .get();

  float *pointer = net_data_.person_net->blobs()[0]->mutable_gpu_data();

  cudaMemcpy(pointer, frame_.data, offset * sizeof(float),
             cudaMemcpyHostToDevice);

  nms_layer->SetThreshold(global_.nms_threshold);
  net_data_.person_net->Forward();
  LOG(INFO) << "CNN time "
            << (get_wall_time() - frame_.gpu_fetched_time) * 1000.0 << " ms.";

  const float *heatmap_pointer = heatmap_blob->mutable_cpu_data();
  const float *peaks = joints_blob->mutable_cpu_data();

  float joints[MAX_NUM_PARTS * 3 * MAX_PEOPLE]; // 10*15*3

  cnt = 0;
  // CHECK_EQ(net_data_.nms_num_parts, 15);
  double tic = get_wall_time();
  const int num_parts = net_data_.nms_num_parts;
  cnt = connectLimbs(subset, connection, heatmap_pointer, peaks,
                     net_data_.nms_max_peaks, joints,
                     net_data_.up_model_descriptor.get());

  LOG(INFO) << "CNT:" << cnt;

  int p_x, p_y;
  
  // confirm the passenger in passageway
  for (int i = 0; i < cnt; i++) {
    for (int j = 0; j < num_parts; j++) {
      p_x = joints[i * num_parts * 3 + j * 3];
      p_y = joints[i * num_parts * 3 + j * 3 + 1];
      skeleton.push_back(cv::Point(p_x, p_y));
      cv::circle(img, skeleton[j], 1, cv::Scalar(255, 0, 255), 2, 8, 0);
    }

    for (int j = 0; j < 17; j++) {
      if (skeleton[limb[2 * j]].x != 0 && skeleton[limb[2 * j + 1]].x != 0)
        cv::line(img, skeleton[limb[2 * j]], skeleton[limb[2 * j + 1]],
                 cv::Scalar(255, 255, 0));
    }
  }

  net_data_.num_people = cnt;
  LOG(INFO) << "CNT: " << cnt << " Connect time "
            << (get_wall_time() - tic) * 1000.0 << " ms.";

  frame_.numPeople = net_data_.num_people;
  frame_.gpu_computed_time = get_wall_time();
}
