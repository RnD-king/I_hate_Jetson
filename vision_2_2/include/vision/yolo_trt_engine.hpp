#pragma once
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <cuda_runtime.h>

#include "vision/types.hpp"

// TensorRT
#include <NvInfer.h>

namespace vision {

class YoloTrtEngine final {
public:
  explicit YoloTrtEngine(const std::string& engine_path, int input_w = 640, int input_h = 640);
  ~YoloTrtEngine();

  bool Infer(const cv::Mat& bgr, std::vector<Detection>& detections);

private:
  bool LoadEngine(const std::string& engine_path);
  void Preprocess(const cv::Mat& bgr, float* gpu_input);
  void Postprocess(std::vector<Detection>& detections, int img_w, int img_h);

private:
  int input_w_{640};
  int input_h_{640};

  std::string in_name_;
  std::string out_name_;

  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  cudaStream_t stream_{};
  cudaEvent_t ready_{};

  // staging / io
  uint8_t* gpu_bgr_{nullptr};
  size_t gpu_bgr_size_{0};

  void* d_input_{nullptr};
  void* d_out_{nullptr};
  float* h_out_{nullptr};
  size_t out_elems_{0};
};

}  // namespace vision
