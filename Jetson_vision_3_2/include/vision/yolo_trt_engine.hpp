#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <memory>

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferVersion.h>

#include "vision/types.hpp"

namespace vision
{

class YoloTrtEngine
{
public:
  explicit YoloTrtEngine(const std::string& engine_path,
                         int input_w = 640,
                         int input_h = 640,
                         float conf_thres = 0.25f);

  ~YoloTrtEngine();

  YoloTrtEngine(const YoloTrtEngine&) = delete;
  YoloTrtEngine& operator=(const YoloTrtEngine&) = delete;

  bool Infer(const cv::Mat& img, std::vector<Detection>& detections);

private:
  bool LoadEngine(const std::string& engine_path);
  void Preprocess(const cv::Mat& img, float* gpu_input);
  void Postprocess(std::vector<Detection>& detections, int img_w, int img_h);

private:
  int input_w_{640};
  int input_h_{640};
  float conf_thres_{0.45f};

  struct TrtDeleter
  {
    template <typename T>
    void operator()(T* p) const noexcept
    {
      if (!p) return;
#if defined(NV_TENSORRT_MAJOR) && (NV_TENSORRT_MAJOR >= 10)
      delete p;
#else
      p->destroy();
#endif
    }
  };

  std::unique_ptr<nvinfer1::IRuntime, TrtDeleter> runtime_{nullptr};
  std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter> engine_{nullptr};
  std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter> context_{nullptr};

  std::string in_name_;
  std::string out_name_;

  cudaStream_t stream_{nullptr};
  cudaEvent_t  ready_{nullptr};

  void*  d_input_{nullptr};   // float*
  void*  d_out_{nullptr};     // float*
  float* h_out_{nullptr};     // pinned host float*

  size_t in_elems_{0};
  size_t out_elems_{0};

  uint8_t* gpu_bgr_{nullptr};
  size_t   gpu_bgr_size_{0};
};

}  // namespace vision