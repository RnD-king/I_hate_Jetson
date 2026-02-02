#include "vision/yolo_trt_engine.hpp"
#include "vision/preprocess.cuh"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace nvinfer1;

namespace {

inline void CheckCuda(cudaError_t e, const char* f, int l) {
  if (e != cudaSuccess) {
    std::cerr << "[CUDA] " << cudaGetErrorString(e) << " at " << f << ":" << l << std::endl;
    throw std::runtime_error("CUDA failure");
  }
}
#define CHECK_CUDA(x) CheckCuda((x), __FILE__, __LINE__)

class Logger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity != Severity::kINFO) {
      std::cout << "[TensorRT] " << msg << std::endl;
    }
  }
};
static Logger gLogger;

}  // namespace

namespace vision {

YoloTrtEngine::YoloTrtEngine(const std::string& engine_path, int input_w, int input_h)
: input_w_(input_w), input_h_(input_h)
{
  if (!LoadEngine(engine_path)) {
    throw std::runtime_error("Failed to load TensorRT engine");
  }
  CHECK_CUDA(cudaStreamCreate(&stream_));
  CHECK_CUDA(cudaEventCreate(&ready_));
}

YoloTrtEngine::~YoloTrtEngine()
{
  if (gpu_bgr_) cudaFree(gpu_bgr_);
  if (d_input_) cudaFree(d_input_);
  if (d_out_)   cudaFree(d_out_);
  if (h_out_)   cudaFreeHost(h_out_);
  if (ready_)   cudaEventDestroy(ready_);
  if (stream_)  cudaStreamDestroy(stream_);

  context_.reset();
  engine_.reset();
  runtime_.reset();
}

bool YoloTrtEngine::LoadEngine(const std::string& engine_path)
{
  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  if (!file) {
    std::cerr << "[ERR] Cannot open engine: " << engine_path << std::endl;
    return false;
  }
  size_t size = (size_t)file.tellg();
  std::vector<char> blob(size);
  file.seekg(0, std::ios::beg);
  file.read(blob.data(), (std::streamsize)size);

  runtime_.reset(createInferRuntime(gLogger));
  if (!runtime_) return false;

  engine_.reset(runtime_->deserializeCudaEngine(blob.data(), size));
  if (!engine_) return false;

  context_.reset(engine_->createExecutionContext());
  if (!context_) return false;

  // Discover I/O names
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char* name = engine_->getIOTensorName(i);
    auto mode = engine_->getTensorIOMode(name);
    if (mode == TensorIOMode::kINPUT)  in_name_  = name;
    if (mode == TensorIOMode::kOUTPUT) out_name_ = name;
  }
  if (in_name_.empty() || out_name_.empty()) return false;

  // Fix dynamic input if needed
  auto in_dims = engine_->getTensorShape(in_name_.c_str());
  bool is_dynamic = false;
  for (int d = 0; d < in_dims.nbDims; ++d) {
    if (in_dims.d[d] == -1) is_dynamic = true;
  }
  if (is_dynamic) {
    Dims4 fix{1, 3, input_h_, input_w_};
    if (!context_->setInputShape(in_name_.c_str(), fix)) return false;
    in_dims = fix;
  }

  auto out_dims = engine_->getTensorShape(out_name_.c_str());

  // Allocate input
  size_t in_elems = 1;
  for (int d = 0; d < in_dims.nbDims; ++d) in_elems *= (size_t)in_dims.d[d];
  CHECK_CUDA(cudaMalloc(&d_input_, in_elems * sizeof(float)));

  // Allocate output
  out_elems_ = 1;
  for (int d = 0; d < out_dims.nbDims; ++d) out_elems_ *= (size_t)out_dims.d[d];
  CHECK_CUDA(cudaMalloc(&d_out_, out_elems_ * sizeof(float)));
  CHECK_CUDA(cudaMallocHost(&h_out_, out_elems_ * sizeof(float)));

  gpu_bgr_ = nullptr;
  gpu_bgr_size_ = 0;

  std::cout << "[INFO] Engine loaded: " << engine_path << "\n";
  std::cout << "       in=" << in_name_ << " out=" << out_name_
            << " out_elems=" << out_elems_ << "\n";
  return true;
}

void YoloTrtEngine::Preprocess(const cv::Mat& img, float* gpu_input)
{
  const int src_w = img.cols;
  const int src_h = img.rows;

  const float scale = std::min(input_w_ / (float)src_w, input_h_ / (float)src_h);
  const int resized_w = (int)std::round(src_w * scale);
  const int resized_h = (int)std::round(src_h * scale);
  const int pad_x = (input_w_ - resized_w) / 2;
  const int pad_y = (input_h_ - resized_h) / 2;

  size_t required = (size_t)src_w * (size_t)src_h * 3; // BGR8
  if (!gpu_bgr_ || required > gpu_bgr_size_) {
    if (gpu_bgr_) CHECK_CUDA(cudaFree(gpu_bgr_));
    CHECK_CUDA(cudaMalloc(&gpu_bgr_, required));
    gpu_bgr_size_ = required;
  }

  CHECK_CUDA(cudaMemcpyAsync(gpu_bgr_, img.data, required, cudaMemcpyHostToDevice, stream_));

  // pad_value = 114 (YOLO letterbox convention)
  PreprocessKernelLauncher(
    (const std::uint8_t*)gpu_bgr_,
    src_w, src_h,
    gpu_input,
    input_w_, input_h_,           // output_w/h (network input)
    resized_w, resized_h,
    pad_x, pad_y,
    stream_,
    (std::uint8_t)114
  );
}

bool YoloTrtEngine::Infer(const cv::Mat& img, std::vector<Detection>& detections)
{
  detections.clear();
  if (img.empty()) return false;
  if (img.type() != CV_8UC3) return false;

  Preprocess(img, (float*)d_input_);

  if (!context_->setTensorAddress(in_name_.c_str(), d_input_))  return false;
  if (!context_->setTensorAddress(out_name_.c_str(), d_out_))   return false;

  if (!context_->enqueueV3(stream_)) return false;

  CHECK_CUDA(cudaMemcpyAsync(h_out_, d_out_, out_elems_ * sizeof(float),
                             cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA(cudaEventRecord(ready_, stream_));
  CHECK_CUDA(cudaEventSynchronize(ready_));

  Postprocess(detections, img.cols, img.rows);
  return true;
}

void YoloTrtEngine::Postprocess(std::vector<Detection>& detections, int img_w, int img_h)
{
  detections.clear();

  if (out_elems_ % 6 != 0) return;              // [x1,y1,x2,y2,conf,cls]
  const int N = (int)(out_elems_ / 6);
  const float* p = h_out_;

  // Undo letterbox (must match Preprocess)
  float scale = std::min(input_w_ / (float)img_w, input_h_ / (float)img_h);
  int resized_w = (int)std::round(img_w * scale);
  int resized_h = (int)std::round(img_h * scale);
  int pad_x = (input_w_ - resized_w) / 2;
  int pad_y = (input_h_ - resized_h) / 2;

  auto unpad = [&](float x, float y) -> cv::Point2f {
    float xx = (x - pad_x) / scale;
    float yy = (y - pad_y) / scale;
    return {xx, yy};
  };

  for (int i = 0; i < N; ++i) {
    float x1 = p[0], y1 = p[1], x2 = p[2], y2 = p[3];
    float conf = p[4];
    int cls = (int)p[5];
    p += 6;

    if (conf <= 0.60f) continue;

    cv::Point2f a = unpad(x1, y1);
    cv::Point2f b = unpad(x2, y2);

    int xi = std::clamp((int)std::round(a.x), 0, img_w - 1);
    int yi = std::clamp((int)std::round(a.y), 0, img_h - 1);
    int xj = std::clamp((int)std::round(b.x), 0, img_w - 1);
    int yj = std::clamp((int)std::round(b.y), 0, img_h - 1);

    int w = std::max(0, xj - xi);
    int h = std::max(0, yj - yi);
    if (w <= 1 || h <= 1) continue;

    detections.push_back({cv::Rect(xi, yi, w, h), conf, cls});
  }
}

}  // namespace vision
