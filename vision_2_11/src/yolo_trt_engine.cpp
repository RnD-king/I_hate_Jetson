#include "vision/yolo_trt_engine.hpp"
#include "vision/preprocess.cuh"   // PreprocessKernelLauncher

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace nvinfer1;

namespace
{
inline void CheckCuda(cudaError_t e, const char* f, int l)
{
  if (e != cudaSuccess)
  {
    std::cerr << "[CUDA] " << cudaGetErrorString(e) << " at " << f << ":" << l << std::endl;
    throw std::runtime_error("CUDA failure");
  }
}
#define CHECK_CUDA(x) CheckCuda((x), __FILE__, __LINE__)

// TensorRT logger
class Logger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char* msg) noexcept override
  {
    if (severity == Severity::kINFO) return;
    std::cout << "[TensorRT] " << msg << std::endl;
  }
};

Logger gLogger;

static void PrintDims(const char* tag, const nvinfer1::Dims& d)
{
  std::cout << tag << " nbDims=" << d.nbDims << " [";
  for (int i = 0; i < d.nbDims; ++i)
  {
    std::cout << d.d[i];
    if (i + 1 < d.nbDims) std::cout << " x ";
  }
  std::cout << "]\n";
}

static bool DimsEqual(const nvinfer1::Dims& a, const std::vector<int>& b)
{
  if (a.nbDims != static_cast<int>(b.size())) return false;
  for (int i = 0; i < a.nbDims; ++i)
  {
    if (a.d[i] != b[i]) return false;
  }
  return true;
}

inline int ClampInt(int v, int lo, int hi)
{
  return (v < lo) ? lo : ((v > hi) ? hi : v);
}

}  // namespace

namespace vision
{

YoloTrtEngine::YoloTrtEngine(const std::string& engine_path, int input_w, int input_h, float conf_thres)
: input_w_(input_w), input_h_(input_h), conf_thres_(conf_thres)
{
  if (engine_path.empty())
  {
    throw std::runtime_error("engine_path is empty");
  }

  if (!LoadEngine(engine_path))
  {
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
  if (!file)
  {
    std::cerr << "[ERR] Cannot open engine: " << engine_path << std::endl;
    return false;
  }

  const size_t size = static_cast<size_t>(file.tellg());
  std::vector<char> blob(size);
  file.seekg(0, std::ios::beg);
  file.read(blob.data(), static_cast<std::streamsize>(size));

  runtime_.reset(createInferRuntime(gLogger));
  if (!runtime_)
  {
    std::cerr << "[ERR] createInferRuntime failed" << std::endl;
    return false;
  }

  engine_.reset(runtime_->deserializeCudaEngine(blob.data(), size));
  if (!engine_)
  {
    std::cerr << "[ERR] deserializeCudaEngine failed" << std::endl;
    return false;
  }

  context_.reset(engine_->createExecutionContext());
  if (!context_)
  {
    std::cerr << "[ERR] createExecutionContext failed" << std::endl;
    return false;
  }

  // 입력/출력 텐서 이름 자동 탐색
  const int n_io = engine_->getNbIOTensors();
  std::cout << "[DEBUG] getNbIOTensors=" << n_io << "\n";
  for (int i = 0; i < n_io; ++i)
  {
    const char* name = engine_->getIOTensorName(i);
    auto mode = engine_->getTensorIOMode(name);
    auto sh = engine_->getTensorShape(name);
    std::cout << "  [IO " << i << "] name=" << name
              << " mode=" << ((mode == TensorIOMode::kINPUT) ? "INPUT" : "OUTPUT")
              << " shape=";
    PrintDims("", sh);

    if (mode == TensorIOMode::kINPUT)  in_name_  = name;
    if (mode == TensorIOMode::kOUTPUT) out_name_ = name;
  }

  if (in_name_.empty() || out_name_.empty())
  {
    std::cerr << "[ERR] Failed to discover I/O tensor names" << std::endl;
    return false;
  }

  std::cout << "[DEBUG] selected in_name_=" << in_name_ << " out_name_=" << out_name_ << "\n";

  // 입력/출력 shape (엔진 기준)
  auto in_dims_engine  = engine_->getTensorShape(in_name_.c_str());
  auto out_dims_engine = engine_->getTensorShape(out_name_.c_str());

  // 동적 입력(-1)이면 런타임에 고정 shape 설정
  bool is_dynamic = false;
  for (int d = 0; d < in_dims_engine.nbDims; ++d)
  {
    if (in_dims_engine.d[d] == -1) is_dynamic = true;
  }

  if (is_dynamic)
  {
    nvinfer1::Dims4 fix{1, 3, input_h_, input_w_};
    if (!context_->setInputShape(in_name_.c_str(), fix))
    {
      std::cerr << "[ERR] setInputShape failed" << std::endl;
      return false;
    }
  }

  // 실행 컨텍스트 기준 shape
  const auto in_dims  = context_->getTensorShape(in_name_.c_str());
  const auto out_dims = context_->getTensorShape(out_name_.c_str());

  std::cout << "[INFO] Engine loaded: " << engine_path << "\n";
  std::cout << "       in=" << in_name_ << " out=" << out_name_ << "\n";
  PrintDims("       in_dims(engine)  =", in_dims_engine);
  PrintDims("       out_dims(engine) =", out_dims_engine);
  PrintDims("       in_dims(context) =", in_dims);
  PrintDims("       out_dims(context)=", out_dims);

  // 기대: in=1x3xHxW, out=1x5x8400 (single-class + obj)
  {
    const std::vector<int> expect_in  = {1, 3, input_h_, input_w_};
    const std::vector<int> expect_out = {1, 5, 8400};

    if (!DimsEqual(in_dims, expect_in))
    {
      std::cerr << "[ERR] Input dims mismatch. expected [1x3x"
                << input_h_ << "x" << input_w_ << "]\n";
      return false;
    }
    if (!DimsEqual(out_dims, expect_out))
    {
      std::cerr << "[ERR] Output dims mismatch. expected [1x5x8400]\n";
      return false;
    }
  }

  // 입력 버퍼
  size_t in_elems = 1;
  for (int d = 0; d < in_dims.nbDims; ++d) in_elems *= static_cast<size_t>(in_dims.d[d]);
  CHECK_CUDA(cudaMalloc(&d_input_, in_elems * sizeof(float)));

  // 출력 버퍼
  out_elems_ = 1;
  for (int d = 0; d < out_dims.nbDims; ++d) out_elems_ *= static_cast<size_t>(out_dims.d[d]);
  CHECK_CUDA(cudaMalloc(&d_out_, out_elems_ * sizeof(float)));
  CHECK_CUDA(cudaMallocHost(&h_out_, out_elems_ * sizeof(float)));

  gpu_bgr_ = nullptr;
  gpu_bgr_size_ = 0;

  std::cout << " out_elems=" << out_elems_ << std::endl;
  return true;
}

void YoloTrtEngine::Preprocess(const cv::Mat& img, float* gpu_input)
{
  cv::Mat src = img;
  if (!src.isContinuous())
  {
    src = src.clone();
  }

  const int src_w = src.cols;
  const int src_h = src.rows;

  const float scale = std::min(input_w_ / static_cast<float>(src_w),
                               input_h_ / static_cast<float>(src_h));
  const int new_w = static_cast<int>(src_w * scale);
  const int new_h = static_cast<int>(src_h * scale);
  const int pad_x = (input_w_ - new_w) / 2;
  const int pad_y = (input_h_ - new_h) / 2;

  const size_t required =
    static_cast<size_t>(src_w) * static_cast<size_t>(src_h) * 3; // BGR8

  if (!gpu_bgr_ || required > gpu_bgr_size_)
  {
    if (gpu_bgr_) CHECK_CUDA(cudaFree(gpu_bgr_));
    CHECK_CUDA(cudaMalloc(&gpu_bgr_, required));
    gpu_bgr_size_ = required;
  }

  CHECK_CUDA(cudaMemcpyAsync(gpu_bgr_, src.data, required, cudaMemcpyHostToDevice, stream_));

  // (BGR8 -> NCHW FP32, letterbox/pad=114 포함)
  PreprocessKernelLauncher(
    gpu_bgr_, src_w, src_h,
    gpu_input,
    input_w_, input_h_,
    new_w, new_h,
    pad_x, pad_y,
    scale,
    stream_,
    114
  );
}

bool YoloTrtEngine::Infer(const cv::Mat& img, std::vector<Detection>& detections)
{
  if (img.empty())
  {
    detections.clear();
    return false;
  }
  if (img.type() != CV_8UC3)
  {
    detections.clear();
    return false;
  }

  Preprocess(img, reinterpret_cast<float*>(d_input_));

  if (!context_->setTensorAddress(in_name_.c_str(), d_input_))
  {
    std::cerr << "[ERR] setTensorAddress(input) failed" << std::endl;
    return false;
  }
  if (!context_->setTensorAddress(out_name_.c_str(), d_out_))
  {
    std::cerr << "[ERR] setTensorAddress(output) failed" << std::endl;
    return false;
  }

  if (!context_->enqueueV3(stream_))
  {
    std::cerr << "[ERR] enqueueV3 failed" << std::endl;
    return false;
  }

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

  // output0 = (1,5,8400)
  constexpr int C = 5;
  constexpr int N = 8400;

  if (out_elems_ != static_cast<size_t>(C * N))
  {
    std::cerr << "[ERR] out_elems unexpected. out_elems=" << out_elems_
              << " expected=" << (C * N) << std::endl;
    return;
  }

  // letterbox undo 파라미터 (전처리와 동일 규칙)
  const float scale = std::min(input_w_ / static_cast<float>(img_w),
                               input_h_ / static_cast<float>(img_h));
  const int new_w = static_cast<int>(img_w * scale);
  const int new_h = static_cast<int>(img_h * scale);
  const int pad_x = (input_w_ - new_w) / 2;
  const int pad_y = (input_h_ - new_h) / 2;

  // ===== 디버그: 채널 통계 1회 출력 =====
  static bool printed_stats = false;
  if (!printed_stats)
  {
    printed_stats = true;
    std::cout << "[DBG] output stats (assume layout out[c*N + i]) C=" << C << " N=" << N << "\n";
    for (int c = 0; c < C; ++c)
    {
      float mn = std::numeric_limits<float>::infinity();
      float mx = -std::numeric_limits<float>::infinity();
      double sum = 0.0;
      for (int i = 0; i < N; ++i)
      {
        const float v = h_out_[c * N + i];
        mn = std::min(mn, v);
        mx = std::max(mx, v);
        sum += static_cast<double>(v);
      }
      const double mean = sum / static_cast<double>(N);
      std::cout << "  [c" << c << "] min=" << mn << " max=" << mx << " mean=" << mean << "\n";
    }
  }

  // 핵심: conf는 이미 [0,1]로 보이므로 sigmoid 절대 금지
  float max_conf = 0.0f;
  int cand = 0;

  for (int i = 0; i < N; ++i)
  {
    const float cx = h_out_[0 * N + i];
    const float cy = h_out_[1 * N + i];
    const float w  = h_out_[2 * N + i];
    const float h  = h_out_[3 * N + i];
    const float conf = h_out_[4 * N + i];   // <-- 그대로 사용 (NO sigmoid)

    max_conf = std::max(max_conf, conf);
    if (conf < conf_thres_) continue;
    ++cand;

    // (cx,cy,w,h) -> (x1,y1,x2,y2) in 640x640 space
    float x1 = cx - 0.5f * w;
    float y1 = cy - 0.5f * h;
    float x2 = cx + 0.5f * w;
    float y2 = cy + 0.5f * h;

    // letterbox 해제: (pad 제거 후) / scale
    x1 = (x1 - static_cast<float>(pad_x)) / scale;
    y1 = (y1 - static_cast<float>(pad_y)) / scale;
    x2 = (x2 - static_cast<float>(pad_x)) / scale;
    y2 = (y2 - static_cast<float>(pad_y)) / scale;

    // clamp
    int ix1 = ClampInt(static_cast<int>(std::floor(x1)), 0, img_w - 1);
    int iy1 = ClampInt(static_cast<int>(std::floor(y1)), 0, img_h - 1);
    int ix2 = ClampInt(static_cast<int>(std::ceil (x2)), 0, img_w - 1);
    int iy2 = ClampInt(static_cast<int>(std::ceil (y2)), 0, img_h - 1);

    const int bw = std::max(0, ix2 - ix1);
    const int bh = std::max(0, iy2 - iy1);
    if (bw <= 0 || bh <= 0) continue;

    Detection det;
    det.box = cv::Rect(ix1, iy1, bw, bh);
    det.confidence = conf;
    det.class_id = 0; // out이 5채널이면 class 정보가 별도로 없음(단일 클래스 가정)
    detections.push_back(det);
  }

  // 로그 1회
  static bool printed = false;
  if (!printed)
  {
    printed = true;
    std::cout << "[Postprocess] conf_thres=" << conf_thres_
              << " max_conf=" << max_conf
              << " cand=" << cand
              << " det=" << detections.size()
              << std::endl;
  }

  std::sort(detections.begin(), detections.end(),
            [](const Detection& a, const Detection& b){ return a.confidence > b.confidence; });
}

}  // namespace vision
