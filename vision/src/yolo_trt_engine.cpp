#include "vision/yolo_trt_engine.hpp"
#include "vision/preprocess.cuh"   // CUDA 전처리 커널 런처

#include <algorithm>
#include <chrono>
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

// TensorRT logger (INFO는 너무 시끄러우면 제외)
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

// Dims 출력 유틸
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

// Dims 비교(assert) 유틸: (nbDims, 각 d값) 완전 일치 검사
static bool DimsEqual(const nvinfer1::Dims& a, const std::vector<int>& b)
{
  if (a.nbDims != static_cast<int>(b.size())) return false;
  for (int i = 0; i < a.nbDims; ++i)
  {
    if (a.d[i] != b[i]) return false;
  }
  return true;
}

// output0 raw 통계 출력(레이아웃 2가지를 모두 확인)
static void DumpOutputStats(const float* out, int C, int N, int sample_n = 5)
{
  auto dump_minmax_L1 = [&](int c) {
    float mn =  std::numeric_limits<float>::infinity();
    float mx = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < N; ++i)
    {
      const float v = out[c * N + i];
      mn = std::min(mn, v);
      mx = std::max(mx, v);
    }
    std::cout << "    L1 c" << c << " min=" << mn << " max=" << mx << "\n";
  };

  auto dump_minmax_L2 = [&](int c) {
    float mn =  std::numeric_limits<float>::infinity();
    float mx = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < N; ++i)
    {
      const float v = out[i * C + c];
      mn = std::min(mn, v);
      mx = std::max(mx, v);
    }
    std::cout << "    L2 c" << c << " min=" << mn << " max=" << mx << "\n";
  };

  std::cout << "[RAW output0 stats] assume C=" << C << " N=" << N << "\n";
  std::cout << "  Layout L1: out[c*N + i]\n";
  for (int c = 0; c < C; ++c) dump_minmax_L1(c);

  std::cout << "  Layout L2: out[i*C + c]\n";
  for (int c = 0; c < C; ++c) dump_minmax_L2(c);

  // 샘플(앞부분 몇 개)도 같이 보기
  std::cout << "  Samples (first " << sample_n << ")\n";
  std::cout << "    L1:\n";
  for (int i = 0; i < std::min(sample_n, N); ++i)
  {
    std::cout << "      i=" << i << " : ";
    for (int c = 0; c < C; ++c)
    {
      std::cout << out[c * N + i] << (c + 1 < C ? ' ' : '\n');
    }
  }
  std::cout << "    L2:\n";
  for (int i = 0; i < std::min(sample_n, N); ++i)
  {
    std::cout << "      i=" << i << " : ";
    for (int c = 0; c < C; ++c)
    {
      std::cout << out[i * C + c] << (c + 1 < C ? ' ' : '\n');
    }
  }
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
  // destructor에서는 throw 피하기 위해 cudaFree 계열만 직접 호출
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
  // 엔진 파일 로드
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
  for (int i = 0; i < n_io; ++i)
  {
    const char* name = engine_->getIOTensorName(i);
    auto mode = engine_->getTensorIOMode(name);
    if (mode == TensorIOMode::kINPUT)  in_name_  = name;
    if (mode == TensorIOMode::kOUTPUT) out_name_ = name;
  }

  if (in_name_.empty() || out_name_.empty())
  {
    std::cerr << "[ERR] Failed to discover I/O tensor names" << std::endl;
    return false;
  }

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

  // 실행 컨텍스트 기준 shape를 다시 조회(동적/정적 모두 여기 값이 “실제 런타임”)
  const auto in_dims  = context_->getTensorShape(in_name_.c_str());
  const auto out_dims = context_->getTensorShape(out_name_.c_str());

  std::cout << "[INFO] Engine loaded: " << engine_path << "\n";
  std::cout << "       in=" << in_name_ << " out=" << out_name_ << "\n";
  PrintDims("       in_dims(engine)  =", in_dims_engine);
  PrintDims("       out_dims(engine) =", out_dims_engine);
  PrintDims("       in_dims(context) =", in_dims);
  PrintDims("       out_dims(context)=", out_dims);

  // === (요청 1) 여기서 shape assert로 “고정” ===
  // 기대: images=1x3x640x640, output0=1x5x8400
  // input_w_/input_h_는 생성자에서 받는 값이므로 (640,640) 전제면 아래가 맞아야 함.
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

  // 입력 버퍼 할당
  size_t in_elems = 1;
  for (int d = 0; d < in_dims.nbDims; ++d)
  {
    in_elems *= static_cast<size_t>(in_dims.d[d]);
  }
  CHECK_CUDA(cudaMalloc(&d_input_, in_elems * sizeof(float)));

  // 출력 버퍼 할당
  out_elems_ = 1;
  for (int d = 0; d < out_dims.nbDims; ++d)
  {
    out_elems_ *= static_cast<size_t>(out_dims.d[d]);
  }
  CHECK_CUDA(cudaMalloc(&d_out_, out_elems_ * sizeof(float)));
  CHECK_CUDA(cudaMallocHost(&h_out_, out_elems_ * sizeof(float)));

  // 전처리 staging 초기화
  gpu_bgr_ = nullptr;
  gpu_bgr_size_ = 0;

  std::cout << "       out_elems=" << out_elems_ << std::endl;

  return true;
}

void YoloTrtEngine::Preprocess(const cv::Mat& img, float* gpu_input)
{
  // 전제: preprocess 커널이
  // - BGR8 -> RGB -> NCHW FP32
  // - letterbox resize + pad
  // - normalize
  // 를 모두 수행

  // (안전) Mat 연속성 보장
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

  PreprocessKernelLauncher(
    gpu_bgr_, src_w, src_h,
    gpu_input,
    input_w_, input_h_,   // 최종 출력 크기(예: 640,640)
    new_w, new_h,
    pad_x, pad_y,
    scale,
    stream_,
    114                   // pad 값
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

  // 1) 전처리 -> d_input_
  Preprocess(img, reinterpret_cast<float*>(d_input_));

  // 2) 바인딩(입력 1개, 출력 1개 가정)
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

  // 3) 실행
  if (!context_->enqueueV3(stream_))
  {
    std::cerr << "[ERR] enqueueV3 failed" << std::endl;
    return false;
  }

  // 4) 출력 복사(비동기) -> 이벤트 동기화
  CHECK_CUDA(cudaMemcpyAsync(h_out_, d_out_, out_elems_ * sizeof(float),
                             cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA(cudaEventRecord(ready_, stream_));
  CHECK_CUDA(cudaEventSynchronize(ready_));

  // 5) 파싱(현재는 디코딩 보류: raw 통계 출력 모드)
  Postprocess(detections, img.cols, img.rows);
  return true;
}

void YoloTrtEngine::Postprocess(std::vector<Detection>& detections, int /*img_w*/, int /*img_h*/)
{
  detections.clear();

  // === (요청 2) (1,N,6) 가정 제거: raw output0 통계 출력 모드 ===
  // output0: (1,5,8400) 가정(LoadEngine assert로 이미 고정됨)
  // 여기서는 “디코딩”을 하지 않는다. (cxcywh / x1y1 등은 다음 단계)
  // 대신 레이아웃 후보 2개(L1/L2)에 대해 min/max 및 샘플을 출력한다.

  // 로그 폭발 방지: 최초 1회만 출력
  static bool printed = false;
  if (!printed)
  {
    printed = true;
    const int C = 5;
    const int N = 8400;

    std::cout << "[INFO] Postprocess in RAW-STATS mode (decode disabled)\n";
    DumpOutputStats(h_out_, C, N, /*sample_n=*/5);
  }

  // detections는 비워둔 채로 리턴 (line_point_extractor 쪽에서도 비게 됨)
  // 다음 단계에서 output0 형식 확정 후 디코딩을 구현하면 된다.
}

}  // namespace vision
