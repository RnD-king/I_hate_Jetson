#include "vision/preprocess.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

namespace
{

__device__ __forceinline__ int clamp_int(int v, int lo, int hi)
{
  return v < lo ? lo : (v > hi ? hi : v);
}

// output_chw 전체를 pad_f로 채우는 커널 (CHW 3채널)
__global__ void FillPadKernel(
    float* __restrict__ output_chw,
    int out_w,
    int out_h,
    float pad_f)
{
  const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= out_w || y >= out_h) return;

  const int idx  = y * out_w + x;
  const int area = out_w * out_h;

  output_chw[idx]            = pad_f; // R
  output_chw[idx + area]     = pad_f; // G
  output_chw[idx + area * 2] = pad_f; // B
}

// resized 영역만 원본에서 샘플링해서 pad 위치로 써줌 (nearest)
__global__ void PreprocessKernel(
    const uint8_t* __restrict__ input_bgr, // device, HWC BGR
    float* __restrict__ output_chw,        // device, CHW RGB float
    int input_w,
    int input_h,
    int resized_w,
    int resized_h,
    int pad_x,
    int pad_y,
    int out_w,
    int out_h,
    float inv_scale)                       // 1/scale
{
  const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= resized_w || y >= resized_h) return;

  // resized 좌표 -> 원본 좌표 (nearest)
  const float src_x_f = (x + 0.5f) * inv_scale - 0.5f;
  const float src_y_f = (y + 0.5f) * inv_scale - 0.5f;

  int sx = (int)lrintf(src_x_f);
  int sy = (int)lrintf(src_y_f);

  sx = clamp_int(sx, 0, input_w - 1);
  sy = clamp_int(sy, 0, input_h - 1);

  const int in_idx = (sy * input_w + sx) * 3;
  const uint8_t bb = input_bgr[in_idx + 0];
  const uint8_t gg = input_bgr[in_idx + 1];
  const uint8_t rr = input_bgr[in_idx + 2];

  const int dst_x = x + pad_x;
  const int dst_y = y + pad_y;
  if (dst_x < 0 || dst_x >= out_w || dst_y < 0 || dst_y >= out_h) return;

  const int out_idx = dst_y * out_w + dst_x;
  const int area    = out_w * out_h;

  // BGR -> RGB, [0..1]
  output_chw[out_idx]            = (float)rr * (1.0f / 255.0f);
  output_chw[out_idx + area]     = (float)gg * (1.0f / 255.0f);
  output_chw[out_idx + area * 2] = (float)bb * (1.0f / 255.0f);
}

} // namespace

extern "C" void PreprocessKernelLauncher(
    const uint8_t* input_bgr,
    int input_w,
    int input_h,
    float* output_chw,
    int resized_w,
    int resized_h,
    int pad_x,
    int pad_y,
    float scale,
    cudaStream_t stream)
{
  // 출력 크기: 640x640 같은 “고정 입력”을 전제로 쓰는 게 안전함
  // (pad_x/pad_y/new_w/new_h 계산은 yolo_trt_engine.cpp에서 하며,
  //  여기서는 out_w/out_h를 엔진 입력 크기로 맞춰야 함)
  //
  // 현재 파이프라인은 input_w_/input_h_ = 640/640을 가정하므로:
  const int out_w = 640;
  const int out_h = 640;

  // 1) pad(114/255)로 전체 채움
  {
    const dim3 threads(16, 16);
    const dim3 blocks((out_w + threads.x - 1) / threads.x,
                      (out_h + threads.y - 1) / threads.y);

    const float pad_f = 114.0f * (1.0f / 255.0f);
    FillPadKernel<<<blocks, threads, 0, stream>>>(output_chw, out_w, out_h, pad_f);
  }

  // 2) resized 영역만 덮어쓰기
  {
    const dim3 threads(16, 16);
    const dim3 blocks((resized_w + threads.x - 1) / threads.x,
                      (resized_h + threads.y - 1) / threads.y);

    const float inv_scale = (scale > 1e-12f) ? (1.0f / scale) : 1.0f;

    PreprocessKernel<<<blocks, threads, 0, stream>>>(
        input_bgr, output_chw,
        input_w, input_h,
        resized_w, resized_h,
        pad_x, pad_y,
        out_w, out_h,
        inv_scale);
  }
}