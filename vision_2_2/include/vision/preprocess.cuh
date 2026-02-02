#pragma once
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void PreprocessKernelLauncher(
    const std::uint8_t* input_bgr,  // HWC, BGR8
    int input_w,
    int input_h,
    float* output_chw,              // CHW, float32, (3*output_w*output_h)
    int output_w,                   // e.g. 640
    int output_h,                   // e.g. 640
    int resized_w,                  // letterbox resized width
    int resized_h,                  // letterbox resized height
    int pad_x,                      // left pad
    int pad_y,                      // top pad
    cudaStream_t stream,
    std::uint8_t pad_value          // typically 114
);
