#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// 전처리 커널 런처
// - 입력: HWC BGR uint8 (device)
// - 출력: CHW RGB float32, [0..1] 정규화 + letterbox pad(114)
// - resized_w/h + pad_x/y는 yolo_trt_engine.cpp에서 계산해서 넘김
extern "C" void PreprocessKernelLauncher(
    const uint8_t* input_bgr,   // device, HWC BGR8
    int input_w,
    int input_h,
    float* output_chw,          // device, CHW float32
    int resized_w,
    int resized_h,
    int pad_x,
    int pad_y,
    float scale,
    cudaStream_t stream
);