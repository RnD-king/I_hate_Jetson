#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

static __global__ void FillKernel(float* out, int n, float v)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) out[i] = v;
}

// BGR8(HWC) -> RGB(float, CHW) + letterbox pad + normalize
static __global__ void PreprocessKernel(
    const uint8_t* __restrict__ input_bgr,
    float* __restrict__ output_chw,
    int input_w,
    int input_h,
    int output_w,
    int output_h,
    int resized_w,
    int resized_h,
    int pad_x,
    int pad_y
) {
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= resized_w || y >= resized_h) return;

    // scale from output(resized) -> input
    float scale_x = (float)input_w / (float)resized_w;
    float scale_y = (float)input_h / (float)resized_h;

    int src_x = min((int)(x * scale_x), input_w - 1);
    int src_y = min((int)(y * scale_y), input_h - 1);

    int in_idx = (src_y * input_w + src_x) * 3;
    uint8_t b = input_bgr[in_idx + 0];
    uint8_t g = input_bgr[in_idx + 1];
    uint8_t r = input_bgr[in_idx + 2];

    int dst_x = x + pad_x;
    int dst_y = y + pad_y;
    if (dst_x < 0 || dst_x >= output_w || dst_y < 0 || dst_y >= output_h) return;

    int area = output_w * output_h;
    int out_idx = dst_y * output_w + dst_x;

    output_chw[out_idx + 0 * area] = (float)r / 255.0f;
    output_chw[out_idx + 1 * area] = (float)g / 255.0f;
    output_chw[out_idx + 2 * area] = (float)b / 255.0f;
}

extern "C" void PreprocessKernelLauncher(
    const uint8_t* input_bgr,
    int input_w,
    int input_h,
    float* output_chw,
    int output_w,
    int output_h,
    int resized_w,
    int resized_h,
    int pad_x,
    int pad_y,
    cudaStream_t stream,
    uint8_t pad_value
) {
    // 1) fill whole output with pad_value/255
    float pv = (float)pad_value / 255.0f;
    int total = 3 * output_w * output_h;
    int threads1 = 256;
    int blocks1  = (total + threads1 - 1) / threads1;
    FillKernel<<<blocks1, threads1, 0, stream>>>(output_chw, total, pv);

    // 2) write valid region
    dim3 threads2(16, 16);
    dim3 blocks2((resized_w + 15) / 16, (resized_h + 15) / 16);
    PreprocessKernel<<<blocks2, threads2, 0, stream>>>(
        input_bgr, output_chw,
        input_w, input_h,
        output_w, output_h,
        resized_w, resized_h,
        pad_x, pad_y
    );
}
