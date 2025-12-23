#include "kernel.cuh"
#include "cuda_intelisense.hpp"
#include "cuda_intellisense_attribute.hpp"
#include <cuda_runtime.h>
__device__ unsigned char clamp(int x) {
    return x < 0 ? 0 : (x > 255 ? 255 : x);
}
__global__ void YUV2RGB(unsigned char* input, unsigned char* GUI_IMAGE, float* ML_IMAGE) {
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * blockDim.y + threadIdx.y;
    int yuv_xidx = xidx * 2;
    int Y, U, V;
    int C, D, E;
    int R, G, B;
    unsigned char r, g, b;
    Y = input[1280 * yidx + yuv_xidx];
    if (xidx % 2 == 0) {
        U = input[1280 * yidx + yuv_xidx + 1];
        V = input[1280 * yidx + yuv_xidx + 3];
    }
    else {
        U = input[1280 * yidx + yuv_xidx - 1];
        V = input[1280 * yidx + yuv_xidx + 1];
    }
    C = Y - 16;
    D = U - 128;   // Cb
    E = V - 128;   // Cr
    R = (298 * C + 409 * E + 128) >> 8;
    G = (298 * C - 100 * D - 208 * E + 128) >> 8;
    B = (298 * C + 516 * D + 128) >> 8;
    r = clamp(R);
    g = clamp(G);
    b = clamp(B);
    float rf = r * (1.0f / 255.0f);
    float gf = g * (1.0f / 255.0f);
    float bf = b * (1.0f / 255.0f);
    // planar rgb 형식으로 ML 전용 이미지 생성
    int rgb_base = 640 * 480;
    ML_IMAGE[640 * yidx + xidx] = rf;
    ML_IMAGE[rgb_base + 640 * yidx + xidx] = gf;
    ML_IMAGE[rgb_base * 2 + 640 * yidx + xidx] = bf;
    // interleaved rgb 형식으로 QLabel 출력 전용 이미지 생성
    int interleaved_base = (yidx * 640 + xidx) * 3;
    GUI_IMAGE[interleaved_base + 0] = b; // cv::Mat 기본 형식이 bgr임 불필요한 변환 방지가능성..?
    GUI_IMAGE[interleaved_base + 1] = g;
    GUI_IMAGE[interleaved_base + 2] = r;
}
__global__ void CROP(unsigned char* gui_input, float* ml_input, unsigned char* GUI_IMAGE_FINAL, float* ML_IMAGE_FINAL) {
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * blockDim.y + threadIdx.y;
    int crop_base_x = (640 - 448) / 2;
    int crop_base_y = (480 - 448) / 2;
    int ml_base = 448 * 448;
    int input_base = 640 * 480;
    ML_IMAGE_FINAL[448 * yidx + xidx] = ml_input[640 * crop_base_y + crop_base_x + 640 * yidx + xidx];
    ML_IMAGE_FINAL[ml_base + 448 * yidx + xidx] = ml_input[input_base + 640 * crop_base_y + crop_base_x + 640 * yidx + xidx];
    ML_IMAGE_FINAL[ml_base * 2 + 448 * yidx + xidx] = ml_input[input_base * 2 + 640 * crop_base_y + crop_base_x + 640 * yidx + xidx];
    int in_idx = ((crop_base_y + yidx) * 640 + (crop_base_x + xidx)) * 3;
    int out_idx = (yidx * 448 + xidx) * 3;
    GUI_IMAGE_FINAL[out_idx + 0] = gui_input[in_idx + 0];
    GUI_IMAGE_FINAL[out_idx + 1] = gui_input[in_idx + 1];
    GUI_IMAGE_FINAL[out_idx + 2] = gui_input[in_idx + 2];
}
// yuv2 변환과 crop을 하나로 합친 커널
// 그리고 work item 을 대폭 줄임.. 성능향상을 위해
__global__ void preprocess(unsigned char* input, unsigned char* gui_image, float* ml_image) {
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * blockDim.y + threadIdx.y;
    int crop_base_x = (1280 - 896) / 2;
    int crop_base_y = (480 - 448) / 2;
    int yuv_xidx = xidx * 2;
    int yuv_yidx = yidx * 16;
    int Y, U, V;
    int C, D, E;
    int R, G, B;
    unsigned char r, g, b;
    for (int i = 0; i < 16; i++) {
        Y = input[1280 * (crop_base_y + yuv_yidx + i) + (crop_base_x + yuv_xidx)];
        if (xidx % 2 == 0) {
            U = input[1280 * (crop_base_y + yuv_yidx + i) + crop_base_x + yuv_xidx + 1];
            V = input[1280 * (crop_base_y + yuv_yidx + i) + crop_base_x + yuv_xidx + 3];
        }
        else {
            U = input[1280 * (crop_base_y + yuv_yidx + i) + crop_base_x + yuv_xidx - 1];
            V = input[1280 * (crop_base_y + yuv_yidx + i) + crop_base_x + yuv_xidx + 1];
        }
        C = Y - 16;
        D = U - 128;   // Cb
        E = V - 128;   // Cr
        R = (298 * C + 409 * E + 128) >> 8;
        G = (298 * C - 100 * D - 208 * E + 128) >> 8;
        B = (298 * C + 516 * D + 128) >> 8;
        r = min(255, max(0, R));
        g = min(255, max(0, G));
        b = min(255, max(0, B));
        float rf = r * (1.0f / 255.0f);
        float gf = g * (1.0f / 255.0f);
        float bf = b * (1.0f / 255.0f);
        int interleaved_base = ((yuv_yidx + i) * 448 + xidx) * 3;
        gui_image[interleaved_base + 0] = b;
        gui_image[interleaved_base + 1] = g;
        gui_image[interleaved_base + 2] = r;
        int rgb_base = 448 * 448;
        ml_image[448 * (yuv_yidx + i) + xidx] = rf;
        ml_image[rgb_base + 448 * (yuv_yidx + i) + xidx] = gf;
        ml_image[(2 * rgb_base) + (448 * (yuv_yidx + i)) + xidx] = bf;
    }
}
void launchYUV2RGB(unsigned char* input, unsigned char* GUI_IMAGE, float* ML_IMAGE, cudaStream_t stream) {
    dim3 thread_in_block(32, 32);
    dim3 block_in_gird(20, 15);
    YUV2RGB KERNEL_ARG4(block_in_gird, thread_in_block, 0, stream) (input, GUI_IMAGE, ML_IMAGE);
}
void launchCROP(unsigned char* gui_input, float* ml_input, unsigned char* GUI_IMAGE, float* ML_IMAGE, cudaStream_t stream) {
    dim3 thread_in_block(32, 32);
    dim3 block_in_gird(14, 14);
    CROP KERNEL_ARG4(block_in_gird, thread_in_block, 0, stream) (gui_input, ml_input, GUI_IMAGE, ML_IMAGE);
}
void launchPREPROCESS(unsigned char* input, unsigned char* gui_image, float* ml_image, cudaStream_t stream) {
    dim3 thread_in_block(64, 1);
    dim3 block_in_gird(7, 28);
    preprocess KERNEL_ARG4(block_in_gird, thread_in_block, 0, stream) (input, gui_image, ml_image);
}