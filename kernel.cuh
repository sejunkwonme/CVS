#pragma once
#include "cuda_intelisense.hpp"
#include "cuda_intellisense_attribute.hpp"

__device__ unsigned char clamp(int x);
__global__ void YUV2RGB(unsigned char* input, unsigned char* GUI_IMAGE, float* ML_IMAGE);
__global__ void CROP(unsigned char* input, unsigned char* GUI_IMAGE_FINAL, float* ML_IMAGE_FINAL);

void launchYUV2RGB(unsigned char* input, unsigned char* GUI_IMAGE, float* ML_IMAGE, cudaStream_t stream);
void launchCROP(unsigned char* gui_input, float* ml_input, unsigned char* GUI_IMAGE, float* ML_IMAGE, cudaStream_t stream);