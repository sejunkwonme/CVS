#pragma once
#include "cuda_intelisense.hpp"
#include "cuda_intellisense_attribute.hpp"
__device__ unsigned char clamp(int x);
__global__ void YUV2RGB(unsigned char* input, unsigned char* GUI_IMAGE, float* ML_IMAGE);
__global__ void CROP(unsigned char* input, unsigned char* GUI_IMAGE_FINAL, float* ML_IMAGE_FINAL);
__global__ void preprocess(unsigned char* input, unsigned char* gui_image, float* ml_image);
void launchYUV2RGB(unsigned char* input, unsigned char* GUI_IMAGE, float* ML_IMAGE, cudaStream_t stream);
void launchCROP(unsigned char* gui_input, float* ml_input, unsigned char* GUI_IMAGE, float* ML_IMAGE, cudaStream_t stream);
void launchPREPROCESS(unsigned char* input, unsigned char* gui_image, float* ml_image, cudaStream_t stream);