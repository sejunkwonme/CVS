#pragma once
#include "cuda_intelisense.hpp"
#include "cuda_intellisense_attribute.hpp"

// CUDA kernel 선언
__global__ void myKernel(float* data, int N);

// CUDA wrapper 함수 선언 (optional, 매우 추천)
void launchMyKernel(float* data, int N, cudaStream_t stream = 0);