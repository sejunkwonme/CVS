#include "kernel.cuh"
#include "cuda_intelisense.hpp"
#include "cuda_intellisense_attribute.hpp"
#include <cuda_runtime.h>

__global__ void myKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx == 0) {
        for (int i = 0; i < 2000; i++) { asm(""); }
    }
    if (idx < N) data[idx] = data[idx] * 2.0f;
}

// 커널 런처 (C++ 코드가 CUDA 커널을 부드럽게 호출할 수 있게 해주는 함수)
void launchMyKernel(float* data, int N, cudaStream_t stream) {
    int block = 256;
    int grid = (N + block - 1) / block;
    myKernel KERNEL_ARG4(grid, block, 0, stream) (data, N);
}