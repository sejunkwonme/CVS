#pragma once
// **********실제 코드 컴파일시에는 모두 무시됨**********
#ifdef __INTELLISENSE__

// __CUDACC__ 가 define 된 것"처럼" 보여주기
#define __CUDACC__

// CUDA 관련 매크로들을 공백으로 재정의한 것"처럼" 보여주기
#define __global__ 
#define __host__ 
#define __device__ 
#define __device_builtin__
#define __device_builtin_texture_type__
#define __device_builtin_surface_type__
#define __cudart_builtin__
#define __constant__ 
#define __shared__ 
#define __restrict__
#define __noinline__
#define __forceinline__
#define __managed__

#endif 
