#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#ifndef _CUDA_HPP_
#define _CUDA_HPP_

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s: %d, ", __FILE__, __LINE__); \
        printf("Code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(0); \
    } \
}

#endif