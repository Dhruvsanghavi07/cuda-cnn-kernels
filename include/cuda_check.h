#pragma once
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        cudaError_t err_ = (call);                                                             \
        if (err_ != cudaSuccess) {                                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_), __FILE__, \
                         __LINE__, cudaGetErrorString(err_));                                  \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

// Kernel launches don't return errors; check right after the launch.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())
