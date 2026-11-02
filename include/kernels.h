// GPU kernels for the two operations that dominate CNN inference time: GEMM and 2D convolution.
// All pointers are device pointers; all matrices are row-major float32.
#pragma once
#include <cuda_runtime.h>

#include "reference.h"

enum class GemmAlgo {
    Naive,     // one thread per output element, every operand read from global memory
    Tiled,     // 32x32 tiles staged in shared memory: each global element is loaded once per tile
    RegBlock,  // 64x64 block tile, 8-deep K slices, each thread computes a 4x4 sub-tile in registers
};

const char* gemm_name(GemmAlgo algo);

// C = A * B (+ bias[row]) (then ReLU). Handles any M, N, K (edges are zero-padded in shared memory).
void gemm(GemmAlgo algo, const float* a, const float* b, float* c, int m, int n, int k,
          const float* bias = nullptr, bool relu = false, cudaStream_t stream = 0);

// Direct convolution: one thread per output element.
void conv2d_direct(const float* in, const float* w, const float* bias, float* out, const ConvShape& sh,
                   bool relu = false, cudaStream_t stream = 0);

// Convolution as im2col + register-blocked GEMM, with bias and ReLU fused into the GEMM epilogue.
// `workspace` must hold conv2d_im2col_workspace(sh) floats.
size_t conv2d_im2col_workspace(const ConvShape& sh);
void conv2d_im2col(const float* in, const float* w, const float* bias, float* out, float* workspace,
                   const ConvShape& sh, bool relu = false, cudaStream_t stream = 0);
