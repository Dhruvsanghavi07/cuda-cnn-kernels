// CPU reference implementations. Every GPU kernel is tested against these.
#pragma once
#include <cstddef>
#include <vector>

// ConvShape is used on both host and device; plain g++ builds (CPU tests) don't know __host__ __device__.
#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

struct ConvShape {
    int n, c, h, w;      // input: batch, channels, height, width (NCHW)
    int k, r, s;         // filters: output channels, kernel height, kernel width (KCRS)
    int stride = 1, pad = 0;
    HOST_DEVICE int oh() const { return (h + 2 * pad - r) / stride + 1; }
    HOST_DEVICE int ow() const { return (w + 2 * pad - s) / stride + 1; }
    size_t input_size() const { return size_t(n) * c * h * w; }
    size_t weight_size() const { return size_t(k) * c * r * s; }
    size_t output_size() const { return size_t(n) * k * oh() * ow(); }
    // Multiply-adds count twice (one multiply, one add).
    double flops() const { return 2.0 * n * k * oh() * ow() * double(c) * r * s; }
};

namespace ref {

// C[M x N] = A[M x K] * B[K x N] (+ bias[row]) (then ReLU), all row-major.
void gemm(const float* a, const float* b, float* c, int m, int n, int k, const float* bias = nullptr,
          bool relu = false);

// Direct convolution: out[n][k][oh][ow] = sum over c, r, s of in * w (+ bias[k]) (then ReLU).
void conv2d(const float* in, const float* w, const float* bias, float* out, const ConvShape& sh, bool relu = false);

// Lower one image to a [C*R*S x OH*OW] matrix so convolution becomes one GEMM: W[K x CRS] * col.
void im2col(const float* img, float* col, const ConvShape& sh);

// Largest |a - b| relative to max(1, |b|); used by every correctness test.
double max_rel_error(const std::vector<float>& a, const std::vector<float>& b);

}  // namespace ref
