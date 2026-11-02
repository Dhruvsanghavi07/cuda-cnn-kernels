// Two ways to run a convolution on the GPU.
//
// Direct:  one thread per output pixel loops over C*R*S inputs. Simple, no extra memory, but every
//          input pixel is re-read from global memory by up to R*S*K threads.
// im2col:  copy each input patch into a column of a [C*R*S x OH*OW] matrix, then the whole layer is
//          one GEMM: W[K x C*R*S] * col. This trades extra memory (R*S copies of the input) for the
//          data reuse of the register-blocked GEMM. Bias and ReLU are applied in the GEMM epilogue, so
//          the output is written once instead of three times.
#include "cuda_check.h"
#include "kernels.h"

namespace {

__global__ void conv2d_direct_kernel(const float* __restrict__ in, const float* __restrict__ w,
                                     const float* __restrict__ bias, float* __restrict__ out, ConvShape sh,
                                     bool relu) {
    const int oh = sh.oh(), ow = sh.ow();
    const size_t total = size_t(sh.n) * sh.k * oh * ow;
    for (size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x; idx < total;
         idx += size_t(gridDim.x) * blockDim.x) {  // grid-stride loop: any size works with any grid
        const int x = idx % ow, y = (idx / ow) % oh, k = (idx / (size_t(ow) * oh)) % sh.k;
        const int n = idx / (size_t(ow) * oh * sh.k);
        float acc = bias ? bias[k] : 0.0f;
        for (int c = 0; c < sh.c; ++c) {
            const float* ip = in + (size_t(n) * sh.c + c) * sh.h * sh.w;
            const float* wp = w + (size_t(k) * sh.c + c) * sh.r * sh.s;
            for (int r = 0; r < sh.r; ++r) {
                const int iy = y * sh.stride - sh.pad + r;
                if (iy < 0 || iy >= sh.h) continue;
                for (int s = 0; s < sh.s; ++s) {
                    const int ix = x * sh.stride - sh.pad + s;
                    if (ix >= 0 && ix < sh.w) acc += ip[iy * sh.w + ix] * wp[r * sh.s + s];
                }
            }
        }
        out[idx] = relu ? fmaxf(acc, 0.0f) : acc;
    }
}

// One thread per element of the column matrix; consecutive threads write consecutive output pixels.
__global__ void im2col_kernel(const float* __restrict__ img, float* __restrict__ col, ConvShape sh) {
    const int oh = sh.oh(), ow = sh.ow();
    const size_t cols = size_t(oh) * ow, total = size_t(sh.c) * sh.r * sh.s * cols;
    for (size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x; idx < total;
         idx += size_t(gridDim.x) * blockDim.x) {
        const size_t row = idx / cols, pix = idx % cols;
        const int s = row % sh.s, r = (row / sh.s) % sh.r, c = row / (size_t(sh.s) * sh.r);
        const int y = pix / ow, x = pix % ow;
        const int iy = y * sh.stride - sh.pad + r, ix = x * sh.stride - sh.pad + s;
        col[idx] = (iy >= 0 && iy < sh.h && ix >= 0 && ix < sh.w) ? img[(size_t(c) * sh.h + iy) * sh.w + ix] : 0.0f;
    }
}

unsigned grid_for(size_t total, int block) {
    const size_t blocks = (total + block - 1) / block;
    return unsigned(blocks < 65535 * 32 ? blocks : 65535 * 32);
}

}  // namespace

void conv2d_direct(const float* in, const float* w, const float* bias, float* out, const ConvShape& sh, bool relu,
                   cudaStream_t stream) {
    conv2d_direct_kernel<<<grid_for(sh.output_size(), 256), 256, 0, stream>>>(in, w, bias, out, sh, relu);
    CUDA_CHECK_LAUNCH();
}

size_t conv2d_im2col_workspace(const ConvShape& sh) { return size_t(sh.c) * sh.r * sh.s * sh.oh() * sh.ow(); }

void conv2d_im2col(const float* in, const float* w, const float* bias, float* out, float* workspace,
                   const ConvShape& sh, bool relu, cudaStream_t stream) {
    const int crs = sh.c * sh.r * sh.s, pixels = sh.oh() * sh.ow();
    for (int n = 0; n < sh.n; ++n) {  // one image at a time keeps the workspace small
        const float* img = in + size_t(n) * sh.c * sh.h * sh.w;
        im2col_kernel<<<grid_for(size_t(crs) * pixels, 256), 256, 0, stream>>>(img, workspace, sh);
        CUDA_CHECK_LAUNCH();
        gemm(GemmAlgo::RegBlock, w, workspace, out + size_t(n) * sh.k * pixels, sh.k, pixels, crs, bias, relu, stream);
    }
}
