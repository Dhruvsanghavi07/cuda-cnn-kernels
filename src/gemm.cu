// Three GEMM kernels, each removing the main bottleneck of the one before it.
//
// Naive:    every multiply-add reads two floats from global memory, so the kernel is limited by
//           memory bandwidth, not arithmetic.
// Tiled:    a 32x32 block loads a tile of A and a tile of B into shared memory once, then every
//           thread reuses them 32 times. Global traffic drops by about 32x.
// RegBlock: each thread computes a 4x4 patch of C held in registers. Per K step it reads 4 values of
//           A and 4 of B from shared memory and does 16 multiply-adds, so shared-memory traffic per
//           FLOP drops 4x as well. A is stored transposed in shared memory so the 4 reads are contiguous.
#include "cuda_check.h"
#include "kernels.h"

namespace {

__device__ __forceinline__ float epilogue(float v, const float* bias, int row, bool relu) {
    if (bias) v += bias[row];
    return relu ? fmaxf(v, 0.0f) : v;
}

__global__ void gemm_naive(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c,
                           int m, int n, int k, const float* __restrict__ bias, bool relu) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // x is the fast index -> coalesced B and C access
    if (row >= m || col >= n) return;
    float acc = 0.0f;
    for (int p = 0; p < k; ++p) acc += a[size_t(row) * k + p] * b[size_t(p) * n + col];
    c[size_t(row) * n + col] = epilogue(acc, bias, row, relu);
}

constexpr int TILE = 32;

__global__ void gemm_tiled(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c,
                           int m, int n, int k, const float* __restrict__ bias, bool relu) {
    __shared__ float as[TILE][TILE];
    __shared__ float bs[TILE][TILE + 1];  // +1 column of padding avoids shared-memory bank conflicts
    const int ty = threadIdx.y, tx = threadIdx.x;
    const int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
    float acc = 0.0f;
    for (int k0 = 0; k0 < k; k0 += TILE) {
        as[ty][tx] = (row < m && k0 + tx < k) ? a[size_t(row) * k + k0 + tx] : 0.0f;  // zero-fill edges
        bs[ty][tx] = (k0 + ty < k && col < n) ? b[size_t(k0 + ty) * n + col] : 0.0f;
        __syncthreads();  // tile fully loaded before anyone reads it
#pragma unroll
        for (int p = 0; p < TILE; ++p) acc += as[ty][p] * bs[p][tx];
        __syncthreads();  // everyone done reading before the next tile overwrites it
    }
    if (row < m && col < n) c[size_t(row) * n + col] = epilogue(acc, bias, row, relu);
}

constexpr int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
constexpr int REG_THREADS = (BM / TM) * (BN / TN);  // 256

__global__ void __launch_bounds__(REG_THREADS)
    gemm_regblock(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int m, int n,
                  int k, const float* __restrict__ bias, bool relu) {
    __shared__ float as[BK][BM];  // transposed: as[p][row]
    __shared__ float bs[BK][BN];
    const int t = threadIdx.x;
    const int trow = (t / (BN / TN)) * TM, tcol = (t % (BN / TN)) * TN;  // this thread's 4x4 patch
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;

    float acc[TM][TN] = {};
    float ra[TM], rb[TN];
    for (int k0 = 0; k0 < k; k0 += BK) {
        for (int i = t; i < BM * BK; i += REG_THREADS) {  // 512 elements of A, 2 per thread
            const int r = i / BK, p = i % BK, gr = row0 + r, gp = k0 + p;
            as[p][r] = (gr < m && gp < k) ? a[size_t(gr) * k + gp] : 0.0f;
        }
        for (int i = t; i < BK * BN; i += REG_THREADS) {  // 512 elements of B, consecutive threads -> consecutive columns
            const int p = i / BN, cc = i % BN, gp = k0 + p, gc = col0 + cc;
            bs[p][cc] = (gp < k && gc < n) ? b[size_t(gp) * n + gc] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < BK; ++p) {
#pragma unroll
            for (int i = 0; i < TM; ++i) ra[i] = as[p][trow + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) rb[j] = bs[p][tcol + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += ra[i] * rb[j];  // outer product in registers
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int r = row0 + trow + i, cc = col0 + tcol + j;
            if (r < m && cc < n) c[size_t(r) * n + cc] = epilogue(acc[i][j], bias, r, relu);
        }
}

int ceil_div(int a, int b) { return (a + b - 1) / b; }

}  // namespace

const char* gemm_name(GemmAlgo algo) {
    switch (algo) {
        case GemmAlgo::Naive: return "naive";
        case GemmAlgo::Tiled: return "smem-tiled";
        case GemmAlgo::RegBlock: return "register-blocked";
    }
    return "?";
}

void gemm(GemmAlgo algo, const float* a, const float* b, float* c, int m, int n, int k, const float* bias, bool relu,
          cudaStream_t stream) {
    if (m <= 0 || n <= 0) return;
    switch (algo) {
        case GemmAlgo::Naive: {
            dim3 block(32, 8), grid(ceil_div(n, 32), ceil_div(m, 8));
            gemm_naive<<<grid, block, 0, stream>>>(a, b, c, m, n, k, bias, relu);
            break;
        }
        case GemmAlgo::Tiled: {
            dim3 block(TILE, TILE), grid(ceil_div(n, TILE), ceil_div(m, TILE));
            gemm_tiled<<<grid, block, 0, stream>>>(a, b, c, m, n, k, bias, relu);
            break;
        }
        case GemmAlgo::RegBlock: {
            dim3 grid(ceil_div(n, BN), ceil_div(m, BM));
            gemm_regblock<<<grid, REG_THREADS, 0, stream>>>(a, b, c, m, n, k, bias, relu);
            break;
        }
    }
    CUDA_CHECK_LAUNCH();
}
