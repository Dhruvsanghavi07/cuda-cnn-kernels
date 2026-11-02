// Benchmarks: GEMM kernels vs cuBLAS, and convolution (direct vs im2col+GEMM) on U-Net/ResNet-34 layer shapes.
// Prints Markdown tables; `--out results/<file>.md` also writes them to a file.
#include <cublas_v2.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "cuda_check.h"
#include "kernels.h"

#define CUBLAS_CHECK(call)                                                       \
    do {                                                                         \
        cublasStatus_t st_ = (call);                                             \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                      \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", int(st_), __FILE__, __LINE__); \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

static std::string report;
static void out(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static void out(const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    std::fputs(buf, stdout);
    report += buf;
}

static float* device_random(size_t n, unsigned seed) {
    std::vector<float> h(n);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (auto& x : h) x = d(gen);
    float* p;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(p, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    return p;
}

// Median time in milliseconds over `reps` runs after `warmup` runs, timed with CUDA events.
static float time_ms(const std::function<void()>& fn, int warmup = 3, int reps = 20) {
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times;
    for (int i = 0; i < reps; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        fn();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::nth_element(times.begin(), times.begin() + reps / 2, times.end());
    return times[reps / 2];
}

static void bench_gemm(cublasHandle_t handle) {
    out("\n## GEMM (FP32, square M = N = K)\n\nGFLOP/s, higher is better. Percent of cuBLAS in brackets.\n\n");
    out("| Size | Naive | Shared-memory tiled | Register-blocked | cuBLAS |\n|---|---|---|---|---|\n");
    for (int sz : {256, 512, 1024, 2048, 4096}) {
        const size_t n2 = size_t(sz) * sz;
        float *a = device_random(n2, 1), *b = device_random(n2, 2), *c;
        CUDA_CHECK(cudaMalloc(&c, n2 * sizeof(float)));
        const double flops = 2.0 * sz * sz * sz;
        const float one = 1.0f, zero = 0.0f;
        // cuBLAS is column-major. Row-major C = A*B is column-major C^T = B^T * A^T, so pass B first.
        const float ms_cublas = time_ms([&] {
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, sz, sz, sz, &one, b, sz, a, sz, &zero, c, sz));
        });
        const double g_cublas = flops / ms_cublas / 1e6;
        out("| %d |", sz);
        for (GemmAlgo algo : {GemmAlgo::Naive, GemmAlgo::Tiled, GemmAlgo::RegBlock}) {
            const float ms = time_ms([&] { gemm(algo, a, b, c, sz, sz, sz); });
            const double g = flops / ms / 1e6;
            out(" %.0f (%.0f%%) |", g, 100.0 * g / g_cublas);
        }
        out(" %.0f |\n", g_cublas);
        cudaFree(a), cudaFree(b), cudaFree(c);
    }
}

static void bench_conv() {
    struct Layer {
        const char* name;
        ConvShape sh;
    };
    // Shapes from a ResNet-34-encoder U-Net on a 1x3x256x256 slice (the zebrafish segmentation model's encoder).
    const Layer layers[] = {
        {"stem 7x7/2, 3->64, 256px", {1, 3, 256, 256, 64, 7, 7, 2, 3}},
        {"enc1 3x3, 64->64, 64px", {1, 64, 64, 64, 64, 3, 3, 1, 1}},
        {"enc2 3x3/2, 64->128, 64px", {1, 64, 64, 64, 128, 3, 3, 2, 1}},
        {"enc3 3x3, 256->256, 16px", {1, 256, 16, 16, 256, 3, 3, 1, 1}},
        {"enc4 3x3, 512->512, 8px", {1, 512, 8, 8, 512, 3, 3, 1, 1}},
        {"dec 3x3, 64->64, 256px", {1, 64, 256, 256, 64, 3, 3, 1, 1}},
        {"batch 8, 3x3, 64->64, 64px", {8, 64, 64, 64, 64, 3, 3, 1, 1}},
    };
    out("\n## Convolution layers (FP32, fused bias + ReLU)\n\nMilliseconds (GFLOP/s), lower time is better.\n\n");
    out("| Layer | Direct | im2col + register-blocked GEMM | Speedup |\n|---|---|---|---|\n");
    for (const auto& l : layers) {
        const auto& sh = l.sh;
        float *in = device_random(sh.input_size(), 3), *w = device_random(sh.weight_size(), 4);
        float *bias = device_random(sh.k, 5), *o, *ws;
        CUDA_CHECK(cudaMalloc(&o, sh.output_size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws, conv2d_im2col_workspace(sh) * sizeof(float)));
        const float t_direct = time_ms([&] { conv2d_direct(in, w, bias, o, sh, true); });
        const float t_im2col = time_ms([&] { conv2d_im2col(in, w, bias, o, ws, sh, true); });
        out("| %s | %.3f (%.0f) | %.3f (%.0f) | %.1fx |\n", l.name, t_direct, sh.flops() / t_direct / 1e6, t_im2col,
            sh.flops() / t_im2col / 1e6, t_direct / t_im2col);
        cudaFree(in), cudaFree(w), cudaFree(bias), cudaFree(o), cudaFree(ws);
    }
}

int main(int argc, char** argv) {
    const char* out_path = nullptr;
    for (int i = 1; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], "--out")) out_path = argv[i + 1];

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int clock_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    out("# Benchmark results\n\nGPU: **%s** (compute %d.%d, %d SMs, %.0f MHz boost, %.1f GB)\n", prop.name, prop.major,
        prop.minor, prop.multiProcessorCount, clock_khz / 1000.0, prop.totalGlobalMem / 1e9);
    out("Median of 20 timed runs after 3 warm-up runs, CUDA events. cuBLAS uses its default FP32 math mode.\n");

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    bench_gemm(handle);
    bench_conv();
    CUBLAS_CHECK(cublasDestroy(handle));

    if (out_path) {
        FILE* f = std::fopen(out_path, "w");
        if (!f) {
            std::perror(out_path);
            return 1;
        }
        std::fputs(report.c_str(), f);
        std::fclose(f);
        std::printf("\nwrote %s\n", out_path);
    }
    return 0;
}
