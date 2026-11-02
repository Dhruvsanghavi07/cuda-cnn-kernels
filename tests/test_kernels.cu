// GPU kernels vs the CPU reference on many shapes, including sizes that are not multiples of any tile.
#include <cstdio>
#include <random>
#include <vector>

#include "cuda_check.h"
#include "kernels.h"

static int failures = 0, checks = 0;

static std::vector<float> random_vec(size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = d(gen);
    return v;
}

struct Dev {  // minimal RAII device buffer
    float* p = nullptr;
    explicit Dev(size_t n) { CUDA_CHECK(cudaMalloc(&p, n * sizeof(float) + 1)); }
    Dev(const std::vector<float>& v) : Dev(v.size()) {
        CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    ~Dev() { cudaFree(p); }
    std::vector<float> get(size_t n) const {
        std::vector<float> v(n);
        CUDA_CHECK(cudaMemcpy(v.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost));
        return v;
    }
};

// float32 GEMM error grows roughly with sqrt(K); 1e-4 relative is loose enough for K up to a few thousand.
static void check(const char* what, const std::vector<float>& got, const std::vector<float>& want, double tol = 1e-4) {
    ++checks;
    const double err = ref::max_rel_error(got, want);
    if (!(err <= tol)) {
        ++failures;
        std::printf("FAIL %-60s max rel error %.3g\n", what, err);
    }
}

static void test_gemm() {
    const int sizes[][3] = {{1, 1, 1},     {7, 5, 3},       {32, 32, 32},   {64, 64, 64},  {65, 63, 66},
                            {100, 1, 300}, {1, 257, 129},   {128, 96, 200}, {255, 257, 8}, {300, 200, 1000},
                            {512, 512, 512}, {33, 1000, 17}};
    unsigned seed = 0;
    for (auto& s : sizes) {
        const int m = s[0], n = s[1], k = s[2];
        auto a = random_vec(size_t(m) * k, ++seed), b = random_vec(size_t(k) * n, ++seed), bias = random_vec(m, ++seed);
        for (bool fused : {false, true}) {
            std::vector<float> want(size_t(m) * n);
            ref::gemm(a.data(), b.data(), want.data(), m, n, k, fused ? bias.data() : nullptr, fused);
            Dev da(a), db(b), dbias(bias), dc(size_t(m) * n);
            for (GemmAlgo algo : {GemmAlgo::Naive, GemmAlgo::Tiled, GemmAlgo::RegBlock}) {
                CUDA_CHECK(cudaMemset(dc.p, 0xFF, size_t(m) * n * sizeof(float)));  // NaN garbage: unwritten outputs fail
                gemm(algo, da.p, db.p, dc.p, m, n, k, fused ? dbias.p : nullptr, fused);
                CUDA_CHECK(cudaDeviceSynchronize());
                char name[128];
                std::snprintf(name, sizeof name, "gemm %s %dx%dx%d%s", gemm_name(algo), m, n, k, fused ? " +bias+relu" : "");
                check(name, dc.get(size_t(m) * n), want);
            }
        }
    }
}

static void test_conv() {
    const ConvShape shapes[] = {
        {1, 1, 5, 5, 1, 3, 3, 1, 1},       {2, 3, 7, 9, 5, 3, 3, 1, 1},     {1, 4, 8, 8, 6, 3, 3, 2, 1},
        {1, 3, 33, 33, 8, 7, 7, 2, 3},     {3, 2, 5, 5, 2, 1, 1, 1, 0},     {1, 16, 31, 17, 33, 3, 3, 1, 0},
        {2, 64, 32, 32, 64, 3, 3, 1, 1},   {1, 3, 64, 64, 64, 7, 7, 2, 3},  {1, 128, 16, 16, 256, 3, 3, 2, 1},
    };
    unsigned seed = 100;
    for (const auto& sh : shapes) {
        auto in = random_vec(sh.input_size(), ++seed), w = random_vec(sh.weight_size(), ++seed);
        auto bias = random_vec(sh.k, ++seed);
        std::vector<float> want(sh.output_size());
        ref::conv2d(in.data(), w.data(), bias.data(), want.data(), sh, true);
        Dev din(in), dw(w), dbias(bias), dout(sh.output_size()), ws(conv2d_im2col_workspace(sh));
        char name[160];

        CUDA_CHECK(cudaMemset(dout.p, 0xFF, sh.output_size() * sizeof(float)));
        conv2d_direct(din.p, dw.p, dbias.p, dout.p, sh, true);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::snprintf(name, sizeof name, "conv direct n%d c%d %dx%d k%d %dx%d s%d p%d", sh.n, sh.c, sh.h, sh.w, sh.k, sh.r,
                      sh.s, sh.stride, sh.pad);
        check(name, dout.get(sh.output_size()), want);

        CUDA_CHECK(cudaMemset(dout.p, 0xFF, sh.output_size() * sizeof(float)));
        conv2d_im2col(din.p, dw.p, dbias.p, dout.p, ws.p, sh, true);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::snprintf(name, sizeof name, "conv im2col n%d c%d %dx%d k%d %dx%d s%d p%d", sh.n, sh.c, sh.h, sh.w, sh.k, sh.r,
                      sh.s, sh.stride, sh.pad);
        check(name, dout.get(sh.output_size()), want);
    }
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s (compute %d.%d)\n", prop.name, prop.major, prop.minor);
    test_gemm();
    test_conv();
    if (failures)
        std::printf("%d of %d GPU checks FAILED\n", failures, checks);
    else
        std::printf("all %d GPU checks passed\n", checks);
    return failures ? 1 : 0;
}
