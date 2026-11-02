// CPU-only tests of the reference implementations (build and run without a GPU).
// The GPU tests trust these, so they are checked against hand-computed values and against each other.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "reference.h"

static int failures = 0;
#define EXPECT(cond, ...)                                     \
    do {                                                      \
        if (!(cond)) {                                        \
            ++failures;                                       \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__);  \
            std::printf(__VA_ARGS__);                         \
            std::printf("\n");                                \
        }                                                     \
    } while (0)

static std::vector<float> random_vec(size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = d(gen);
    return v;
}

static void test_gemm_hand_computed() {
    // [1 2 3]   [7  8 ]   [ 58  64]
    // [4 5 6] x [9  10] = [139 154]
    //           [11 12]
    std::vector<float> a{1, 2, 3, 4, 5, 6}, b{7, 8, 9, 10, 11, 12}, c(4);
    ref::gemm(a.data(), b.data(), c.data(), 2, 2, 3);
    EXPECT(c == (std::vector<float>{58, 64, 139, 154}), "gemm values");

    std::vector<float> bias{-100, -200};
    ref::gemm(a.data(), b.data(), c.data(), 2, 2, 3, bias.data(), true);
    EXPECT(c == (std::vector<float>{0, 0, 0, 0}), "bias + relu clamps negatives: %g %g", c[0], c[2]);
}

static void test_conv_hand_computed() {
    // 3x3 input 1..9, 2x2 kernel of ones, no padding -> each output is the sum of a 2x2 window.
    ConvShape sh{1, 1, 3, 3, 1, 2, 2, 1, 0};
    std::vector<float> in{1, 2, 3, 4, 5, 6, 7, 8, 9}, w{1, 1, 1, 1}, out(sh.output_size());
    ref::conv2d(in.data(), w.data(), nullptr, out.data(), sh);
    EXPECT(out == (std::vector<float>{12, 16, 24, 28}), "conv values");

    sh.pad = 1;  // padded corners only see one input pixel
    out.assign(sh.output_size(), 0);
    ref::conv2d(in.data(), w.data(), nullptr, out.data(), sh);
    EXPECT(sh.oh() == 4 && out[0] == 1 && out[15] == 9, "padding: %g %g", out[0], out[15]);
}

static void test_im2col_gemm_equals_direct_conv() {
    const ConvShape shapes[] = {
        {2, 3, 7, 9, 5, 3, 3, 1, 1},   // odd sizes, "same" padding
        {1, 4, 8, 8, 6, 3, 3, 2, 1},   // stride 2
        {1, 3, 11, 11, 4, 7, 7, 2, 3}, // ResNet-style 7x7 stem
        {3, 2, 5, 5, 2, 1, 1, 1, 0},   // 1x1 convolution
    };
    for (const auto& sh : shapes) {
        auto in = random_vec(sh.input_size(), 1), w = random_vec(sh.weight_size(), 2), b = random_vec(sh.k, 3);
        std::vector<float> direct(sh.output_size()), lowered(sh.output_size());
        ref::conv2d(in.data(), w.data(), b.data(), direct.data(), sh, true);
        const int crs = sh.c * sh.r * sh.s, pix = sh.oh() * sh.ow();
        std::vector<float> col(size_t(crs) * pix);
        for (int n = 0; n < sh.n; ++n) {
            ref::im2col(in.data() + size_t(n) * sh.c * sh.h * sh.w, col.data(), sh);
            ref::gemm(w.data(), col.data(), lowered.data() + size_t(n) * sh.k * pix, sh.k, pix, crs, b.data(), true);
        }
        const double err = ref::max_rel_error(lowered, direct);
        EXPECT(err < 1e-5, "im2col+gemm vs direct, shape c=%d h=%d r=%d stride=%d: err %g", sh.c, sh.h, sh.r,
               sh.stride, err);
    }
}

static void test_flops_and_shapes() {
    ConvShape sh{1, 64, 128, 128, 64, 3, 3, 1, 1};
    EXPECT(sh.oh() == 128 && sh.ow() == 128, "same padding keeps size");
    EXPECT(std::abs(sh.flops() - 2.0 * 64 * 128 * 128 * 64 * 9) < 1, "flop count");
}

int main() {
    test_gemm_hand_computed();
    test_conv_hand_computed();
    test_im2col_gemm_equals_direct_conv();
    test_flops_and_shapes();
    std::printf(failures ? "%d reference test(s) FAILED\n" : "all reference tests passed\n", failures);
    return failures ? 1 : 0;
}
