#include "reference.h"

#include <algorithm>
#include <cmath>

namespace ref {

void gemm(const float* a, const float* b, float* c, int m, int n, int k, const float* bias, bool relu) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double acc = bias ? bias[i] : 0.0;  // accumulate in double: the reference must be the most accurate
            for (int p = 0; p < k; ++p) acc += double(a[size_t(i) * k + p]) * b[size_t(p) * n + j];
            float v = float(acc);
            c[size_t(i) * n + j] = relu ? std::max(v, 0.0f) : v;
        }
    }
}

void conv2d(const float* in, const float* w, const float* bias, float* out, const ConvShape& sh, bool relu) {
    const int oh = sh.oh(), ow = sh.ow();
    for (int n = 0; n < sh.n; ++n)
        for (int k = 0; k < sh.k; ++k)
            for (int y = 0; y < oh; ++y)
                for (int x = 0; x < ow; ++x) {
                    double acc = bias ? bias[k] : 0.0;
                    for (int c = 0; c < sh.c; ++c)
                        for (int r = 0; r < sh.r; ++r)
                            for (int s = 0; s < sh.s; ++s) {
                                const int iy = y * sh.stride - sh.pad + r, ix = x * sh.stride - sh.pad + s;
                                if (iy < 0 || iy >= sh.h || ix < 0 || ix >= sh.w) continue;  // zero padding
                                acc += double(in[((size_t(n) * sh.c + c) * sh.h + iy) * sh.w + ix]) *
                                       w[((size_t(k) * sh.c + c) * sh.r + r) * sh.s + s];
                            }
                    float v = float(acc);
                    out[((size_t(n) * sh.k + k) * oh + y) * ow + x] = relu ? std::max(v, 0.0f) : v;
                }
}

void im2col(const float* img, float* col, const ConvShape& sh) {
    const int oh = sh.oh(), ow = sh.ow();
    for (int c = 0; c < sh.c; ++c)
        for (int r = 0; r < sh.r; ++r)
            for (int s = 0; s < sh.s; ++s) {
                const size_t row = (size_t(c) * sh.r + r) * sh.s + s;
                for (int y = 0; y < oh; ++y)
                    for (int x = 0; x < ow; ++x) {
                        const int iy = y * sh.stride - sh.pad + r, ix = x * sh.stride - sh.pad + s;
                        const bool inside = iy >= 0 && iy < sh.h && ix >= 0 && ix < sh.w;
                        col[row * oh * ow + size_t(y) * ow + x] = inside ? img[(size_t(c) * sh.h + iy) * sh.w + ix] : 0.0f;
                    }
            }
}

double max_rel_error(const std::vector<float>& a, const std::vector<float>& b) {
    double worst = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        worst = std::max(worst, std::abs(double(a[i]) - b[i]) / std::max(1.0, std::abs(double(b[i]))));
    return a.size() == b.size() ? worst : INFINITY;
}

}  // namespace ref
