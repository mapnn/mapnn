#include "Matrix.hpp"
namespace MNN {
namespace Math {
void Matrix::multi(float* c, const float* a, const float* b, int h, int k, int w) {

    const int aw = k;
    const int bw = w;
    const int cw = w;

    int y = 0;
    for (; y < h; ++y) {
        int x            = 0;
        const auto aLine = a + y * aw;
        auto cLine       = c + y * cw;
#ifdef MNN_USE_NEON
        // firstly, compute 16 together
        for (; x <= w - 16; x += 16) {
            auto bColumn     = b + x;
            float32x4_t sum0 = vdupq_n_f32(0.0);
            float32x4_t sum1 = vdupq_n_f32(0.0);
            float32x4_t sum2 = vdupq_n_f32(0.0);
            float32x4_t sum3 = vdupq_n_f32(0.0);
            for (int i = 0; i < k; ++i) {
                const auto bLine = bColumn + i * bw;
                float32x4_t a0   = vdupq_n_f32(aLine[i]);
                float32x4_t b0   = vld1q_f32(bLine);
                float32x4_t b1   = vld1q_f32(bLine + 4);
                float32x4_t b2   = vld1q_f32(bLine + 8);
                float32x4_t b3   = vld1q_f32(bLine + 12);
                sum0             = vmlaq_f32(sum0, a0, b0);
                sum1             = vmlaq_f32(sum1, a0, b1);
                sum2             = vmlaq_f32(sum2, a0, b2);
                sum3             = vmlaq_f32(sum3, a0, b3);
            }
            vst1q_f32(cLine + x, sum0);
            vst1q_f32(cLine + x + 4, sum1);
            vst1q_f32(cLine + x + 8, sum2);
            vst1q_f32(cLine + x + 12, sum3);
        }
        // secondly, compute 4 together
        for (; x <= w - 4; x += 4) {
            auto bColumn    = b + x;
            float32x4_t sum = vdupq_n_f32(0.0);
            for (int i = 0; i < k; ++i) {
                const auto bLine = bColumn + i * bw;
                float32x4_t a4   = vdupq_n_f32(aLine[i]);
                float32x4_t b4   = vld1q_f32(bLine);
                sum              = vmlaq_f32(sum, a4, b4);
            }
            vst1q_f32(cLine + x, sum);
        }
#endif
        for (; x < w; ++x) {
            auto bColumn = b + x;
            float sum    = 0.0f;
            for (int i = 0; i < k; ++i) {
                sum += aLine[i] * bColumn[i * bw];
            }
            cLine[x] = sum;
        }
    }
}
}
}
