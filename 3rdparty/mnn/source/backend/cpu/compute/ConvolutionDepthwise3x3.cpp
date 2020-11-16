//
//  ConvolutionDepthwise3x3.cpp
//  MNN
//
//  Created by MNN on 2019/4/3.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/compute/ConvolutionDepthwise3x3.hpp"

#include <cstring>
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "math/Vec4.hpp"

using namespace MNN::Math;
extern "C" {
void MNNConvDwF23MulTransUnit(float **cacheLine, const float *weigth, float *dest, size_t ow);
void MNNConvDwF23SourceTransUnit(const float *source, float *dest, size_t unit);
}
static void _multiAndDestTransformCommon(float **cacheLine, const float *weigth, float *dest, int cacheLineSize,
                                         int ow) {
    int unit = ow / 2;
    for (int x = 0; x < unit; ++x) {
        auto offset = 4 * 4 * x;
        Vec4 m0     = 0.0f;
        Vec4 m1     = 0.0f;
        Vec4 m2     = 0.0f;
        Vec4 m3     = 0.0f;

        for (int i = 0; i < cacheLineSize; ++i) {
            m0 = m0 + Vec4::load(weigth + i * 16 + 4 * 0) * Vec4::load(cacheLine[i] + offset + 4 * 0);
            m1 = m1 + Vec4::load(weigth + i * 16 + 4 * 1) * Vec4::load(cacheLine[i] + offset + 4 * 1);
            m2 = m2 + Vec4::load(weigth + i * 16 + 4 * 2) * Vec4::load(cacheLine[i] + offset + 4 * 2);
            m3 = m3 + Vec4::load(weigth + i * 16 + 4 * 3) * Vec4::load(cacheLine[i] + offset + 4 * 3);
        }

        auto o0 = m0 + m1 + m2;
        auto o1 = m1 - m2 + m3;
        Vec4::save(dest + 8 * x + 0 * 4, o0);
        Vec4::save(dest + 8 * x + 1 * 4, o1);
    }
    if (unit * 2 < ow) {
        auto offset = 4 * 4 * unit;
        Vec4 m0     = 0.0f;
        Vec4 m1     = 0.0f;
        Vec4 m2     = 0.0f;

        for (int i = 0; i < cacheLineSize; ++i) {
            m0 = m0 + Vec4::load(weigth + i * 16 + 4 * 0) * Vec4::load(cacheLine[i] + offset + 4 * 0);
            m1 = m1 + Vec4::load(weigth + i * 16 + 4 * 1) * Vec4::load(cacheLine[i] + offset + 4 * 1);
            m2 = m2 + Vec4::load(weigth + i * 16 + 4 * 2) * Vec4::load(cacheLine[i] + offset + 4 * 2);
        }

        auto o0 = m0 + m1 + m2;
        Vec4::save(dest + 8 * unit + 0 * 4, o0);
    }
}

static void _sourceTransformCommon(const float *source, float *dest, int unit, int iw, int pad, int su, int eu) {
    for (int x = 0; x < su; ++x) {
        auto dstX = dest + 4 * 4 * x;
        auto sx   = x * 2 - (int)pad;
        auto ex   = sx + 4;

        auto clampSx = std::max(sx, 0);
        auto clampEx = std::min(ex, (int)iw);

        Vec4 v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = clampSx; i < clampEx; ++i) {
            v[i - sx] = Vec4::load(source + 4 * i);
        }
        auto m0 = v[0] - v[2];
        auto m1 = v[1] + v[2];
        auto m2 = v[2] - v[1];
        auto m3 = v[3] - v[1];

        Vec4::save(dstX + 4 * 0, m0);
        Vec4::save(dstX + 4 * 1, m1);
        Vec4::save(dstX + 4 * 2, m2);
        Vec4::save(dstX + 4 * 3, m3);
    }
    MNNConvDwF23SourceTransUnit(source + 4 * (su * 2 - pad), dest + 4 * 4 * su, eu - su);

    for (int x = eu; x < unit; ++x) {
        auto dstX = dest + 4 * 4 * x;
        auto sx   = x * 2 - (int)pad;
        auto ex   = sx + 4;

        auto clampSx = std::max(sx, 0);
        auto clampEx = std::min(ex, (int)iw);

        Vec4 v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = clampSx; i < clampEx; ++i) {
            v[i - sx] = Vec4::load(source + 4 * i);
        }
        auto m0 = v[0] - v[2];
        auto m1 = v[1] + v[2];
        auto m2 = v[2] - v[1];
        auto m3 = v[3] - v[1];

        Vec4::save(dstX + 4 * 0, m0);
        Vec4::save(dstX + 4 * 1, m1);
        Vec4::save(dstX + 4 * 2, m2);
        Vec4::save(dstX + 4 * 3, m3);
    }
}

#ifndef MNN_USE_NEON
void MNNConvDwF23MulTransUnit(float **cacheLine, const float *weigth, float *dest, size_t ow) {
    _multiAndDestTransformCommon(cacheLine, weigth, dest, 3, ow);
}
void MNNConvDwF23SourceTransUnit(const float *source, float *dest, size_t unit) {
    for (int x = 0; x < unit; ++x) {
        auto dstX = dest + 4 * 4 * x;
        auto sx   = x * 2;
        Vec4 v[4];
        for (int i = 0; i < 4; ++i) {
            v[i] = Vec4::load(source + 4 * sx + 4 * i);
        }
        auto m0 = v[0] - v[2];
        auto m1 = v[1] + v[2];
        auto m2 = v[2] - v[1];
        auto m3 = v[3] - v[1];

        Vec4::save(dstX + 4 * 0, m0);
        Vec4::save(dstX + 4 * 1, m1);
        Vec4::save(dstX + 4 * 2, m2);
        Vec4::save(dstX + 4 * 3, m3);
    }
}
#endif

namespace MNN {
void ConvolutionDepthwise3x3(const float* input_data, int inc4, int inh, int inw,
                             const float* weight_data, int wc, int wh, int ww,
                             float* temp_data, int tn, int tv, int ta, int tb,
                             float* output_data, int outc4, int outh, int outw) {
    const int in_chw4 = inc4*inh*inw*4;
    const int out_chw4 = outc4*outh*outw*4;
    const int weight_stride = wc*wh*ww;
    
    int channelC4 = inc4;
    int initSize  = std::min(inh, 2);
    int batch     = 1;
    int ow        = outw;
    int oh        = outh;
    int owUnit    = UP_DIV(ow, 2);
    const int t_stride = 3*owUnit*4*4;

    auto iw           = inw;
    auto ih           = inh;
    auto kernelOrigin = weight_data;
    auto mPadX = 0;
    auto mPadY = 0;
    auto mSourceStartX = UP_DIV(mPadX, 2);
    auto mSourceEndX   = std::max((iw + mPadX - 4) / 2, mSourceStartX);

    /*oy-mPadY>=0*/
    int middelYStart = mPadY;

    /*oy-mPadY+3-1 < ih*/
    int middelYEnd = std::max(ih - 2 + mPadY, middelYStart);

    int threadNumber = 1;
    auto maxKernelH  = std::min(mPadY + ih, 3);

    for (int batchIndex = 0; batchIndex < batch; ++batchIndex) {
        auto inputOrigin  = input_data + batchIndex * in_chw4;
        auto outputOrigin = output_data + batchIndex * out_chw4;
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            auto cacheLineStart = temp_data + tId * t_stride;
            for (int z = (int)tId; z < channelC4; z += threadNumber) {
                auto inputZ     = inputOrigin + 4 * z * iw * ih;
                auto outputZ    = outputOrigin + 4 * z * ow * oh;
                auto kernelZ    = kernelOrigin + z * weight_stride;
                auto cacheLine0 = cacheLineStart + 16 * owUnit * 0;
                auto cacheLine1 = cacheLineStart + 16 * owUnit * 1;
                auto cacheLine2 = cacheLineStart + 16 * owUnit * 2;

                float *cacheLine[3] = {cacheLine0, cacheLine1, cacheLine2};

                // Init
                for (int i = 0; i < initSize; ++i) {
                    _sourceTransformCommon(inputZ + i * iw * 4, cacheLine[i], owUnit, iw, mPadX, mSourceStartX,
                                           mSourceEndX);
                }

                // Compute Top
                for (int y = 0; y < middelYStart; ++y) {
                    auto outputY      = outputZ + y * 4 * ow;
                    int cacheLineSize = y - mPadY + maxKernelH;
                    if (cacheLineSize <= 0) {
                        ::memset(outputY, 0, 4 * ow * sizeof(float));
                        continue;
                    }
                    auto kernelPtr = kernelZ + (maxKernelH - cacheLineSize) * 16;
                    _multiAndDestTransformCommon(cacheLine, kernelPtr, outputY, cacheLineSize, ow);
                }

                // Compute Mid
                for (int y = middelYStart; y < middelYEnd; ++y) {
                    auto outputY = outputZ + y * 4 * ow;
                    auto iy      = y - mPadY + 2;
                    _sourceTransformCommon(inputZ + 4 * iy * iw, cacheLine[2], owUnit, iw, mPadX, mSourceStartX,
                                           mSourceEndX);
                    // FUNC_PRINT(ow);
                    MNNConvDwF23MulTransUnit(cacheLine, kernelZ, outputY, ow);

                    auto temp    = cacheLine[0];
                    cacheLine[0] = cacheLine[1];
                    cacheLine[1] = cacheLine[2];
                    cacheLine[2] = temp;
                }

                // Compute Bottom
                for (int y = middelYEnd; y < oh; ++y) {
                    auto outputY      = outputZ + y * 4 * ow;
                    int cacheLineSize = (ih - y + mPadY);
                    if (cacheLineSize <= 0) {
                        ::memset(outputY, 0, 4 * ow * sizeof(float));
                        continue;
                    }
                    _multiAndDestTransformCommon(cacheLine, kernelZ, outputY, cacheLineSize, ow);
                    cacheLine[0] = cacheLine[1];
                    cacheLine[1] = cacheLine[2];
                }
                //mPostFunction(outputZ, mBias->host<float>() + 4 * z, ow * oh, 1);
            }
        }
        MNN_CONCURRENCY_END();
    }
}
} // namespace MNN
