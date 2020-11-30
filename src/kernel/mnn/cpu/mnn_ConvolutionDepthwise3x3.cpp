/* Copyright 2020 The Mapnn Team. All Rights Reserved. 
 *                                                                            
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *                                                                            
 *     http://www.apache.org/licenses/LICENSE-2.0
 *                                                                            
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mnn_kernel.h"

#include <backend/cpu/compute/CommonOptFunction.h>
#include <backend/cpu/compute/ConvOpt.h>
#include <core/Macro.h>
#include <core/Concurrency.h>
#include <math/Vec4.hpp>

#include "conv.h"

using namespace MNN;
using namespace MNN::Math;

namespace mapnn {
typedef Vec4 float4;
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

#ifdef MNN_USE_NEON
extern "C" {
void MNNConvDwF23MulTransUnit(float **cacheLine, const float *weigth, float *dest, size_t ow);
void MNNConvDwF23SourceTransUnit(const float *source, float *dest, size_t unit);
}
#else
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

void mnn_ConvolutionDepthwise3x3::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    LCHW4 output(out);
    LCHW4 input(ins[0]); 
    LUVAB temp(tmp[0]); 
    const int extented_filter_h = conv.hdilation * (conv.hkernel - 1) + 1;
    const int extented_filter_w = conv.wdilation * (conv.wkernel - 1) + 1;
    const int outw = (input.w4/4 - extented_filter_w) / conv.wstride + 1;
    const int outh = (input.h - extented_filter_h) / conv.hstride + 1;
    const int owUnit = UP_DIV(output.w4/4, 2);
    output.c = (conv.outch+3)/4;
    output.h = outh;
    output.w4 = outw*4;
    temp.u = 1;
    temp.v = 3;
    temp.a = owUnit * 4;
    temp.b = 4;

}
void mnn_ConvolutionDepthwise3x3::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    LCHW4 output(out);
    LCHW4 input(ins[0]); 
    LUVAB weight(ins[1]); 
    LUVAB temp(tmp[0]); 
    L111W bias(ins[2]);
    
    int channelC4 = input.c;
    int initSize  = std::min(input.h, 2);
    int batch     = 1;
    int ow        = output.w4/4;
    int oh        = output.h;
    int owUnit    = UP_DIV(ow, 2);

    auto iw           = input.w4/4;
    auto ih           = input.h;
    auto kernelOrigin = weight.data;
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
        auto inputOrigin  = input.data + batchIndex * input.chw4;
        auto outputOrigin = output.data + batchIndex * output.chw4;
        std::function<void(int)> function = [=](int tId) {
            auto cacheLineStart = temp.data + tId * temp.vab;
            for (int z = (int)tId; z < channelC4; z += threadNumber) {
                auto inputZ     = inputOrigin + 4 * z * iw * ih;
                auto outputZ    = outputOrigin + 4 * z * ow * oh;
                auto kernelZ    = kernelOrigin + z * weight.vab;
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
                MNNAddBias(outputZ, bias.data + 4 * z, ow * oh, 1);
                /*
                */
            }
        };
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            function((int)tId);
        }
        MNN_CONCURRENCY_END();
    }
}
}
