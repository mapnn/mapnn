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

#define CONVOLUTION_TILED_NUMBER 8
#define SOURCE_BLOCK 64
#define WEIGHT_BLOCK 256
#define SOURCE_BLOCK_VEC 16
#define SRC_BLOCK_UNIT 3
#define SRC_BLOCK_UNIT2 9
#define BLOCK_UNIT 4
#define BLOCK_UNIT2 16

using namespace MNN;
using namespace MNN::Math;

namespace mapnn {
typedef Vec4 float4;
void sourceTransform(const float* srcBlock, float* dstStart, size_t step) {
    auto _x = (float*)srcBlock;
    float4 m00;
    float4 m01;
    float4 m02;
    float4 m03;
    float4 m10;
    float4 m11;
    float4 m12;
    float4 m13;
    float4 m20;
    float4 m21;
    float4 m22;
    float4 m23;
    float4 m30;
    float4 m31;
    float4 m32;
    float4 m33;
    auto _y = dstStart;
    m00     = Vec4::load(_x + 4 * 0) - Vec4::load(_x + 4 * 8);
    m01     = Vec4::load(_x + 4 * 1) - Vec4::load(_x + 4 * 9);
    m02     = Vec4::load(_x + 4 * 2) - Vec4::load(_x + 4 * 10);
    m03     = Vec4::load(_x + 4 * 3) - Vec4::load(_x + 4 * 11);
    m10     = Vec4::load(_x + 4 * 4) + Vec4::load(_x + 4 * 8);
    m11     = Vec4::load(_x + 4 * 5) + Vec4::load(_x + 4 * 9);
    m12     = Vec4::load(_x + 4 * 6) + Vec4::load(_x + 4 * 10);
    m13     = Vec4::load(_x + 4 * 7) + Vec4::load(_x + 4 * 11);
    m20     = Vec4::load(_x + 4 * 8) - Vec4::load(_x + 4 * 4);
    m21     = Vec4::load(_x + 4 * 9) - Vec4::load(_x + 4 * 5);
    m22     = Vec4::load(_x + 4 * 10) - Vec4::load(_x + 4 * 6);
    m23     = Vec4::load(_x + 4 * 11) - Vec4::load(_x + 4 * 7);
    m30     = Vec4::load(_x + 4 * 12) - Vec4::load(_x + 4 * 4);
    m31     = Vec4::load(_x + 4 * 13) - Vec4::load(_x + 4 * 5);
    m32     = Vec4::load(_x + 4 * 14) - Vec4::load(_x + 4 * 6);
    m33     = Vec4::load(_x + 4 * 15) - Vec4::load(_x + 4 * 7);

    Vec4::save(_y + step * 0, m00 - m02);
    Vec4::save(_y + step * 1, m01 + m02);
    Vec4::save(_y + step * 2, m02 - m01);
    Vec4::save(_y + step * 3, m03 - m01);
    Vec4::save(_y + step * 4, m10 - m12);
    Vec4::save(_y + step * 5, m11 + m12);
    Vec4::save(_y + step * 6, m12 - m11);
    Vec4::save(_y + step * 7, m13 - m11);
    Vec4::save(_y + step * 8, m20 - m22);
    Vec4::save(_y + step * 9, m21 + m22);
    Vec4::save(_y + step * 10, m22 - m21);
    Vec4::save(_y + step * 11, m23 - m21);
    Vec4::save(_y + step * 12, m30 - m32);
    Vec4::save(_y + step * 13, m31 + m32);
    Vec4::save(_y + step * 14, m32 - m31);
    Vec4::save(_y + step * 15, m33 - m31);
}

void destTransform(const float* srcZ, float* dstBlock, size_t step) {
    auto yy = dstBlock;
    float4 m00;
    float4 m01;
    float4 m02;
    float4 m03;
    float4 m10;
    float4 m11;
    float4 m12;
    float4 m13;
    auto x = srcZ;
    m00    = Vec4::load(x + step * 0) + Vec4::load(x + step * 4) + Vec4::load(x + step * 8);
    m01    = Vec4::load(x + step * 1) + Vec4::load(x + step * 5) + Vec4::load(x + step * 9);
    m02    = Vec4::load(x + step * 2) + Vec4::load(x + step * 6) + Vec4::load(x + step * 10);
    m03    = Vec4::load(x + step * 3) + Vec4::load(x + step * 7) + Vec4::load(x + step * 11);
    m10    = Vec4::load(x + step * 4) - Vec4::load(x + step * 8) + Vec4::load(x + step * 12);
    m11    = Vec4::load(x + step * 5) - Vec4::load(x + step * 9) + Vec4::load(x + step * 13);
    m12    = Vec4::load(x + step * 6) - Vec4::load(x + step * 10) + Vec4::load(x + step * 14);
    m13    = Vec4::load(x + step * 7) - Vec4::load(x + step * 11) + Vec4::load(x + step * 15);
    Vec4::save(yy + 4 * 0, m00 + m01 + m02);
    Vec4::save(yy + 4 * 1, m01 - m02 + m03);
    Vec4::save(yy + 4 * 2, m10 + m11 + m12);
    Vec4::save(yy + 4 * 3, m11 - m12 + m13);
}

void mnn_convolution3x3_gemm::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1CHW output(out);
    L1CHW input(ins[0]); 
    L1VAB temp(tmp[0]);
    const int extented_filter_h = conv.hdilation * (conv.hkernel - 1) + 1;
    const int extented_filter_w = conv.wdilation * (conv.wkernel - 1) + 1;
    const int outw = (input.w - extented_filter_w) / conv.wstride + 1;
    const int outh = (input.h - extented_filter_h) / conv.hstride + 1;
    const int outputCount = conv.outch;
    const int srcCount = conv.inch;
    output.c = conv.outch;
    output.h = outh;
    output.w = outw;
    temp.u = CONVOLUTION_TILED_NUMBER;
    temp.v = UP_DIV(srcCount, 4) + UP_DIV(outputCount, 4) + 1;
    temp.a = SOURCE_BLOCK;
}
void mnn_convolution3x3_gemm::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1CHW output(out);
    L1CHW input(ins[0]); 
    L111W weight(ins[1]); 
    L111W bias(ins[2]);
    L1VAB temp(tmp[0]);

    int ow   = op.ow;
    int oh   = op.oh;
    int iw   = op.iw;
    int ih   = op.ih;
    int ic_4 = UP_DIV(conv.inch, 4);
    int dc_4 = UP_DIV(conv.outch, 4);

    int padY = 0;//mPadY;
    int padX = 0;//mPadX;

    const int wUnit = UP_DIV(ow, 2), hUnit = UP_DIV(oh, 2);
    const int totalCount = hUnit * wUnit;
    const int tileCount = UP_DIV(totalCount, CONVOLUTION_TILED_NUMBER);
    const int threadNumber = 1;

    auto sourceTransformFunc = [=](int xIndex, int xC, const float* srcOrigin, float* dstOrigin, float* dstBlock) {
        // Source Transform
        for (int xi = 0; xi < xC; ++xi) {
            auto index   = xIndex + xi;
            auto dstUnit = dstOrigin + 4 * xi;

            int wIndex = index % wUnit;
            int hIndex = index / wUnit;

            int srcX = wIndex * 2 - padX;
            int srcY = hIndex * 2 - padY;
            int sy   = ALIMAX(0, srcY) - srcY;
            int ey   = ALIMIN(srcY + 4, ih) - srcY;
            int sx   = ALIMAX(0, srcX) - srcX;
            int ex   = ALIMIN(srcX + 4, iw) - srcX;

            auto srcStart = srcOrigin + (srcX + srcY * iw) * 4;

            memset(dstBlock, 0, SOURCE_BLOCK * sizeof(float));
            for (int z = 0; z < ic_4; ++z) {
                auto _dstStart = dstUnit + z * 4 * xC;

                auto src_z = srcStart + z * 4 * iw * ih;
                if (ex > sx) {
                    // Extract One Block
                    for (int yy = sy; yy < ey; ++yy) {
                        auto dst_yy = dstBlock + yy * 16;
                        auto src_yy = src_z + 4 * iw * yy;
                        ::memcpy(dst_yy + 4 * sx, src_yy + sx * 4, 4 * (ex - sx) * sizeof(float));
                    }
                }
                // Transform
                sourceTransform(dstBlock, _dstStart, 4 * xC * ic_4);
            }
        }
    };

    auto destTransformFunc = [=](int xIndex, int xC, const float* srcOrigin, float* dstOrigin, float* dstBlock) {
        // Dest Transform
        for (int xi = 0; xi < xC; ++xi) {
            auto index   = xIndex + xi;
            auto srcUnit = srcOrigin + 4 * xi;

            int wIndex = index % wUnit;
            int hIndex = index / wUnit;

            int dstX = wIndex * 2;
            int dstY = hIndex * 2;

            auto dstStart = dstOrigin + 4 * (dstX + dstY * ow);

            for (int z = 0; z < dc_4; ++z) {
                auto srcZ = srcUnit + z * xC * 4;
                auto dstZ = dstStart + z * ow * oh * 4;
                destTransform(srcZ, dstBlock, dc_4 * 4 * xC);

                Vec4::save(dstZ, Vec4::load(dstBlock));
                if (wIndex * 2 + 1 < ow) {
                    Vec4::save(dstZ + 4, Vec4::load(dstBlock + 4));
                }
                if (hIndex * 2 + 1 < oh) {
                    Vec4::save(dstZ + ow * 4, Vec4::load(dstBlock + 8));
                    if (wIndex * 2 + 1 < ow) {
                        Vec4::save(dstZ + ow * 4 + 4, Vec4::load(dstBlock + 12));
                    }
                }
            }
        }
    };

    auto gemmFunc = [=](int xC, int start, int end, const float* srcOrigin, const float* weight, float* dstOrigin) {
        // Multi
        if (xC == CONVOLUTION_TILED_NUMBER) {
            for (int i = start; i < end; ++i) {
                MNNGemmFloatUnit_4(dstOrigin + i * dc_4 * 4 * xC, srcOrigin + i * ic_4 * 4 * xC,
                                   weight + i * 16 * ic_4 * dc_4, ic_4, xC * 4, dc_4, 0);
            }
        } else {
            for (int i = start; i < end; ++i) {
                MNNGemmFloatCommon_4(dstOrigin + (i * dc_4) * xC * 4, srcOrigin + i * ic_4 * 4 * xC,
                                     weight + (i * dc_4) * ic_4 * 16, ic_4, xC * 4, dc_4, xC, 0);
            }
        }
    };

    auto gemmConcurrencyFunc = [=, &gemmFunc](int xC, const float* srcOrigin, const float* weight, float* dstOrigin) {
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            const int step = UP_DIV(BLOCK_UNIT2, threadNumber);
            gemmFunc(xC, tId * step, ALIMIN((tId + 1) * step, BLOCK_UNIT2), srcOrigin, weight, dstOrigin);
        }
        MNN_CONCURRENCY_END()
    };

    auto tFunction = [&](const int tId, const int tileStart, const int tileStep, const int tileEnd, const float* srcOrigin, float* dstOrigin) {
        auto _srcOrigin = temp.data + tId * temp.a;
        for (int tIndex = tileStart; tIndex < tileEnd; tIndex += tileStep) {
            int xIndex      = (int)tIndex * CONVOLUTION_TILED_NUMBER;
            int xReamin     = totalCount - xIndex;
            int xC          = xReamin > CONVOLUTION_TILED_NUMBER ? CONVOLUTION_TILED_NUMBER : xReamin;
            auto _dstOrigin = _srcOrigin + xC * SOURCE_BLOCK * ic_4;
            auto dstBlock   = _srcOrigin + xC * SOURCE_BLOCK * (ic_4 + dc_4);

            sourceTransformFunc(xIndex, xC, srcOrigin, _srcOrigin, dstBlock);

            if (threadNumber != tileStep) {
                gemmConcurrencyFunc(xC, _srcOrigin, weight.data, _dstOrigin);
            } else {
                gemmFunc(xC, 0, BLOCK_UNIT2, _srcOrigin, weight.data, _dstOrigin);
            }

            destTransformFunc(xIndex, xC, _dstOrigin, dstOrigin, dstBlock);
        }
    };

    for (int batchIndex = 0; batchIndex < 1; ++batchIndex) {
        auto srcOrigin = input.data + iw * ih * ic_4 * 4 * batchIndex;
        auto dstOrigin = output.data + ow * oh * dc_4 * 4 * batchIndex;

        if (tileCount >= threadNumber) {
            MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
                tFunction((int)tId, (int)tId, threadNumber, tileCount / threadNumber * threadNumber, srcOrigin, dstOrigin);
            }
            MNN_CONCURRENCY_END();
        }

        if (tileCount % threadNumber != 0) {
            tFunction(0, tileCount / threadNumber * threadNumber, 1, tileCount, srcOrigin, dstOrigin);
        }
    }
}
}
