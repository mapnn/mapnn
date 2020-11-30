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
static void _sourceTransformUnit8x8(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec4 s0 = Vec4::load(srcBlock + 0 * srcStep);
    Vec4 s1 = Vec4::load(srcBlock + 1 * srcStep);
    Vec4 s2 = Vec4::load(srcBlock + 2 * srcStep);
    Vec4 s3 = Vec4::load(srcBlock + 3 * srcStep);
    Vec4 s4 = Vec4::load(srcBlock + 4 * srcStep);
    Vec4 s5 = Vec4::load(srcBlock + 5 * srcStep);
    Vec4 s6 = Vec4::load(srcBlock + 6 * srcStep);
    Vec4 s7 = Vec4::load(srcBlock + 7 * srcStep);

    Vec4 m0 = s0 - s2 * 5.4444446563720703f + s4 * 6.2222223281860352f - s6 * 1.7777777910232544f;

    Vec4 m1 = s1 * 1.5000000f + s2 * 3.0000000f - s3 * 2.1666667461395264f - s4 * 4.3333334922790527f +
              s5 * 0.6666666865348816f + s6 * 1.3333333730697632f;
    Vec4 m2 = s2 * 3.0000000f - s1 * 1.5000000f + s3 * 2.1666667461395264f - s4 * 4.3333334922790527f -
              s5 * 0.6666666865348816f + s6 * 1.3333333730697632f;

    Vec4 m3 = (s3 + s4) * 1.3333333730697632f - (s1 + s2) * 0.3000000f - (s5 + s6) * 0.5333333611488342f;
    Vec4 m4 = (s4 - s3) * 1.3333333730697632f + (s1 - s2) * 0.3000000f + (s5 - s6) * 0.5333333611488342f;

    Vec4 m5 = s1 * 0.0333333350718021f + s2 * 0.0222222227603197f - s3 * 0.1666666716337204f -
              s4 * 0.1111111119389534f + s5 * 0.1333333402872086f + s6 * 0.0888888910412788f;
    Vec4 m6 = s2 * 0.0222222227603197f - s1 * 0.0333333350718021f + s3 * 0.1666666716337204f -
              s4 * 0.1111111119389534f - s5 * 0.1333333402872086f + s6 * 0.0888888910412788f;

    Vec4 m7 = s3 * 3.0625000f - s1 * 0.5625000f - s5 * 3.5f + s7;

    Vec4::save(dstStart + 0 * dstStep, m0);
    Vec4::save(dstStart + 1 * dstStep, m1);
    Vec4::save(dstStart + 2 * dstStep, m2);
    Vec4::save(dstStart + 3 * dstStep, m3);
    Vec4::save(dstStart + 4 * dstStep, m4);
    Vec4::save(dstStart + 5 * dstStep, m5);
    Vec4::save(dstStart + 6 * dstStep, m6);
    Vec4::save(dstStart + 7 * dstStep, m7);
}
static void _destTransformUnit8x6(const float* srcBlock, float* dstStart, size_t srcStep, size_t dstStep) {
    Vec4 s0 = Vec4::load(srcBlock + 0 * srcStep);
    Vec4 s1 = Vec4::load(srcBlock + 1 * srcStep);
    Vec4 s2 = Vec4::load(srcBlock + 2 * srcStep);
    Vec4 s3 = Vec4::load(srcBlock + 3 * srcStep);
    Vec4 s4 = Vec4::load(srcBlock + 4 * srcStep);
    Vec4 s5 = Vec4::load(srcBlock + 5 * srcStep);
    Vec4 s6 = Vec4::load(srcBlock + 6 * srcStep);
    Vec4 s7 = Vec4::load(srcBlock + 7 * srcStep);

    auto m0 = s0 + s1 + s2 + s3 + s4 + s5 + s6;
    auto m1 = (s1 - s2) * 0.5f + s3 - s4 + (s5 - s6) * 1.5f;
    auto m2 = (s1 + s2) * 0.25f + s3 + s4 + (s5 + s6) * 2.25f;
    auto m3 = (s1 - s2) * 0.125f + (s3 - s4) + (s5 - s6) * 3.375f;
    auto m4 = (s1 + s2) * 0.0625f + (s3 + s4) + (s5 + s6) * 5.0625f;
    auto m5 = (s1 - s2) * 0.03125f + (s3 - s4) + (s5 - s6) * 7.59375f + s7;

    Vec4::save(dstStart + 0 * dstStep, m0);
    Vec4::save(dstStart + 1 * dstStep, m1);
    Vec4::save(dstStart + 2 * dstStep, m2);
    Vec4::save(dstStart + 3 * dstStep, m3);
    Vec4::save(dstStart + 4 * dstStep, m4);
    Vec4::save(dstStart + 5 * dstStep, m5);
}

void mnn_ConvolutionWinogradF63::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    LCHW4 output(out);
    LCHW4 input(ins[0]); 
    LUVAB temp0(tmp[0]);
    LUVAB temp1(tmp[1]);
    //const int kernel_size = conv.wkernel*conv.hkernel;
    const int extented_filter_h = conv.hdilation * (conv.hkernel - 1) + 1;
    const int extented_filter_w = conv.wdilation * (conv.wkernel - 1) + 1;
    const int outw = (input.w4/4 - extented_filter_w) / conv.wstride + 1;
    const int outh = (input.h - extented_filter_h) / conv.hstride + 1;
    const int alpha = 8;
    const int alpha2 = alpha * alpha;
    output.c = (conv.outch+3)/4;
    output.h = outh;
    output.w4 = outw*4;
    temp0.u = 1;
    temp0.v = CONVOLUTION_TILED_NUMBER;
    temp0.a = input.c + output.c;
    temp0.b = 4 * alpha2;
    temp1.u = 1;
    temp1.v = 2;
    temp1.a = alpha2;
    temp1.b = 4;
}
void mnn_ConvolutionWinogradF63::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    LCHW4 output(out);
    LCHW4 input(ins[0]); 
    LUVAB weight(ins[1]); 
    LUVAB temp0(tmp[0]);
    LUVAB temp1(tmp[1]);
    L111W biasT(ins[2]);
    
    auto dstUnit = 6;
    auto srcUnit = 8;

    auto srcUnit2 = srcUnit * srcUnit;
    auto dstUnit2 = dstUnit * dstUnit;

    int ow   = output.w4/4;
    int oh   = output.h;
    int iw   = input.w4/4;
    int ih   = input.h;
    int ic_4 = input.c;
    int dc_4 = output.c;

    int padY = 0;//mPadY;
    int padX = 0;//mPadX;

    auto wUnit = UP_DIV(ow, dstUnit);
    auto hUnit = UP_DIV(oh, dstUnit);

    auto totalCount   = wUnit * hUnit;
    int threadNumber = 1;
    int tileCount    = UP_DIV(totalCount, CONVOLUTION_TILED_NUMBER);
    threadNumber     = std::min(threadNumber, tileCount);

    for (int batchIndex = 0; batchIndex < 1; ++batchIndex) {
        auto srcOrigin = input.data + batchIndex * input.chw4;
        auto dstOrigin = output.data + batchIndex * output.chw4;

        auto weight_data    = weight.data;
        auto bias      = biasT.data;
        std::function<void(int)> function = [=](int tId) {
            auto _srcOrigin = temp0.data + tId * temp0.vab;
            auto midBuffer0 = temp1.data + tId * temp1.vab;
            auto midBuffer1 = temp1.data + tId * temp1.vab + temp1.ab;
            for (int tIndex = (int)tId; tIndex < tileCount; tIndex += threadNumber) {
                int xIndex  = (int)tIndex * CONVOLUTION_TILED_NUMBER;
                int xReamin = totalCount - xIndex;
                int xC      = xReamin > CONVOLUTION_TILED_NUMBER ? CONVOLUTION_TILED_NUMBER : xReamin;

                /*Source Transform Begin*/
                {
                    int sourceZStep = iw * ih * 4;
                    int dstZStep    = xC * 4;
                    int unitStep    = ic_4 * xC * 4;
                    for (int xi = 0; xi < xC; ++xi) {
                        auto index = xIndex + xi;

                        int wIndex = index % wUnit;
                        int hIndex = index / wUnit;

                        int srcX  = wIndex * dstUnit - padX;
                        int srcY  = hIndex * dstUnit - padY;
                        int sy    = ALIMAX(0, srcY) - srcY;
                        int ey    = ALIMIN(srcY + srcUnit, ih) - srcY;
                        int sx    = ALIMAX(0, srcX) - srcX;
                        int ex    = ALIMIN(srcX + srcUnit, iw) - srcX;
                        int count = 4 * (ex - sx);

                        auto dst_x = _srcOrigin + 4 * xi;

                        auto srcStart = srcOrigin + (srcX + srcY * iw) * 4;
                        if (ex - sx == srcUnit && ey - sy == srcUnit) {
                            for (int z = 0; z < ic_4; ++z) {
                                auto srcZ = srcStart + z * sourceZStep;
                                // Transform
                                for (int i = 0; i < srcUnit; ++i) {
                                    _sourceTransformUnit8x8(srcZ + 4 * i * iw, midBuffer1 + 4 * i, 4, 4 * srcUnit);
                                }
                                auto dstZ = dst_x + z * dstZStep;
                                for (int i = 0; i < srcUnit; ++i) {
                                    _sourceTransformUnit8x8(midBuffer1 + 4 * i * srcUnit, dstZ + i * unitStep, 4,
                                                     unitStep * srcUnit);
                                }
                            }
                        } else {
                            for (int z = 0; z < ic_4; ++z) {
                                // Extract
                                auto srcZ = srcStart + z * sourceZStep;
                                ::memset(midBuffer0, 0, temp1.ab * sizeof(float));
                                if (count > 0) {
                                    for (int yy = sy; yy < ey; ++yy) {
                                        auto dst_yy = midBuffer0 + yy * srcUnit * 4 + sx * 4;
                                        auto src_yy = srcZ + 4 * iw * yy + sx * 4;
                                        ::memcpy(dst_yy, src_yy, count * sizeof(float));
                                    }
                                }
                                // Transform
                                for (int i = 0; i < srcUnit; ++i) {
                                    _sourceTransformUnit8x8(midBuffer0 + 4 * i * srcUnit, midBuffer1 + 4 * i, 4, 4 * srcUnit);
                                }
                                auto dstZ = dst_x + z * dstZStep;
                                for (int i = 0; i < srcUnit; ++i) {
                                    _sourceTransformUnit8x8(midBuffer1 + 4 * i * srcUnit, dstZ + i * unitStep, 4,
                                                     unitStep * srcUnit);
                                }
                            }
                        }
                    }
                }
                /*Source Transform End*/

                // Multi
                auto _dstOrigin = _srcOrigin + xC * srcUnit2 * ic_4 * 4;

                if (xC == CONVOLUTION_TILED_NUMBER) {
                    for (int i = 0; i < srcUnit2; ++i) {
                        MNNGemmFloatUnit_4(_dstOrigin + i * dc_4 * 4 * xC, _srcOrigin + i * ic_4 * 4 * xC,
                                           weight_data + i * 16 * ic_4 * dc_4, ic_4, xC * 4, dc_4, 0);
                    }
                } else {
                    for (int i = 0; i < srcUnit2; ++i) {
                        MNNGemmFloatCommon_4(_dstOrigin + i * dc_4 * 4 * xC, _srcOrigin + i * ic_4 * 4 * xC,
                                             weight_data + i * 16 * ic_4 * dc_4, ic_4, xC * 4, dc_4, xC, 0);
                    }
                }

                /* Dest Transform And Post Treat Begin */
                {
                    int dstZStep = ow * oh * 4;
                    int srcZStep = xC * 4;
                    int unitStep = dc_4 * xC * 4;
                    for (int xi = 0; xi < xC; ++xi) {
                        auto index = xIndex + xi;
                        auto srcXi = _dstOrigin + 4 * xi;

                        int wIndex = index % wUnit;
                        int hIndex = index / wUnit;

                        int dstX = wIndex * dstUnit;
                        int dstY = hIndex * dstUnit;

                        auto dstStart = dstOrigin + 4 * (dstX + dstY * ow);

                        int ey = ALIMIN(dstY + dstUnit, oh) - dstY;
                        int ex = ALIMIN(dstX + dstUnit, ow) - dstX;

                        int count = ex * 4;
                        if (ex == dstUnit) {
                            for (int z = 0; z < dc_4; ++z) {
                                auto dstZAddr = dstStart + z * dstZStep;
                                auto srcZ     = srcXi + z * srcZStep;
                                auto biasZ    = bias + 4 * z;
                                // Transform
                                for (int i = 0; i < srcUnit; ++i) {
                                    _destTransformUnit8x6(srcZ + i * unitStep, midBuffer0 + i * dstUnit * 4,
                                                   srcUnit * unitStep, 4);
                                }
                                for (int i = 0; i < ey; ++i) {
                                    auto dstAddr = dstZAddr + i * 4 * ow;
                                    _destTransformUnit8x6(midBuffer0 + i * 4, dstAddr, 4 * dstUnit, 4);
                                    MNNAddBias(dstAddr, biasZ, dstUnit, 1);
                                }
                            }
                        } else {
                            for (int z = 0; z < dc_4; ++z) {
                                auto dstZAddr = dstStart + z * dstZStep;
                                auto srcZ     = srcXi + z * srcZStep;
                                // Transform
                                for (int i = 0; i < srcUnit; ++i) {
                                    _destTransformUnit8x6(srcZ + i * unitStep, midBuffer0 + i * dstUnit * 4,
                                                   srcUnit * unitStep, 4);
                                }
                                for (int i = 0; i < ey; ++i) {
                                    _destTransformUnit8x6(midBuffer0 + i * 4, midBuffer1 + i * dstUnit * 4, 4 * dstUnit, 4);
                                }
                                // PostTreat
                                MNNAddBias(midBuffer1, bias + 4 * z, dstUnit2, 1);

                                for (int yy = 0; yy < ey; ++yy) {
                                    auto dstYAddr = dstZAddr + yy * 4 * ow;
                                    auto srcYAddr = midBuffer1 + yy * 4 * dstUnit;
                                    ::memcpy(dstYAddr, srcYAddr, count * sizeof(float));
                                }
                            }
                        }
                    }
                }
                /*Dest Transform And Post Treat End*/
            }
        };

        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            function((int)tId);
        }
        MNN_CONCURRENCY_END();
    }
}
}
