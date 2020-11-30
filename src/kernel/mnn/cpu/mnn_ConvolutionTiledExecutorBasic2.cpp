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
void mnn_ConvolutionTiledExecutorBasic2::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    LCHW4 output(out);
    LCHW4 input(ins[0]); 
    LUVA4 temp(tmp[0]);
    const int kernel_size = conv.wkernel*conv.hkernel;
    const int extented_filter_h = conv.hdilation * (conv.hkernel - 1) + 1;
    const int extented_filter_w = conv.wdilation * (conv.wkernel - 1) + 1;
    const int outw = (input.w4/4 - extented_filter_w) / conv.wstride + 1;
    const int outh = (input.h - extented_filter_h) / conv.hstride + 1;
    output.c = (conv.outch+3)/4;
    output.h = outh;
    output.w4 = outw*4;
    temp.u = 1;
    temp.v = CONVOLUTION_TILED_NUMBER;
    temp.a4 = input.c * kernel_size * 4;
}
void mnn_ConvolutionTiledExecutorBasic2::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    LCHW4 output(out);
    LCHW4 input(ins[0]); 
    LUVAB weight(ins[1]); 
    LUVA4 temp(tmp[0]);
    L111W bias(ins[2]);

    int kernel_height  = conv.hkernel;
    int kernel_width   = conv.wkernel;
    int padX           = 0;//mPadX;
    int padY           = 0;//mPadY;
    int strideX        = conv.wstride;
    int strideY        = conv.hstride;
    int dilateX        = conv.wdilation;
    int dilateY        = conv.hdilation;
    //int src_depth_quad = input.c;
    int width          = output.w4/4;
    int height         = output.h;
    int src_width      = input.w4/4;
    int src_height     = input.h;
    int l = 0, t = 0, r = width, b = height;
    for (; l * strideX - padX < 0 && l < width-1; l++);
    for (; t * strideY - padY < 0 && t < height-1; t++);
    for (; (r - 1) * strideX - padX + kernel_width * dilateX > src_width && r > l; r--);
    for (; (b - 1) * strideY - padY + kernel_height * dilateY > src_height && b > t; b--);
    //int dilateY_step = src_width * 4 * dilateY;
    //int dilateX_step = dilateX * 4;
    int dst_depth_quad = output.c;
    int threadNumber    = 1;
    int threadNumberSecond    = 1;
    float* biasPtr      = bias.data;
    float* weightPtr    = weight.data;
    //auto weight_z_step  = kernel_height * kernel_width * src_depth_quad * 16;
    //auto weight_sy_step = kernel_width * 16;
    //auto weight_sz_step = kernel_width * kernel_height * 16;
    //int strideX_step    = strideX * 4;
    int src_z_step      = src_width * src_height * 4;

    auto icC4 = input.c;
    auto ocC4 = output.c;

    int count = UP_DIV(width*height, CONVOLUTION_TILED_NUMBER);
    int plane = width * height;
    auto threadNumberFirst = 1;
    std::function<void(int)> function = [=](int tId) {
        auto colBuffer = temp.data + temp.va4 * tId;
        for (int batchIndex = 0; batchIndex < 1; ++batchIndex) {
            auto dstOrigin = output.data + batchIndex * output.chw4;
            auto srcOrigin = input.data + batchIndex * input.chw4;

            for (int x = (int)tId; x < count; x += threadNumberFirst) {
                int start    = (int)x * CONVOLUTION_TILED_NUMBER;
                int remain   = plane - start;
                int xC        = remain > CONVOLUTION_TILED_NUMBER ? CONVOLUTION_TILED_NUMBER : remain;
                // Im2Col
                ::memset(colBuffer, 0, temp.va4 * sizeof(float));
                for (int i = 0; i<xC; ++i) {
                    int index = start + i;
                    int ox = index % width;
                    int oy = index / width;
                    int sxSta = ox * strideX - padX;
                    int sySta = oy * strideY - padY;
                    for (int ky=0; ky<kernel_height; ++ky) {
                        auto sy = sySta + ky * dilateY;
                        if (sy < 0 || sy >= src_height) {
                            continue;
                        }
                        for (int kx=0; kx<kernel_width; ++kx) {
                            auto sx = sxSta + kx * dilateX;
                            if (sx < 0 || sx >= src_width) {
                                continue;
                            }
                            auto src = srcOrigin + sx * 4 + sy * 4 * src_width;
                            auto dst = colBuffer + i * 4 + 4 * xC * (kx + ky*kernel_width);
                            for (int sz=0; sz<icC4; ++sz) {
                                Math::Vec4::save(dst + 4 * xC * kernel_height * kernel_width * sz, Math::Vec4::load(src + src_z_step * sz));
                            }
                        }
                    }
                }
                // GEMM
                if (xC == CONVOLUTION_TILED_NUMBER) {
                    MNNGemmFloatUnit_4(dstOrigin + start * 4, colBuffer,
                                        weightPtr, icC4 * kernel_width * kernel_height, width * height * 4, ocC4, 0);
                } else {
                    MNNGemmFloatCommon_4(dstOrigin + start * 4, colBuffer,
                                        weightPtr, icC4 * kernel_width * kernel_height, width * height * 4, ocC4, xC, 0);
                }
            }
        }
    };
    std::function<void(int)> secondFunction = [biasPtr, width, height, dst_depth_quad, output,
                                               threadNumberSecond](int tId) {
        for (int batchIndex = 0; batchIndex < 1; ++batchIndex) {
            auto dstOrigin = output.data + batchIndex * output.chw4;
            for (int dz = tId; dz < dst_depth_quad; dz += threadNumberSecond) {
                float* dst_z  = dstOrigin + dz * width * height * 4;
                float* bias_z = biasPtr + 4 * dz;
                MNNAddBias(dst_z, bias_z, width * height, 1);
            }
        }
    };

    MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
        function((int)tId);
    }
    MNN_CONCURRENCY_END();

    MNN_CONCURRENCY_BEGIN(tId, threadNumberSecond) {
        secondFunction((int)tId);
    }
    MNN_CONCURRENCY_END();
    //}
}
}
