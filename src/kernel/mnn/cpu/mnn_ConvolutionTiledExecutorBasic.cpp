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
void mnn_ConvolutionTiledExecutorBasic::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    LCHW4 output(out);
    LCHW4 input(ins[0]); 
    const int extented_filter_h = conv.hdilation * (conv.hkernel - 1) + 1;
    const int extented_filter_w = conv.wdilation * (conv.wkernel - 1) + 1;
    const int outw = (input.w4/4 - extented_filter_w) / conv.wstride + 1;
    const int outh = (input.h - extented_filter_h) / conv.hstride + 1;
    output.c = (conv.outch+3)/4;
    output.h = outh;
    output.w4 = outw*4;
}
void mnn_ConvolutionTiledExecutorBasic::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    LCHW4 output(out);
    LCHW4 input(ins[0]); 
    LUVAB weight(ins[1]); 
    L111W bias(ins[2]);

    int kernel_height  = conv.hkernel;
    int kernel_width   = conv.wkernel;
    int padX           = 0;//mPadX;
    int padY           = 0;//mPadY;
    int strideX        = conv.wstride;
    int strideY        = conv.hstride;
    int dilateX        = conv.wdilation;
    int dilateY        = conv.hdilation;
    int src_depth_quad = input.c;
    int width          = output.w4/4;
    int height         = output.h;
    int src_width      = input.w4/4;
    int src_height     = input.h;
    int l = 0, t = 0, r = width, b = height;
    for (; l * strideX - padX < 0 && l < width-1; l++);
    for (; t * strideY - padY < 0 && t < height-1; t++);
    for (; (r - 1) * strideX - padX + kernel_width * dilateX > src_width && r > l; r--);
    for (; (b - 1) * strideY - padY + kernel_height * dilateY > src_height && b > t; b--);
    int dilateY_step = src_width * 4 * dilateY;
    int dilateX_step = dilateX * 4;
    int dst_depth_quad = output.c;
    int threadNumber    = 1;
    float* biasPtr      = bias.data;
    float* weightPtr    = weight.data;
    auto weight_z_step  = kernel_height * kernel_width * src_depth_quad * 16;
    auto weight_sy_step = kernel_width * 16;
    auto weight_sz_step = kernel_width * kernel_height * 16;
    int strideX_step    = strideX * 4;
    int src_z_step      = src_width * src_height * 4;
    //if (width * height <= CONVOLUTION_TILED_NUMBER * 4 || dst_depth_quad < 4 || src_depth_quad < 4) {
        // Use Slice Window
        threadNumber                      = std::min(dst_depth_quad, threadNumber);
        std::function<void(int)> function = [=](int tId) {
            for (int batchIndex = 0; batchIndex < 1; ++batchIndex) {
                auto dstOrigin = output.data + batchIndex * output.chw4;
                auto srcOrigin = input.data + batchIndex * input.chw4;
                for (int dz = tId; dz < dst_depth_quad; dz += threadNumber) {
                    float* dst_z     = dstOrigin + dz * width * height * 4;
                    float* bias_z    = biasPtr + 4 * dz;
                    float* weight_dz = weightPtr + dz * weight_z_step;
                    int dx, dy;
                    // Compute Border
                    CONVOLUVTION_RUN_BASIC(0, 0, width, t, float, nullptr);
                    CONVOLUVTION_RUN_BASIC(0, b, width, height, float, nullptr);
                    CONVOLUVTION_RUN_BASIC(0, t, l, b, float, nullptr);
                    CONVOLUVTION_RUN_BASIC(r, t, width, b, float, nullptr);

                    if (r > l && b > t) {
                        // Compute Mid
                        for (dy = t; dy < b; ++dy) {
                            int srcStartY = dy * strideY - padY;
                            float* dst_y  = dst_z + width * 4 * dy;
                            float* src_dy = srcOrigin + srcStartY * src_width * 4;
                            MNNConvSlideWindowMiddle(dst_y + l * 4, src_dy + (l * strideX - padX) * 4, weight_dz, r - l,
                                                     strideX_step, src_depth_quad, src_z_step, kernel_width,
                                                     kernel_height, dilateX_step, dilateY_step, nullptr);
                        }
                    }
                    MNNAddBias(dst_z, bias_z, width * height, 1);
                }
            }
        };
        MNN_CONCURRENCY_BEGIN(tId, 1) {
            function((int)tId);
        }
        MNN_CONCURRENCY_END();
    //}
}
}
