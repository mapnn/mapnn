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

#include "ncnn_kernel.h"
#include <layer/arm/conv1x1s1_sgemm_pack4_neon_interleave.h>
#include <layer/arm/conv1x1s1_sgemm_pack4_neon_sgemm.h>
namespace mapnn {
void ncnn_conv1x1s1_sgemm_pack4_neon::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    LCHW4 input(ins[0]);
    LCHW4 output(out);
    LUVA4 temp0(tmp[0]);
    const int extented_filter_h = conv.hdilation * (conv.hkernel - 1) + 1;
    const int extented_filter_w = conv.wdilation * (conv.wkernel - 1) + 1;
    const int outw = (input.w4/4 - extented_filter_w) / conv.wstride + 1;
    const int outh = (input.h - extented_filter_h) / conv.hstride + 1;
    const int size = input.hw4/4;

    output.c = (conv.outch+3)/4;
    output.h = outh;
    output.w4 = outw*4;

#if __aarch64__
    temp0.u = size/12 + (size%12)/8 + (size%12%8)/4 + (size%12%4)/2 + size%12%2;
    temp0.v = (conv.inch+3)/4;
    temp0.a4 = 12*4;
#else
    temp0.u = size/8 + (size%8)/4 + (size%4)/2 + size%2;
    temp0.v = (conv.inch+3)/4;
    temp0.a4 = 8*4;
#endif
}
void ncnn_conv1x1s1_sgemm_pack4_neon::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    LCHW4 input(ins[0]); 
    LCHW4 output(out); 
    LUVAB weight(ins[1]); 
    L111W bias(ins[2]);
    LUVA4 temp0(tmp[0]);

    {
        const ncnn::Mat bottom_blob(input.w4/4, input.h, input.c, input.data, 4u*4, 4);
        ncnn::Mat top_blob(temp0.a4/4, temp0.v, temp0.u, temp0.data, 4u*4, 4);
        ncnn::Option opt;
        ncnn::conv1x1s1_sgemm_pack4_neon_interleave(bottom_blob, top_blob, opt, output.c, input.c);
    }
    {
        const ncnn::Mat bottom_blob(temp0.a4/4, temp0.v, temp0.u, temp0.data, 4u*4, 4);
        const ncnn::Mat kernel(weight.a, weight.v, weight.u, weight.data, 4u*16, 16);
        const ncnn::Mat _bias(bias.w, bias.data, 4u, 1);
        ncnn::Mat top_blob(output.w4/4, output.h, output.c, output.data, 4u*4, 4);
        ncnn::Option opt;
        ncnn::conv1x1s1_sgemm_pack4_neon_sgemm(bottom_blob, top_blob, kernel, _bias, opt, input.c, output.w4/4, output.h, output.c);
    }
}
}
