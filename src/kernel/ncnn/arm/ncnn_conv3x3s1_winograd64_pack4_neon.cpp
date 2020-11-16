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
#include <layer/arm/conv3x3s1_winograd64_pack4_neon_BdB.h>
#include <layer/arm/conv3x3s1_winograd64_pack4_neon_permute.h>
#include <layer/arm/conv3x3s1_winograd64_pack4_neon_dot.h>
#include <layer/arm/conv3x3s1_winograd64_pack4_neon_AoA.h>
void ncnn_conv3x3s1_winograd64_pack4_neon::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    LCHW4 input(ins[0]);
    LCHW4 output(out);
    LUVA4 temp0(tmp[0]);
    LUVA4 temp1(tmp[1]);
    LUVA4 temp2(tmp[2]);

    const int extented_filter_h = conv.hdilation * (conv.hkernel - 1) + 1;
    const int extented_filter_w = conv.wdilation * (conv.wkernel - 1) + 1;
    const int outw = (input.w4/4 - extented_filter_w) / conv.wstride + 1;
    const int outh = (input.h - extented_filter_h) / conv.hstride + 1;
    const int tiles = (outw/6)*(outh/6);

    output.c = conv.outch/4;
    output.h = outh;
    output.w4 = outw*4;

    temp0.u = (conv.inch+3)/4;
    temp0.v = 64;
    temp0.a4 = tiles * 4;

#if __aarch64__
    if (tiles >= 12) {
        temp1.u = 64;
        temp1.v = tiles/12 + (tiles%12)/8 + (tiles%12%8)/4 + (tiles%12%4)/2 + tiles%12%2;
        temp1.a4 = 12 * conv.inch;
    }
    else if (tiles >= 8) {
        temp1.u = 64;
        temp1.v = tiles/8 + (tiles%8)/4 + (tiles%4)/2 + tiles%2;
        temp1.a4 = 8 * conv.inch;
    }
    else if (tiles >= 4) {
        temp1.u = 64;
        temp1.v = tiles/4 + (tiles%4)/2 + tiles%2;
        temp1.a4 = 4 * conv.inch;
    }
    else if (tiles >= 2) {
        temp1.u = 64;
        temp1.v = tiles/2 + tiles%2;
        temp1.a4 = 2 * conv.inch;
    }
    else {// if (tiles >= 1)
        temp1.u = 64;
        temp1.v = tiles;
        temp1.a4 = conv.inch;
    }
#else
    if (tiles >= 8) {
        temp1.u = 64;
        temp1.v = tiles/8 + (tiles%8)/4 + (tiles%4)/2 + tiles%2;
        temp1.a4 = 8*conv.inch;
    }
    else if (tiles >= 4) {
        temp1.u = 64;
        temp1.v = tiles/4 + (tiles%4)/2 + tiles%2;
        temp1.a4 = 4 * conv.inch;
    }
    else if (tiles >= 2) {
        temp1.u = 64;
        temp1.v = tiles/2 + tiles%2;
        temp1.a4 = 2* conv.inch;
    }
    else {// if (tiles >= 1)
        temp1.u = 64;
        temp1.v = tiles;
        temp1.a4 = conv.inch;
    }
#endif

    temp2.u = conv.outch/4;
    temp2.v = 64;
    temp2.a4 = tiles*4;
}
void ncnn_conv3x3s1_winograd64_pack4_neon::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    LCHW4 output(out);
    LCHW4 input(ins[0]);
    LUVAB weight(ins[1]); 
    L111W bias(ins[2]);
    LUVA4 temp0(tmp[0]);
    LUVA4 temp1(tmp[1]);
    LUVA4 temp2(tmp[2]);

    {
        const ncnn::Mat bottom_blob(input.w4/4, input.h, input.c, input.data, 4u*4, 4);
        ncnn::Mat top_blob(temp0.a4/4, temp0.v, temp0.u, temp0.data, 4u*4, 4);
        ncnn::Option opt;
        ncnn::conv3x3s1_winograd64_pack4_neon_BdB(bottom_blob, top_blob, opt,
                input.c, output.h, output.w4/4);
    }

    {
        const ncnn::Mat bottom_blob(temp0.a4/4, temp0.v, temp0.u, temp0.data, 4u*4, 4);
        ncnn::Mat top_blob(temp1.a4/4, temp1.v, temp1.u, temp1.data, 4u*4, 4);
        ncnn::Option opt;
        ncnn::conv3x3s1_winograd64_pack4_neon_permute(bottom_blob, top_blob, opt,
                output.c, input.c, output.w4/4, output.h);
    }

    {
        const ncnn::Mat bottom_blob(temp1.a4/4, temp1.v, temp1.u, temp1.data, 4u*4, 4);
        const ncnn::Mat kernel(weight.a, weight.v, weight.u, weight.data, 4u*16, 16);
        ncnn::Mat top_blob(temp2.a4/4, temp2.v, temp2.u, temp2.data, 4u*4, 4);
        ncnn::Option opt;
        ncnn::conv3x3s1_winograd64_pack4_neon_dot(bottom_blob, top_blob, kernel, opt,
                output.c, input.c, output.h, output.w4/4);
    }

    {
        const ncnn::Mat bottom_blob(temp2.a4/4, temp2.v, temp2.u, temp2.data, 4u*4, 4);
        const ncnn::Mat _bias(bias.w, bias.data, 4u, 1);
        ncnn::Mat top_blob(output.w4/4, output.h, output.c, output.data, 4u*4, 4);
        ncnn::Option opt;
        ncnn::conv3x3s1_winograd64_pack4_neon_AoA(bottom_blob, top_blob, _bias, opt,
                output.c, input.c, output.h, output.w4/4);
    }
}
