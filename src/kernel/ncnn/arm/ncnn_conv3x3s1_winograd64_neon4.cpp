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
#include <layer/arm/conv3x3s1_winograd64_neon4_BdB.h>
#include <layer/arm/conv3x3s1_winograd64_neon4_dot.h>
#include <layer/arm/conv3x3s1_winograd64_neon4_AoA.h>
namespace mapnn {
void ncnn_conv3x3s1_winograd64_neon4::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1CHW output(out);
    L1CHW input(ins[0]);
    L1VAB temp0(tmp[0]);
    L1VAB temp1(tmp[1]);

    const int extented_filter_h = conv.hdilation * (conv.hkernel - 1) + 1;
    const int extented_filter_w = conv.wdilation * (conv.wkernel - 1) + 1;
    const int outw = (input.w - extented_filter_w) / conv.wstride + 1;
    const int outh = (input.h - extented_filter_h) / conv.hstride + 1;

    output.c = conv.outch;
    output.h = outh;
    output.w = outw;

    temp0.u = conv.inch;
    temp0.v = 16*(outw/6)*(outh/6);
    temp0.a = 4;

    temp1.u = conv.outch;
    temp1.v = 16 * (outw/6)*(outh/6);
    temp1.a = 4;
}
void ncnn_conv3x3s1_winograd64_neon4::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    L1CHW output(out);
    L1CHW input(ins[0]);
    LUVAB weight(ins[1]); 
    L111W bias(ins[2]);
    L1VAB temp0(tmp[0]);
    L1VAB temp1(tmp[1]);

    {
        const ncnn::Mat bottom_blob(input.w, input.h, input.c, input.data, 4u, 1);
        ncnn::Mat top_blob(temp0.a, temp0.v, temp0.u, temp0.data, 4u, 1);
        ncnn::Option opt;
        ncnn::conv3x3s1_winograd64_neon4_BdB(bottom_blob, top_blob, opt,
                output.c, input.c, output.h, output.w);
    }

    {
        const ncnn::Mat bottom_blob(temp0.a, temp0.v, temp0.u, temp0.data, 4u, 1);
        const ncnn::Mat kernel(weight.b, weight.a, weight.v, weight.data, 4u, 1);
        ncnn::Mat top_blob(temp1.a, temp1.v, temp1.u, temp1.data, 4u, 1);
        ncnn::Option opt;
        ncnn::conv3x3s1_winograd64_neon4_dot(bottom_blob, top_blob, kernel, opt,
                output.c, input.c, output.h, output.w);

    }
    {
        const ncnn::Mat bottom_blob(temp1.a, temp1.v, temp1.u, temp1.data, 4u, 1);
        const ncnn::Mat _bias(bias.w, bias.data, 4u, 1);
        ncnn::Mat top_blob(output.w, output.h, output.c, output.data, 4u, 1);
        ncnn::Option opt;
        ncnn::conv3x3s1_winograd64_neon4_AoA(bottom_blob, top_blob, _bias, opt, output.c, input.c, output.h, output.w);
    }
}
}
