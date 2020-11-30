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
#include <layer/arm/conv3x3s1_winograd64_neon5_BdB.h>
#include <layer/arm/conv3x3s1_winograd64_neon5_permute.h>
#include <layer/arm/conv3x3s1_winograd64_neon5_dot.h>
#include <layer/arm/conv3x3s1_winograd64_neon5_AoA.h>
namespace mapnn {
void ncnn_conv3x3s1_winograd64_neon5::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1CHW input(ins[0]);
    L1CHW output(out);
    L1VAB temp0(tmp[0]);
    L1VAB temp1(tmp[1]);
    L1VAB temp2(tmp[2]);

    const int extented_filter_h = conv.hdilation * (conv.hkernel - 1) + 1;
    const int extented_filter_w = conv.wdilation * (conv.wkernel - 1) + 1;
    const int outw = (input.w - extented_filter_w) / conv.wstride + 1;
    const int outh = (input.h - extented_filter_h) / conv.hstride + 1;
    const int tiles = (outh/6)*(outw/6);

    output.c = conv.outch;
    output.h = outh;
    output.w = outw;

    temp0.u = conv.inch;
    temp0.v = 64*tiles;
    temp0.a = 1;

    temp1.u = 64;
    temp1.v = tiles/8 + (tiles%8)/4 + tiles%4;
    temp1.a = 8*conv.inch;

    temp2.u = conv.outch;
    temp2.v = 64*tiles;
    temp2.a = 1;

}
void ncnn_conv3x3s1_winograd64_neon5::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    L1CHW output(out);
    L1CHW input(ins[0]);
    L1VAB weight(ins[1]); 
    L111W bias(ins[2]);
    L1VAB temp0(tmp[0]);
    L1VAB temp1(tmp[1]);
    L1VAB temp2(tmp[2]);

    {
        const ncnn::Mat bottom_blob(input.w, input.h, input.c, input.data, 4u, 1);
        ncnn::Mat top_blob(temp0.a, temp0.v, temp0.u, temp0.data, 4u, 1);
        ncnn::Option opt;
        ncnn::conv3x3s1_winograd64_neon5_BdB(bottom_blob, top_blob, opt,
                input.c, output.w, output.h, output.c);
    }

    {
        const ncnn::Mat bottom_blob(temp0.a, temp0.v, temp0.u, temp0.data, 4u, 1);
        ncnn::Mat top_blob(temp1.a, temp1.v, temp1.u, temp1.data, 4u, 1);
        ncnn::Option opt;
        ncnn::conv3x3s1_winograd64_neon5_permute(bottom_blob, top_blob, opt,
                input.c, output.w, output.h, output.c);
    }

    {
        const ncnn::Mat bottom_blob(temp1.a, temp1.v, temp1.u, temp1.data, 4u, 1);
        const ncnn::Mat kernel(weight.a, weight.v, weight.u, weight.data, 4u, 1);
        ncnn::Mat top_blob(temp2.a, temp2.v, temp2.u, temp2.data, 4u, 1);
        ncnn::Option opt;
        ncnn::conv3x3s1_winograd64_neon5_dot(bottom_blob, top_blob, kernel, opt,
                input.c, output.w, output.h, output.c);
    }

    {
        const ncnn::Mat bottom_blob(temp2.a, temp2.v, temp2.u, temp2.data, 4u, 1);
        const ncnn::Mat _bias(bias.w, bias.data, 4u, 1);
        ncnn::Mat top_blob(output.w, output.h, output.c, output.data, 4u, 1);
        ncnn::Option opt;
        ncnn::conv3x3s1_winograd64_neon5_AoA(bottom_blob, top_blob, _bias, opt,
                input.c, output.w, output.h, output.c);
    }
}
}
