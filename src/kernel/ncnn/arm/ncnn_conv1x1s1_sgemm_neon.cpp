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
#include <layer/arm/conv1x1s1_sgemm_neon_interleave.h>
#include <layer/arm/conv1x1s1_sgemm_neon_sgemm.h>

void ncnn_conv1x1s1_sgemm_neon::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1CHW input(ins[0]);
    L1CHW output(out);
    L1VAB temp0(tmp[0]);
    //const int kernel_size = conv.wkernel*conv.hkernel;
    const int extented_filter_h = conv.hdilation * (conv.hkernel - 1) + 1;
    const int extented_filter_w = conv.wdilation * (conv.wkernel - 1) + 1;
    const int outw = (input.w - extented_filter_w) / conv.wstride + 1;
    const int outh = (input.h - extented_filter_h) / conv.hstride + 1;
    const int size = input.hw;
    const int inch = conv.inch;
    const int outch = conv.outch;

    output.c = outch;
    output.h = outh;
    output.w = outw;

    temp0.u = size/8 + (size%8)/4 + size%4;
    temp0.v = inch/4 + inch%4;
    temp0.a = 4*8;
}
void ncnn_conv1x1s1_sgemm_neon::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1CHW output(out); 
    L1CHW input(ins[0]); 
    L1VAB weight(ins[1]); 
    L111W bias(ins[2]);
    L1VAB temp0(tmp[0]);

    {
        const ncnn::Mat bottom_blob(input.w, input.h, input.c, input.data, 4u, 1);
        ncnn::Mat top_blob(temp0.a, temp0.v, temp0.u, temp0.data, 4u, 1);
        ncnn::Option opt;
        ncnn::conv1x1s1_sgemm_neon_interleave(bottom_blob, top_blob, opt,
                input.w, input.h, input.c, output.c);
    }
    {
        const ncnn::Mat bottom_blob(temp0.a, temp0.v, temp0.u, temp0.data, 4u, 1);
        const ncnn::Mat kernel(weight.a, weight.v, weight.u, weight.data, 4u, 1);
        const ncnn::Mat _bias(bias.w, bias.data, 4u, 1);
        ncnn::Mat top_blob(output.w, output.h, output.c, output.data, 4u, 1);
        ncnn::Option opt;
        ncnn::conv1x1s1_sgemm_neon_sgemm(bottom_blob, top_blob, kernel, _bias, opt, 
                output.w, output.h, input.c, output.c);
    }
}
