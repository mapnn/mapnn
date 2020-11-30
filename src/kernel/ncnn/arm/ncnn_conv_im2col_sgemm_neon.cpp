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
#include <layer/arm/conv_im2col_sgemm_neon_im2col.h>
#include <layer/arm/conv_im2col_sgemm_neon_packed.h>
#include <layer/arm/conv_im2col_sgemm_neon_sgemm.h>
namespace mapnn {
void ncnn_conv_im2col_sgemm_neon::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1CHW input(ins[0]);
    L1CHW output(out);
    L1VAB temp0(tmp[0]);
    L1VAB temp1(tmp[1]);

    const int extented_filter_h = conv.hdilation * (conv.hkernel - 1) + 1;
    const int extented_filter_w = conv.wdilation * (conv.wkernel - 1) + 1;
    const int outw = (input.w - extented_filter_w) / conv.wstride + 1;
    const int outh = (input.h - extented_filter_h) / conv.hstride + 1;
    const int out_size = outw * outh;
    const int kernel_size = conv.hkernel* conv.wkernel;

    output.c = conv.outch;
    output.h = outh;
    output.w = outw;

    temp0.u = 1;
    temp0.v = kernel_size*conv.inch;
    temp0.a = outh*outw;

    temp1.u = out_size/8 + out_size%8;
    temp1.v = conv.inch;
    temp1.a = 8*kernel_size;
}
void ncnn_conv_im2col_sgemm_neon::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1CHW output(out); 
    L1CHW input(ins[0]); 
    L1VAB weight(ins[1]); 
    L111W bias(ins[2]);
    L1VAB temp0(tmp[0]);
    L1VAB temp1(tmp[1]);

    {
        const ncnn::Mat bottom_blob(input.w, input.h, input.c, input.data, 4u, 1);
        ncnn::Mat top_blob(temp0.a, temp0.v, temp0.u, temp0.data, 4u, 1);
        ncnn::Option opt;
        ncnn::conv_im2col_sgemm_neon_im2col(bottom_blob, top_blob,
                conv.wkernel, conv.hkernel,
                conv.wstride, conv.hstride,
                opt, input.w, input.c, output.w, output.h, output.c);
    }

    {
        const ncnn::Mat bottom_blob(temp0.a, temp0.v, temp0.u, temp0.data, 4u, 1);
        ncnn::Mat top_blob(temp1.a, temp1.v, temp1.u, temp1.data, 4u, 1);
        ncnn::Option opt;
        ncnn::conv_im2col_sgemm_neon_packed(bottom_blob, top_blob, 
                conv.wkernel, conv.hkernel,
                conv.wstride, conv.hstride,
                opt, input.c, output.w, output.h, output.c);
    }

    {
        const ncnn::Mat bottom_blob(temp1.a, temp1.v, temp1.u, temp1.data, 4u, 1);
        const ncnn::Mat kernel(weight.a, weight.v, weight.u, weight.data, 4u, 1);
        const ncnn::Mat _bias(bias.w, bias.data, 4u, 1);
        ncnn::Mat top_blob(output.w, output.h, output.c, output.data, 4u, 1);
        ncnn::Option opt;
        ncnn::conv_im2col_sgemm_neon_sgemm(bottom_blob, top_blob, kernel, _bias,
                conv.wkernel, conv.hkernel, conv.wstride, conv.hstride, opt,
                input.c, output.w, output.h, output.c);
    }
}
}
