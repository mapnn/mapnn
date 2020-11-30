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
#include <layer/arm/convdw5x5s2_pack4_neon.h>
namespace mapnn {
void ncnn_convdw5x5s2_pack4_neon::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    LCHW4 input(ins[0]); 
    LCHW4 output(out); 
    const int extented_filter_h = conv.hdilation * (conv.hkernel - 1) + 1;
    const int extented_filter_w = conv.wdilation * (conv.wkernel - 1) + 1;
    output.c = (conv.outch+3)/4;
    output.h = (input.h - extented_filter_h) / conv.hstride + 1;
    output.w4 = ((input.w4/4 - extented_filter_w) / conv.wstride + 1)*4;
}

void ncnn_convdw5x5s2_pack4_neon::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    LCHW4 output(out); 
    LCHW4 input(ins[0]); 
    LUVAB weight(ins[1]); 
    L111W bias(ins[2]);
    const ncnn::Mat bottom_blob(input.w4/4, input.h, input.c, input.data, 4u*4, 4);
    const ncnn::Mat kernel(weight.a, weight.v, weight.u, weight.data, 4u*16, 16);
    const ncnn::Mat _bias(bias.w, bias.data, 4u, 1);
    ncnn::Mat top_blob(output.w4/4, output.h, output.c, output.data, 4u*4, 4);
    ncnn::Option opt;
    ncnn::convdw5x5s2_pack4_neon(bottom_blob, top_blob, kernel, _bias, opt);
}
}
