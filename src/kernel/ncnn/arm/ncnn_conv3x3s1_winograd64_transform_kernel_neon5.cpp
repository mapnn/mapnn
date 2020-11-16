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
#include <layer/arm/conv3x3s1_winograd64_transform_kernel_neon5_GgG.h>
#include <layer/arm/conv3x3s1_winograd64_transform_kernel_neon5_pack.h>
void ncnn_conv3x3s1_winograd64_transform_kernel_neon5::init(const Tensors& /*ins*/, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1VAB output(out);
    int inch = conv.inch;
    int outch = conv.outch;
#if __ARM_NEON && __aarch64__
    output.u = outch/8 + (outch%8)/4 + outch%4;
    output.v = 64;
    output.a = 8*4*(inch/4) + 8*(inch%4);
#else
    output.u = outch/4 + outch%4;
    output.v = 64;
    output.a = 4*4*(inch/4) + 4*(inch%4);
#endif
}
void ncnn_conv3x3s1_winograd64_transform_kernel_neon5::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    int inch = conv.inch;
    int outch = conv.outch;
    Tensor temp_tensor(outch, inch, 64, FLOAT);
    L111W input(ins[0]); 
    L1VAB output(out); 
    LUVAB temp(temp_tensor);
    const ncnn::Mat bottom_blob(input.w, input.data, 4u, 1);
    ncnn::Mat temp_blob(temp.b, temp.a, temp.v, temp.data, 4u, 1);
    ncnn::Mat top_blob(output.a, output.v, output.u, output.data, 4u, 1);
    //ncnn::Option opt;
    ncnn::conv3x3s1_winograd64_transform_kernel_neon5_GgG(bottom_blob, temp_blob, inch, outch);
    ncnn::conv3x3s1_winograd64_transform_kernel_neon5_pack(temp_blob, top_blob, inch, outch);
}
