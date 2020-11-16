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
#include <layer/arm/conv3x3s1_winograd64_transform_kernel_neon_GgG.h>
#include <layer/arm/conv3x3s1_winograd64_transform_kernel_neon_pack.h>
void ncnn_conv3x3s1_winograd64_transform_kernel_neon::init(const Tensors& /*ins*/, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1VAB output(out);
    //const int kernel_size = conv.wkernel*conv.hkernel;
    int inch = conv.inch;
    int outch = conv.outch;
    output.u = outch/4 + (outch % 4 + 3) / 4;
    output.v = 1;
    output.a = 8*8*inch*4;
}
void ncnn_conv3x3s1_winograd64_transform_kernel_neon::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
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
    ncnn::conv3x3s1_winograd64_transform_kernel_neon_GgG(bottom_blob, temp_blob, inch, outch);
    ncnn::conv3x3s1_winograd64_transform_kernel_neon_pack(temp_blob, top_blob, inch, outch);
}
