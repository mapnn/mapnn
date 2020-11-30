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
#include <layer/arm/conv3x3s1_winograd64_transform_kernel_pack4_neon_GgG.h>
#include <layer/arm/conv3x3s1_winograd64_transform_kernel_pack4_neon_pack.h>
namespace mapnn {
void ncnn_conv3x3s1_winograd64_transform_kernel_pack4_neon::init(const Tensors& /*ins*/, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    LUVAB output(out);
    int inch = conv.inch;
    int outch = conv.outch;
#if __aarch64__
    output.u = (outch/4)/2 + (outch/4)%2;
    output.v = 64;
    output.a = 2 *inch/4;
    output.b = 16;
#else
    output.u = outch/4;
    output.v = 64;
    output.a = inch/4;
    output.b = 16;
#endif
}
void ncnn_conv3x3s1_winograd64_transform_kernel_pack4_neon::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    int inch = conv.inch;
    int outch = conv.outch;
    Tensor temp_tensor(outch, inch, 64, 1, FLOAT);
    L111W input(ins[0]); 
    LUVAB output(out); 
    L1VAB temp(temp_tensor);
    const ncnn::Mat bottom_blob(input.w, input.data, 4u, 1);
    ncnn::Mat temp_blob(temp.a, temp.v, temp.u, temp.data, 4u, 1);
    ncnn::Mat top_blob(output.a, output.v, output.u, output.data, 4u*16, 16);
    //ncnn::Option opt;
    ncnn::conv3x3s1_winograd64_transform_kernel_pack4_neon_GgG(bottom_blob, temp_blob, inch, outch);
    ncnn::conv3x3s1_winograd64_transform_kernel_pack4_neon_pack(temp_blob, top_blob, inch, outch);
}
}
