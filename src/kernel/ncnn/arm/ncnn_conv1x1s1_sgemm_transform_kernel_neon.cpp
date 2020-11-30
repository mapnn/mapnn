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
#include <layer/arm/conv1x1s1_sgemm_transform_kernel_neon.h>
namespace mapnn {
void ncnn_conv1x1s1_sgemm_transform_kernel_neon::init(const Tensors& /*ins*/, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1VAB output(out);
    //const int kernel_size = conv.wkernel*conv.hkernel;
    int inch = conv.inch;
    int outch = conv.outch;
#if __ARM_NEON && __aarch64__
    output.u = outch/8 + (outch%8)/4 + outch%4;
    output.v = inch/4 + inch%4;
    output.a = 4*8;
#else
    output.u = outch/4 + outch%4;
    output.v = inch/4 + inch%4;
    output.a = 4*4;
#endif // __ARM_NEON && __aarch64__
}
void ncnn_conv1x1s1_sgemm_transform_kernel_neon::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L111W input(ins[0]); 
    L1VAB output(out); 

    //int w = input.w;
    int inch = conv.inch;
    int outch = conv.outch;

    const ncnn::Mat bottom_blob(input.w, input.data, 4u, 1);
    ncnn::Mat top_blob(output.a, output.v, output.u, output.data, 4u, 1);
    ncnn::conv1x1s1_sgemm_transform_kernel_neon(bottom_blob, top_blob, inch, outch);
}
}
