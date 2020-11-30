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
#include <layer/arm/conv1x1s1_sgemm_transform_kernel_pack4_neon.h>
namespace mapnn {
void ncnn_conv1x1s1_sgemm_transform_kernel_pack4_neon::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    LUVAB output(out);
    int outch = conv.outch;
    int inch = conv.inch;
#if __aarch64__
    output.u = (outch/4)/2 + (outch/4)%2;
    output.v = inch/4;
    output.a = 2;
    output.b = 16;
#else
    output.u = outch/4;
    output.v = inch/4;
    output.a = 1;
    output.b = 16;
#endif

}
void ncnn_conv1x1s1_sgemm_transform_kernel_pack4_neon::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L111W input(ins[0]); 
    LUVAB output(out); 

    int inch = conv.inch;
    int outch = conv.outch;

    const ncnn::Mat bottom_blob(input.w, input.data, 4u, 1);
    ncnn::Mat top_blob(output.a, output.v, output.u, output.data, 4u*16, 16);
    ncnn::conv1x1s1_sgemm_transform_kernel_pack4_neon(bottom_blob, top_blob, inch, outch);
}
}
