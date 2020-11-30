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
#include <layer/arm/conv_weight_pack4x1_neon.h>
namespace mapnn {
void ncnn_conv_weight_pack4x1_neon::init(const Tensors& /*ins*/, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    LUVA4 output(out);
    const int maxk = conv.wkernel*conv.hkernel;
    int inch = conv.inch;
    int outch = conv.outch;
    output.u = (outch+3)/4;
    output.v = inch;
    output.a4 = maxk*4;
}
void ncnn_conv_weight_pack4x1_neon::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L111W input(ins[0]); 
    LUVA4 output(out); 
    const int maxk = conv.wkernel*conv.hkernel;
    const ncnn::Mat bottom_blob(maxk, conv.inch, conv.outch, input.data, 4u, 1);
    ncnn::Mat top_blob(output.a4/4, output.v, output.u, output.data, 4u*4, 4);
    ncnn::Option opt;
    ncnn::conv_weight_pack4x1_neon(bottom_blob, top_blob, opt, conv.outch, conv.inch, maxk);
}
}
