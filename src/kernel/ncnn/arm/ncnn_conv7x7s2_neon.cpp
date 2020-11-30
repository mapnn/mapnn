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
#include "layer/arm/conv7x7s2_neon.h"
namespace mapnn {
void ncnn_conv7x7s2_neon::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    L1CHW output(out); 
    L1CHW input(ins[0]); 
    L111W weight(ins[1]); 
    L111W bias(ins[2]);
    const ncnn::Mat bottom_blob(input.w, input.h, input.c, input.data, 4u, 1);
    const ncnn::Mat kernel(weight.w, weight.data, 4u, 1);
    const ncnn::Mat _bias(bias.w, bias.data, 4u, 1);
    ncnn::Mat top_blob(output.w, output.h, output.c, output.data, 4u, 1);
    ncnn::Option opt;
    ncnn::conv7x7s2_neon(bottom_blob, top_blob, kernel, _bias, opt);
}
}
