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
#include <layer/arm/conv1x1s2_pack4to1_neon.h>
void ncnn_conv1x1s2_pack4to1_neon::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    LCHW4 input(ins[0]); 
    LCHW4 output(out); 
    output.c = input.c;
    output.h = input.h;
    output.w4 = input.w4;
}

void ncnn_conv1x1s2_pack4to1_neon::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    LCHW4 output(out); 
    LCHW4 input(ins[0]); 
    const ncnn::Mat bottom_blob(input.w4/4, input.h, input.c, input.data, 4u*4, 4);
    ncnn::Mat top_blob(output.w4/4, output.h, output.c, output.data, 4u*4, 4);
    ncnn::Option opt;
    ncnn::conv1x1s2_pack4to1_neon(bottom_blob, top_blob, opt);
}
