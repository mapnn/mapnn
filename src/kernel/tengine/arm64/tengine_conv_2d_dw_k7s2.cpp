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

#include "tengine_kernel.h"
#include <executor/operator/arm64/conv/conv_depthwise/depthwise_conv.hpp>
namespace mapnn {
void tengine_conv_2d_dw_k7s2::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1CHW output(out); 
    L1CHW input(ins[0]); 
    L111W weight(ins[1]); 
    L111W bias(ins[2]);
    depthwise_conv_k7s2(input.data, weight.data, bias.data, output.data,
            input.h, input.w, input.c, output.h, output.w, -1);
}
}
