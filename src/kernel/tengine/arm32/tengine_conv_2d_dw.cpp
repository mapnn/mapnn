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
#include <executor/operator/arm32/conv/conv_depthwise/conv_2d_dw_general.h>
void tengine_conv_2d_dw::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1CHW output(out); 
    L1CHW input(ins[0]); 
    L111W weight(ins[1]); 
    L111W bias(ins[2]);
    int input_c0 = input.c/conv.g;
    int output_c0 = output.c/conv.g;

    TEngine::conv_2d_dw::initial_output(output.data, bias.data, output.c, output.hw);
    TEngine::conv_2d_dw::conv_dw_genreal_kernel(input.data, weight.data, output.data,
            0, conv.g, -1, 
            input_c0, input.h, input.w, 
            output_c0, output.h, output.w,
            conv.hkernel, conv.wkernel,
            0,0,
            conv.hstride, conv.wstride,
            conv.hdilation, conv.wdilation);
}
