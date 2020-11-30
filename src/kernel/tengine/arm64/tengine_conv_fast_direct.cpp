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
#include <executor/operator/arm64/conv/im2col_gemm_fp32/conv_2d_fast.h>
namespace mapnn {
void tengine_conv_fast_direct::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1CHW output(out); 
    L1CHW input(ins[0]); 
    L1VAB weight(ins[1]); 
    L111W bias(ins[2]);

    int output_chan = conv.outch / conv.g;
    int input_chan = conv.inch / conv.g;
    int kernel_size = input_chan * conv.hkernel * conv.wkernel;
    int kernel_interleaved_size_g = kernel_size * ((output_chan + 3) & -4);
    int kernel_size_g = kernel_size * output_chan;

    for(int i = 0; i < output_chan; i += 16)
    {
        float* kernel = weight.data + input_chan * 9 * i;
        float* outptr = output.data + input.w * input.h * i;
        float* biasptr = nullptr;
        if(bias.data)
            biasptr = bias.data + i;

        TEngine::conv_fast::direct_k3s1p1_4x16(biasptr, input.data, kernel, outptr, input_chan, input.w, input.h, -1, TYPE_A53);
    }
}
}
