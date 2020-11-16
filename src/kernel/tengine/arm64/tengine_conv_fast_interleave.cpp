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
void tengine_conv_fast_interleave::init(const Tensors& /*ins*/, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1VAB output(out);
    int output_chan = conv.outch / conv.g;
    int input_chan = conv.inch / conv.g;
    int kernel_size = input_chan * conv.hkernel * conv.wkernel;
    int kernel_interleaved_size_g = kernel_size * ((output_chan + 3) & -4);
    output.u = 1;
    output.v = 1;
    output.a = kernel_interleaved_size_g * conv.g+ 32;
}
void tengine_conv_fast_interleave::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1VAB output(out); 
    L111W input(ins[0]); 

    int output_chan = conv.outch / conv.g;
    int input_chan = conv.inch / conv.g;
    int kernel_size = input_chan * conv.hkernel * conv.wkernel;
    int kernel_interleaved_size_g = kernel_size * ((output_chan + 3) & -4);
    int kernel_size_g = kernel_size * output_chan;

    for(int g = 0; g < conv.g; ++g) {
        float* kernel = input.data + g * kernel_size_g;
        float* kernel_interleaved_g = output.data + g * kernel_interleaved_size_g;
        TEngine::conv_fast::interleave_kernel(kernel, kernel_interleaved_g, output_chan, kernel_size);
    }
}
