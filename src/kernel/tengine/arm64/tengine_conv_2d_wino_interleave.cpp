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
#include <executor/operator/arm64/conv/winograd/wino_trans_ker.h>
#include <executor/operator/arm64/conv/winograd/wino_trans_inp.h>
#include <executor/operator/arm64/conv/winograd/wino_sgemm.h>
#include <executor/operator/arm64/conv/winograd/conv_2d_wino.h>
namespace mapnn {
void tengine_conv_2d_wino_interleave::init(const Tensors& /*ins*/, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1VAB output(out);
    L1VAB temp(tmp[0]);

    int trans_ker_size = conv.outch * conv.inch * 36;
    output.u = 1;
    output.v = 1;
    output.a = trans_ker_size + 32;
    temp.u = 1;
    temp.v = 1;
    temp.a = trans_ker_size + 32;
}
void tengine_conv_2d_wino_interleave::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1VAB output(out); 
    L111W input(ins[0]); 
    L1VAB temp(tmp[0]);

    int input_c = conv.inch;
    int output_c = conv.outch;
    float* kernel_org = input.data;
    float* kernel_trans = temp.data;
    float* kernel_interleaved = output.data;
    transform_kernel_f43_tile(kernel_org, kernel_trans, input_c, output_c);
    interleave_kernel(kernel_trans, kernel_interleaved, output_c, input_c);
}
}
