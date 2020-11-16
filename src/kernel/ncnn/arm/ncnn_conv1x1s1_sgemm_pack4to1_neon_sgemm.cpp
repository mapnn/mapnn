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


/*#include "ncnn_kernel.h"
#include <layer/arm/conv1x1s1_sgemm_pack4to1_neon_sgemm.h>
#include "conv.h"
void ncnn_conv1x1s1_sgemm_pack4to1_neon_sgemm::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1CHW output(out);
    int outch = ins[0].whisper[0].i;
    //int inch = ins[0].whisper[1].i;
    int outh = ins[0].whisper[2].i;
    int outw = ins[0].whisper[3].i;
    output.c = outch;
    output.h = outh;
    output.w = outw;
}
void ncnn_conv1x1s1_sgemm_pack4to1_neon_sgemm::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1CHW output(out); 
    LUVA4 input(ins[0]); 
    LUVAB weight(ins[1]); 
    L111W bias(ins[2]);

    int inch = conv.inch;
    int outch = conv.outch;
    int outh = ins[0].whisper[2].i;
    int outw = ins[0].whisper[3].i;
    const ncnn::Mat bottom_blob(input.a4/4, input.v, input.u, input.data, 4u*4, 4);
    const ncnn::Mat kernel(weight.a, weight.v, weight.u, weight.data, 4u*4, 4);
    const ncnn::Mat _bias(bias.w, bias.data, 4u, 1);
    ncnn::Mat top_blob(output.w, output.h, output.c, output.data, 4u, 1);
    ncnn::Option opt;
    ncnn::conv1x1s1_sgemm_pack4to1_neon_sgemm(bottom_blob, top_blob, kernel, _bias, opt, inch/4, outw, outh, outch);
}
*/
