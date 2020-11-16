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
void tengine_conv_fast_gemm::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1CHW input(ins[0]); 
    L1CHW output(out); 
    L1VAB temp(tmp[0]);
    const int extented_filter_h = conv.hdilation * (conv.hkernel - 1) + 1;
    const int extented_filter_w = conv.wdilation * (conv.wkernel - 1) + 1;
    const int output_xy = output.hw;
    const int kernel_size = input.c * conv.hkernel * conv.wkernel;
    output.c = conv.outch;
    output.h = (input.h - extented_filter_h) / conv.hstride + 1;
    output.w = (input.w - extented_filter_w) / conv.wstride + 1;
    temp.u = 1;
    temp.v = 1;
    temp.a = ((kernel_size * ((output_xy + 3) & -4)) + 32);
}
void tengine_conv_fast_gemm::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1CHW output(out); 
    L1CHW input(ins[0]); 
    L1VAB weight(ins[1]); 
    L111W bias(ins[2]);
    L1VAB temp(tmp[0]);

    int output_chan = conv.outch / conv.g;
    int input_chan = conv.inch / conv.g;
    int kernel_size = input_chan * conv.hkernel * conv.wkernel;
    int input_size = input.chw;
    int group = conv.g;
    int output_xy = output.hw;
    int output_n = 1;
    int output_y = output.h;
    int output_x = output.w;
    int input_h = input.h;
    int input_w = input.w;
    int kernel_x = conv.wkernel;
    int kernel_y = conv.hkernel;
    int stride_x = conv.wstride;
    int stride_y = conv.hstride;
    int dilation_x = conv.wdilation;
    int dilation_y = conv.hdilation;
    int activation = -1;
    int cpu_type = TYPE_A53;
    int pad_w0 = 0;
    int pad_w1 = 0;
    int pad_h0 = 0;
    int pad_h1 = 0;
    bool have_biases = bias.data != NULL;
    float* input_org = input.data;
    float* output_org = output.data;
    float* kernel_interleaved = weight.data;
    float* col = temp.data;
    float* biases = bias.data;

    int L2_CACHE_SIZE = (cpu_type == TYPE_A53) ? 512 * 1024 : 1024 * 1024;
    int kernel_size_l1 = kernel_size;
    int col_cnt_l2 = L2_CACHE_SIZE / 4 / kernel_size_l1 * 7 / 8;
    col_cnt_l2 = col_cnt_l2 > 4 ? (col_cnt_l2 & -4) : 4;
    int chan_16_num = output_chan / 16;

    long im2col_time = 0;
    long gemm_time = 0;
    /* one image per time */
    for(int i = 0; i < output_n; i++) {
        float* input = input_org + i * input_size * group;
        float* output = output_org + i * output_xy * output_chan * group;
        for(int g = 0; g < group; g++) {
            float* input_g = input + g * input_size; {
                TEngine::conv_fast::im2col(input_g, col, input_chan, input_w, input_h, kernel_x, kernel_y, stride_x, stride_y,
                        dilation_x, dilation_y, pad_w0, pad_w1, pad_h0, pad_h1, output_x, output_y, 0,
                        output_xy);
            }
            float* kernel_g = kernel_interleaved + g * (kernel_size * ((output_chan + 3) & -4));
            float* output_g = output + g * output_xy * output_chan;
            float* bias_g = biases + g * output_chan;

            int chan_4_num = (output_chan & 0xf) ? 1 : 0;
            int l2_loop = (output_xy - 1) / col_cnt_l2 + 1;
            int max_task_num = l2_loop * (chan_16_num + chan_4_num);
            // for input block of L2 cache size
            for(int col_i = 0; col_i < output_xy; col_i += col_cnt_l2) {
                int col_start = col_i;
                int col_end = col_i + col_cnt_l2;
                col_end = col_end > output_xy ? output_xy : col_end;

                TEngine::conv_fast::sgemm4x16(col, kernel_g, bias_g, have_biases, output_g, kernel_size, col_start, col_end, 0,
                        output_chan & -16, output_xy, activation, cpu_type);
                if(output_chan & 0xf)
                    TEngine::conv_fast::sgemm4x4(col, kernel_g, bias_g, have_biases, output_g, kernel_size, col_start, col_end,
                            output_chan & -16, output_chan, output_xy, activation, cpu_type);
            }
        }
    }
}
