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
#include <executor/operator/arm64/conv/winograd/conv_2d_wino_1.h>

#define TILE 4

void tengine_conv_2d_wino_1::init(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1CHW input(ins[0]); 
    L1CHW output(out); 
    L1VAB temp0(tmp[0]);
    L1VAB temp1(tmp[1]);
    L1VAB temp2(tmp[2]);
    const int extented_filter_h = conv.hdilation * (conv.hkernel - 1) + 1;
    const int extented_filter_w = conv.wdilation * (conv.wkernel - 1) + 1;
    const int output_xy = output.hw;
    const int kernel_size = input.c * conv.hkernel * conv.wkernel;
    output.c = conv.outch;
    output.h = (input.h - extented_filter_h) / conv.hstride + 1;
    output.w = (input.w - extented_filter_w) / conv.wstride + 1;

    int block_h = (output.h + TILE - 1) / TILE;
    int block_w = (output.w + TILE - 1) / TILE;
    int block_hw = block_h * block_w;
    int padded_inh = TILE * block_h + 2;
    int padded_inw = TILE * block_w + 2;
    int pad_inhw = padded_inh * padded_inw;

    int inp_padded_size = (input.c * pad_inhw + 2);

    temp0.u = 1;
    temp0.v = 1;
    temp0.a = inp_padded_size;

    temp1.u = 1;
    temp1.v = 1;
    temp1.a = ELEM_SIZE * input.c * block_hw + 32;

    temp2.u = 1;
    temp2.v = 1;
    temp2.a = ELEM_SIZE * output.c * block_hw + 32;
}
void tengine_conv_2d_wino_1::run(const Tensors& ins, Tensor& out, Tensors& tmp, Operator& op) {
    Conv conv(op);
    L1CHW output(out); 
    L1CHW input(ins[0]); 
    L1VAB weight(ins[1]); 
    L111W biast(ins[2]);
    L1VAB temp0(tmp[0]);
    L1VAB temp1(tmp[1]);
    L1VAB temp2(tmp[2]);

    int pad_h0 = 0;
    int pad_w0 = 0;
    int cpu_type = TYPE_A53;
    int activation = -1;

    float* input_org = input.data;
    int input_c = input.c;
    int input_h = input.h;
    int input_w = input.w;
    int inp_chw = input_c * input_h * input_w;
    float* output_org = output.data;
    int output_h = output.h;
    int output_w = output.w;
    int output_c = output.c;
    int out_hw = output_h * output_w;
    int out_chw = out_hw * output_c;
    int output_n = 1;

    int block_h = (output_h + TILE - 1) / TILE;
    int block_w = (output_w + TILE - 1) / TILE;
    int block_hw = block_h * block_w;
    int padded_inh = TILE * block_h + 2;
    int padded_inw = TILE * block_w + 2;
    int pad_inhw = padded_inh * padded_inw;

    int nn_block = block_hw / BLOCK_HW_UNIT;
    int resi_block = nn_block * BLOCK_HW_UNIT;
    int resi_w = block_w * TILE - output_w;
    int resi_h = block_h * TILE - output_h;

    float* kernel_interleaved = weight.data;
    float* input_padded = temp0.data;
    float* trans_inp = temp1.data;
    float* trans_out = temp2.data;

    float* bias = biast.data;
    int bias_term = biast.data != NULL;


    int block_4 = (block_hw+3)/4;
    int L2_CACHE_SIZE = (cpu_type == TYPE_A53) ? 512 * 1024 : 1024 * 1024;
    int L2_n = L2_CACHE_SIZE * 0.3 / (ELEM_SIZE * input_c * sizeof(float));
    L2_n = L2_n > 16 ? (L2_n & -16) : 16;

    int cout_count16 = output_c/16;
    int cout_nn16 =  cout_count16*16;

    for(int n = 0; n < output_n; n++)
    {
        float* input = input_org + n * inp_chw;
        float* output = output_org + n * out_chw;
        // pad_trans_interleave_inp


        pad_input1(input, input_padded, input_c, input_h, input_w, padded_inh, padded_inw, pad_h0, pad_w0);
        tran_input_4block_1(input_padded, trans_inp, input_c, 0, nn_block, block_w, pad_inhw, padded_inw, block_hw);
        if(resi_block != block_hw)
            tran_input_resi_block_1(input_padded, trans_inp, input_c, nn_block, resi_block, block_hw, block_w, pad_inhw,
                    padded_inw);
        for(int s = 0; s < ELEM_SIZE; s++)
        {
            wino_sgemm_4x16_1(kernel_interleaved, trans_inp, trans_out, input_c, cpu_type, 0, cout_nn16, 0, block_hw,
                    block_hw, output_c, s);
            if(cout_nn16!=output_c)
            {
                wino_sgemm_4x4_1(kernel_interleaved, trans_inp, trans_out, input_c, cpu_type, cout_nn16, output_c, 0, block_hw,
                        block_hw, output_c, s);
            }
        }
        trans_output(trans_out, output, bias, bias_term, block_h, block_w, 0, output_c, out_hw, output_w, resi_h,
                resi_w, activation);

    }
}
