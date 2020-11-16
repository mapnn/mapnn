/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2019, Open AI Lab
 * Author: chunyinglv@openailab.com
 */
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdio.h>
#include <sys/time.h>

#include "wino_trans_ker.h"
#include "wino_trans_inp.h"
#include "wino_sgemm.h"

#ifdef CONFIG_AUTH_DEVICE
#include "auth_nodeops.hpp"
#endif

namespace TEngine {

namespace conv_2d_wino_1 {

static void wino_trans_inp_kernel(const int i, const int tid, const void* step, int input_c, int cin_64,
                          const float* input, float* trans_inp, int block_w, int in_hw, int inw, int block_hw)
{
    int my_step = (( int* )step)[0];

    for(int idx = tid; idx < cin_64; idx += my_step)
    {
        int cin_start = idx * 64;
        int cin_end = cin_start + 64;
        cin_end = cin_end > input_c ? input_c : cin_end;

        tran_input_1(input, trans_inp, input_c, cin_start, cin_end, block_w, in_hw, inw, block_hw);
    }
}

static void wino_sgemm_kernel(const int i, const int tid,const void* step,float* kernel_interleaved,float* trans_inp,float* trans_out,
    int input_c,int cpu_type,int cout_nn16,int output_c,int block_hw)
{
    int my_step = ((int*)step)[0];
    for(int s = tid; s < ELEM_SIZE; s+=my_step)
    {
        s = (s<ELEM_SIZE)?(s):(ELEM_SIZE-1);
        wino_sgemm_4x16_1(kernel_interleaved, trans_inp, trans_out, input_c, cpu_type, 0, cout_nn16, 0, block_hw,
                            block_hw, output_c, s);
        if(cout_nn16!=output_c)
        {
            wino_sgemm_4x4_1(kernel_interleaved, trans_inp, trans_out, input_c, cpu_type, cout_nn16, output_c, 0, block_hw,
                            block_hw, output_c, s);
        }
    }
}

static void wino_trans_out_kernel(const int i, const int tid,const void* step, int cout_16,int output_c,
                    float* trans_out, float* output, float* bias, int bias_term, int block_h, int block_w,
                    int out_hw, int out_w, int resi_h, int resi_w,int activation)
{
    int my_step = ((int*)step)[0];

    for(int cout_idx = tid; cout_idx < cout_16; cout_idx += my_step)
    {
        int cout_start = cout_idx*16;
        int cout_end = cout_start + 16;
        cout_end = cout_end > output_c ? output_c : cout_end;

        trans_output(trans_out, output, bias, bias_term, block_h, block_w, 
                    cout_start, cout_end, 
                    out_hw, out_w, resi_h, resi_w, activation);
    }
}
}    // namespace conv_2d_wino_1
}    // namespace TEngine
