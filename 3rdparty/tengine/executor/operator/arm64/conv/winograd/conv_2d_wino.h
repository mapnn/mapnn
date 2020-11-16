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

#ifdef CONFIG_AUTH_DEVICE
#include "auth_nodeops.hpp"
#endif

namespace TEngine {

namespace conv_2d_wino {

static void wino_func_kernel(const int i, const int tid,const void* step, int block_4,int block_hw,
        float*input_padded,int input_c,
        int block_w,int pad_inhw,int padded_inw,
        float* kernel_interleaved, float*output, float*bias,int bias_term,int cpu_type,int cout_nn16,int output_c,
        int block_h,int out_hw,int output_w,int resi_h,int resi_w,int activation)
{
    int my_step = ((int*)step)[0];
    float trans_inp[4*input_c*ELEM_SIZE];
    for(int block_idx = tid; block_idx < block_4; block_idx+=my_step)
    {
        
        int is_4block=1;

        int resi = block_hw - block_idx*4;
        if(resi >= 4)
        {
            tran_input_4block_func(input_padded, trans_inp, input_c, 
                            block_idx, 
                            block_w, pad_inhw, padded_inw);
        }
        else
        {
            is_4block=0;
            tran_input_resi_block_func(input_padded, trans_inp, input_c, 
                          block_idx*4, block_hw, block_w, pad_inhw,
                            padded_inw);
        }
        wino_sgemm_4x16_func(kernel_interleaved, trans_inp, output, bias, bias_term, input_c, cpu_type, 0, cout_nn16, 
                        is_4block, block_idx*4,resi, block_h, block_w, out_hw, output_w, resi_h, resi_w, activation);

        if(cout_nn16!=output_c)
        {
            wino_sgemm_4x4_func(kernel_interleaved, trans_inp, output, bias, bias_term, input_c, cpu_type, cout_nn16,output_c,
                        is_4block,block_idx*4,resi, block_h, block_w, out_hw, output_w, resi_h, resi_w, activation);
        }
    }
}


}    // namespace conv_2d_wino
}    // namespace TEngine
