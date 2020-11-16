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

#include <iostream>
#include <cstring>
#include <cstdlib>

#ifdef CONFIG_AUTH_DEVICE
#include "auth_nodeops.hpp"
#endif

#define CONV_DW_MAX(a, b) ((a) > (b) ? (a) : (b))
#define CONV_DW_MIN(a, b) ((a) < (b) ? (a) : (b))


namespace TEngine {

namespace conv_2d_dw {

inline float do_activation(float input, int activation)
{
    if(activation == 0)
    {
        input = CONV_DW_MAX(input, 0);
        if(activation == 6)
            input =CONV_DW_MIN(input, 6);
    }
    return input;
}

void initial_output(float* output, float* bias, int output_ch, int output_wh)
{
    int i, j;
    // no bias
    if(bias == nullptr)
    {
        memset(output, 0.f, output_ch * output_wh* sizeof(float));
    }
    else
    {
        float* out_ptr= output;
        for(i = 0; i < output_ch; i++)
            for(j = 0; j < output_wh; j++)
                *out_ptr++ = bias[i];
    }
}
void conv_dw_genreal_kernel(const float *input, const float *kernel,float *output, 
                            int group_start,int group_end, int activation,
                            int input_c, int input_h, int input_w,
                            int output_c, int output_h, int output_w,
                            int kernel_h, int kernel_w,
                            int pad_h, int pad_w, int stride_h, int stride_w,
                            int dilation_h, int dilation_w)
{
    int c, h, w, kc, k_h, k_w;
    int input_offset = 0;
    int kernel_offset = 0;
    int output_offset = 0;
    int kernel_size = input_c * kernel_h * kernel_w;
    int out_hw = output_w * output_h;

    for(int g= group_start; g < group_end; g++)
    {
        for (c = 0; c < output_c; c++)
        {
            for (h = 0; h < output_h; h++)
            {
                for (w = 0; w < output_w; w++)
                {
                    const int h_start = (h * stride_h) - pad_h;
                    const int w_start = (w * stride_w) - pad_w;
                    float total = 0.f;
                    output_offset = (g * output_c  + c) * out_hw + h * output_w + w;
                    for (kc = 0; kc < input_c; kc++)
                    {
                        for (k_h = 0; k_h < kernel_h; k_h++)
                        {
                            for (k_w = 0; k_w < kernel_w; k_w++)
                            {
                                const int cur_y = h_start + dilation_h * k_h;
                                const int cur_x = w_start + dilation_w * k_w;
                                if((cur_x >= 0) && (cur_x < input_w) && (cur_y >= 0) && (cur_y < input_h))
                                {
                                    input_offset = (g * input_c + kc)* input_h * input_w + cur_y * input_w + cur_x;
                                    kernel_offset = (g * output_c + c) * kernel_size + kc * kernel_h* kernel_w +
                                                    k_h * kernel_w + k_w;
                                    total += (input[input_offset] * kernel[kernel_offset]);
                                }
                            }
                        }
                    }
                    output[output_offset] = do_activation(output[output_offset]+total, activation);
                }
            }
        }
    }
}

}    // namespace conv_2d_dw
}    // namespace TEngine
