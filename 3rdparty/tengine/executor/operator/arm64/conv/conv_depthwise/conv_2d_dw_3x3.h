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
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 */
#include <cstdlib>
#include <cstring>
#include <iostream>

#ifdef CONFIG_AUTH_DEVICE
#include "auth_nodeops.hpp"
#endif

#include <math.h>
namespace TEngine {

namespace conv_2d_dw_3x3 {

#define TYPE_A53 0
#define TYPE_A72 1

#include "conv_2d_dw_3x3_kernel/A72.inl"

extern "C" void dw_k3s2p0(float* data, int h, int w, float* kernel, float* output, float* bias, int out_w, int act);
extern "C" void dw_k3s2p0p1(float* data, int h, int w, float* kernel, float* output, float* bias, int out_w, int act);

void DirectConv(float* input_buf, int input_h, int input_w, float* output_buf, int output_h, int output_w,
                             float* weight_buf, int channel_num, int stride, float* bias, int* pads, int cpu_type,
                             int activation)
{
    int channel_size = input_h * input_w;
    float* bias_tmp = bias;

    int pad_h0 = pads[0];
    int pad_h1 = pads[2];

    {
        for(int i = 0; i < channel_num; i++)
        {
            if(bias)
                bias_tmp = bias + i;

            if(stride == 1)
            {
                dw_k3s1p1_a72(input_buf, input_h, input_w, weight_buf, output_buf, bias_tmp, activation);
            }
            else if(stride == 2)
            {
                if(pad_h0 == 0)
                {
                    if(pad_h1 == 0)
                        dw_k3s2p0(input_buf, input_h, input_w, weight_buf, output_buf, bias_tmp, output_w, activation);
                    else
                        dw_k3s2p0p1(input_buf, input_h, input_w, weight_buf, output_buf, bias_tmp, output_w,
                                    activation);
                }
                else
                    dw_k3s2p1_a72(input_buf, input_h, input_w, weight_buf, output_buf, bias_tmp, activation);
            }

            input_buf += channel_size;
            output_buf += output_h * output_w;
            weight_buf += 9;
        }
    }
}

}    // namespace conv_2d_dw_3x3
}    // namespace TEngine
