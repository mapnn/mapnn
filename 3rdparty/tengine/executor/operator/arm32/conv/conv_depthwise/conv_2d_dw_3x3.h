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
 * Copyright (c) 2018, Open AI Lab
 * Author: xiaowei@openailab.com
 */

#include <iostream>
#include <cstring>
#include <cstdlib>

#ifdef CONFIG_AUTH_DEVICE
#include "auth_nodeops.hpp"
#endif

namespace TEngine {

namespace conv_2d_dw_3x3 {

extern "C" void dw_k3s2(float* input, float* kernel, float* output, int channel, int width, int height, float* bias,
                        int pad0);
extern "C" void dw_k3s1p1(float* input, float* kernel, float* output, int channel, int width, int height, float* bias);
extern "C" void dw_k3s2_relu_fused(float* input, float* kernel, float* output, int channel, int width, int height,
                                   float* bias, int pad0);
extern "C" void dw_k3s2_relu6_fused(float* input, float* kernel, float* output, int channel, int width, int height,
                                    float* bias, int pad0);
extern "C" void dw_k3s1p1_relu_fused(float* input, float* kernel, float* output, int channel, int width, int height,
                                     float* bias);
extern "C" void dw_k3s1p1_relu6_fused(float* input, float* kernel, float* output, int channel, int width, int height,
                                      float* bias);


void DirectConv(float* input_buf, int input_h, int input_w, float* output_buf, int output_h, int output_w,
                             float* weight_buf, int channel_num, int stride, float* bias, int* pads, int activation)
{
    int pad_h0 = pads[0];

    if(stride == 1)
    {
        if(activation >= 0)
        {
            if(activation == 0)
                dw_k3s1p1_relu_fused(input_buf, weight_buf, output_buf, channel_num, input_w, input_h, bias);
            else
                dw_k3s1p1_relu6_fused(input_buf, weight_buf, output_buf, channel_num, input_w, input_h, bias);
        }
        else
        {
            dw_k3s1p1(input_buf, weight_buf, output_buf, channel_num, input_w, input_h, bias);
        }
    }
    else if(stride == 2)
    {
        if(activation >= 0)
        {
            if(activation == 0)
                dw_k3s2_relu_fused(input_buf, weight_buf, output_buf, channel_num, input_w, input_h, bias, pad_h0);
            else
                dw_k3s2_relu6_fused(input_buf, weight_buf, output_buf, channel_num, input_w, input_h, bias, pad_h0);
        }
        else
        {
            dw_k3s2(input_buf, weight_buf, output_buf, channel_num, input_w, input_h, bias, pad_h0);
        }
    }

    /* for relu6 */
    // if(activation>0)
    //{
    //     for(int i=0;i<output_h*output_w*channel_num;i++)
    //     {
    //       output_buf[i]=std::min(output_buf[i],6.0f);
    //     }
    //}
}

}    // namespace conv_2d_dw_3x3
}    // namespace TEngine
