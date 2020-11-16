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
 * Author: haoluo@openailab.com
 */
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <arm_neon.h>

#ifdef CONFIG_AUTH_DEVICE
#include "auth_nodeops.hpp"
#endif

#include <math.h>
namespace TEngine {

namespace conv_2d_direct_3x3_dilation {

static void vector_set_value(float* buf, int size, float value)
{
    float32x4_t value_4 = vdupq_n_f32(value);
    int i = 0;
    for(i = 0; i + 3 < size; i +=4)
    {
        vst1q_f32(buf + i, value_4);
    }
    for(;i < size;i++)
        buf[i] = value;
}

static void vector_activation(float* buf, int size, int type)
{
    if(type == 0)
    {
        float32x4_t zero = vdupq_n_f32(0.0);
        float32x4_t max = vdupq_n_f32((float)type);
        int i = 0;
        for(i = 0; i + 3 < size; i +=4)
        {
            float32x4_t value_4 = vld1q_f32(buf + i);
            value_4 = vmaxq_f32(value_4, zero);
            if(type > 0)
            {
                value_4 = vminq_f32(value_4, max);
                
            }
            vst1q_f32(buf + i, value_4);
        }
        for(;i < size;i++)
        {
            float value = buf[i];
            value = value > 0? value:0.f;
            if(type > 0)
                value = value > type? (float)type : value;
            buf[i] = value;
        }

    }
}

void DirectConv(float* input_buf, float* weight_buf, float* bias, float* output_buf, int input_h,
                                     int input_w, int input_c, int output_c, int pad, int activation)
{
    int channel_size = input_h * input_w;
    int mid_w = input_h - pad * 2;
    int mid_block_end = (mid_w & -4) + pad;
    int mid_end = mid_w + pad;
    int h = 0, w = 0;
    for(int c = 0; c < output_c; c++)
    {
        float* output_buf_c = output_buf + c * channel_size;
        float bias_c = bias ? bias[c] : 0;
        vector_set_value(output_buf_c, channel_size, bias_c);
        
        for(int inc = 0; inc < input_c; inc++)
        {
            float* input_buf_c = input_buf + inc * channel_size;
            float* weight_buf_c = weight_buf + (c * input_c + inc) * 9;
            float32x4_t kernel_0 = vdupq_n_f32(weight_buf_c[0]);
            float32x4_t kernel_1 = vdupq_n_f32(weight_buf_c[1]);
            float32x4_t kernel_2 = vdupq_n_f32(weight_buf_c[2]);
            float32x4_t kernel_3 = vdupq_n_f32(weight_buf_c[3]);
            float32x4_t kernel_4 = vdupq_n_f32(weight_buf_c[4]);
            float32x4_t kernel_5 = vdupq_n_f32(weight_buf_c[5]);
            float32x4_t kernel_6 = vdupq_n_f32(weight_buf_c[6]);
            float32x4_t kernel_7 = vdupq_n_f32(weight_buf_c[7]);
            float32x4_t kernel_8 = vdupq_n_f32(weight_buf_c[8]);
            for(h = 0; h < pad; h++)
            {
                for(w = 0; w < pad; w++)
                {
                    float tmp = weight_buf_c[4] * input_buf_c[h * input_w + w];
                    tmp += weight_buf_c[5] * input_buf_c[h * input_w + w + pad];
                    tmp += weight_buf_c[7] * input_buf_c[(h + pad) * input_w + w];
                    tmp += weight_buf_c[8] * input_buf_c[(h + pad) * input_w + w + pad];
                    output_buf_c[h * input_w + w] += tmp;
                    // if(h==0 && w==0)
                        // printf("output[%d]= %f\n",c, output_buf_c[0]);
                }
                for(; w < mid_block_end; w += 4)
                {
                    float32x4_t out_4 = vld1q_f32(output_buf_c + h * input_w + w);
                    out_4 = vmlaq_f32(out_4, kernel_3, vld1q_f32(input_buf_c + h * input_w + w - pad));
                    out_4 = vmlaq_f32(out_4, kernel_4, vld1q_f32(input_buf_c + h * input_w + w));
                    out_4 = vmlaq_f32(out_4, kernel_5, vld1q_f32(input_buf_c + h * input_w + w + pad));
                    out_4 = vmlaq_f32(out_4, kernel_6, vld1q_f32(input_buf_c + (h + pad) * input_w + w - pad));
                    out_4 = vmlaq_f32(out_4, kernel_7, vld1q_f32(input_buf_c + (h + pad) * input_w + w));
                    out_4 = vmlaq_f32(out_4, kernel_8, vld1q_f32(input_buf_c + (h + pad) * input_w + w + pad));
                    vst1q_f32(output_buf_c + h * input_w + w, out_4);
                }
                for(; w < mid_end; w++)
                {
                    float tmp = weight_buf_c[3] * input_buf_c[h * input_w + w - pad];
                    tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                    tmp += weight_buf_c[5] * input_buf_c[h * input_w + w + pad];
                    tmp += weight_buf_c[6] * input_buf_c[(h + pad) * input_w + w - pad];
                    tmp += weight_buf_c[7] * input_buf_c[(h + pad) * input_w + w];
                    tmp += weight_buf_c[8] * input_buf_c[(h + pad) * input_w + w + pad];
                    output_buf_c[h * input_w + w] += tmp;
                }
                for(; w < input_w; w++)
                {
                    float tmp = weight_buf_c[3] * input_buf_c[h * input_w + w - pad];
                    tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                    tmp += weight_buf_c[6] * input_buf_c[(h + pad) * input_w + w - pad];
                    tmp += weight_buf_c[7] * input_buf_c[(h + pad) * input_w + w];
                    output_buf_c[h * input_w + w] += tmp;
                }
            }
            for(; h < input_h - pad; h++)
            {
                for(w = 0; w < pad; w++)
                {
                    float tmp = weight_buf_c[1] * input_buf_c[(h - pad) * input_w + w];
                    tmp += weight_buf_c[2] * input_buf_c[(h - pad) * input_w + w + pad];
                    tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                    tmp += weight_buf_c[5] * input_buf_c[h * input_w + w + pad];
                    tmp += weight_buf_c[7] * input_buf_c[(h + pad) * input_w + w];
                    tmp += weight_buf_c[8] * input_buf_c[(h + pad) * input_w + w + pad];
                    output_buf_c[h * input_w + w] += tmp;
                }
                for(; w < mid_block_end; w += 4)
                {
                    float32x4_t out_4 = vld1q_f32(output_buf_c + h * input_w + w);
                    out_4 = vmlaq_f32(out_4, kernel_0, vld1q_f32(input_buf_c + (h - pad) * input_w + w - pad));
                    out_4 = vmlaq_f32(out_4, kernel_1, vld1q_f32(input_buf_c + (h - pad) * input_w + w));
                    out_4 = vmlaq_f32(out_4, kernel_2, vld1q_f32(input_buf_c + (h - pad) * input_w + w + pad));
                    out_4 = vmlaq_f32(out_4, kernel_3, vld1q_f32(input_buf_c + h * input_w + w - pad));
                    out_4 = vmlaq_f32(out_4, kernel_4, vld1q_f32(input_buf_c + h * input_w + w));
                    out_4 = vmlaq_f32(out_4, kernel_5, vld1q_f32(input_buf_c + h * input_w + w + pad));
                    out_4 = vmlaq_f32(out_4, kernel_6, vld1q_f32(input_buf_c + (h + pad) * input_w + w - pad));
                    out_4 = vmlaq_f32(out_4, kernel_7, vld1q_f32(input_buf_c + (h + pad) * input_w + w));
                    out_4 = vmlaq_f32(out_4, kernel_8, vld1q_f32(input_buf_c + (h + pad) * input_w + w + pad));
                    vst1q_f32(output_buf_c + h * input_w + w, out_4);
                }
                for(; w < mid_end; w++)
                {
                    float tmp = weight_buf_c[0] * input_buf_c[(h - pad) * input_w + w - pad];
                    tmp += weight_buf_c[1] * input_buf_c[(h - pad) * input_w + w];
                    tmp += weight_buf_c[2] * input_buf_c[(h - pad) * input_w + w + pad];
                    tmp += weight_buf_c[3] * input_buf_c[h * input_w + w - pad];
                    tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                    tmp += weight_buf_c[5] * input_buf_c[h * input_w + w + pad];
                    tmp += weight_buf_c[6] * input_buf_c[(h + pad) * input_w + w - pad];
                    tmp += weight_buf_c[7] * input_buf_c[(h + pad) * input_w + w];
                    tmp += weight_buf_c[8] * input_buf_c[(h + pad) * input_w + w + pad];
                    output_buf_c[h * input_w + w] += tmp;
                }
                for(; w < input_w; w++)
                {
                    float tmp = weight_buf_c[0] * input_buf_c[(h - pad) * input_w + w - pad];
                    tmp += weight_buf_c[1] * input_buf_c[(h - pad) * input_w + w];
                    tmp += weight_buf_c[3] * input_buf_c[h * input_w + w - pad];
                    tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                    tmp += weight_buf_c[6] * input_buf_c[(h + pad) * input_w + w - pad];
                    tmp += weight_buf_c[7] * input_buf_c[(h + pad) * input_w + w];
                    output_buf_c[h * input_w + w] += tmp;
                }
            }
            for(; h < input_h; h++)
            {
                for(w = 0; w < pad; w++)
                {
                    float tmp = weight_buf_c[1] * input_buf_c[(h - pad) * input_w + w];
                    tmp += weight_buf_c[2] * input_buf_c[(h - pad) * input_w + w + pad];
                    tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                    tmp += weight_buf_c[5] * input_buf_c[h * input_w + w + pad];
                    output_buf_c[h * input_w + w] += tmp;
                }
                for(; w < mid_block_end; w += 4)
                {
                    float32x4_t out_4 = vld1q_f32(output_buf_c + h * input_w + w);
                    out_4 = vmlaq_f32(out_4, kernel_0, vld1q_f32(input_buf_c + (h - pad) * input_w + w - pad));
                    out_4 = vmlaq_f32(out_4, kernel_1, vld1q_f32(input_buf_c + (h - pad) * input_w + w));
                    out_4 = vmlaq_f32(out_4, kernel_2, vld1q_f32(input_buf_c + (h - pad) * input_w + w + pad));
                    out_4 = vmlaq_f32(out_4, kernel_3, vld1q_f32(input_buf_c + h * input_w + w - pad));
                    out_4 = vmlaq_f32(out_4, kernel_4, vld1q_f32(input_buf_c + h * input_w + w));
                    out_4 = vmlaq_f32(out_4, kernel_5, vld1q_f32(input_buf_c + h * input_w + w + pad));
                    vst1q_f32(output_buf_c + h * input_w + w, out_4);
                }
                for(; w < mid_end; w++)
                {
                    float tmp = weight_buf_c[0] * input_buf_c[(h - pad) * input_w + w - pad];
                    tmp += weight_buf_c[1] * input_buf_c[(h - pad) * input_w + w];
                    tmp += weight_buf_c[2] * input_buf_c[(h - pad) * input_w + w + pad];
                    tmp += weight_buf_c[3] * input_buf_c[h * input_w + w - pad];
                    tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                    tmp += weight_buf_c[5] * input_buf_c[h * input_w + w + pad];
                    output_buf_c[h * input_w + w] += tmp;
                }
                for(; w < input_w; w++)
                {
                    float tmp = weight_buf_c[0] * input_buf_c[(h - pad) * input_w + w - pad];
                    tmp += weight_buf_c[1] * input_buf_c[(h - pad) * input_w + w];
                    tmp += weight_buf_c[3] * input_buf_c[h * input_w + w - pad];
                    tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                    output_buf_c[h * input_w + w] += tmp;
                }
            }
        }
        vector_activation(output_buf_c, channel_size, activation);
    }
}
}    // namespace conv_2d_direct_3x3_dilation
}    // namespace TEngine
