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
 * Author: xiaowei@openailab.com
 */
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <math.h>
#include <arm_neon.h>

#ifdef CONFIG_AUTH_DEVICE
#include "auth_nodeops.hpp"
#endif


extern "C" void im2col_fp32_1x1(float* input, long input_xy, float* col, long col_cnt, long input_chan);
extern "C" void im2col_fp32_3x3(float* input, long input_x, long input_y, long input_chan, float* col, long stride);

#include "conv_2d_fast.h"
#include "conv_2d_fast_kernel/A72.inl"                                    

namespace TEngine {

namespace conv_fast {

#define TYPE_A53 0
#define TYPE_A72 1

void direct_k3s1p1_4x16(float* biases, float* input, float* kernel, float* output, int input_chan, int input_w,
                               int input_h, int activatioin, int cpu_type)
{
    {
        direct_k3s1p1_4x16_a72(biases, input, kernel, output, input_chan, input_w, input_h, activatioin);
    }
}

void direct_k1s1p0_4x16(float* input_data, float* output, float* kernel, float* bias, const int c_in,
                               const int hw, int k_start, int k_end, int bias_term, int activation, int cpu_type)
{
    // only support hw%4==0  cout%16==0
    float* bias_ptr = NULL;
    int block_hw = hw >> 2;

    // [c_out: block 16]

    {
        for(int i = k_start; i < k_end; i += 16)
        {
            float* out_ptr = output + i * hw;
            float* ker_ptr = kernel + i * c_in;
            float* inp_ptr = input_data;
            if(bias_term)
            {
                bias_ptr = bias + i;
            }

            for(int k = 0; k < block_hw; k++)
            {
                direct_k1s1p0_4x16_a72(bias_ptr, inp_ptr, ker_ptr, out_ptr, hw, c_in, activation);
                out_ptr += 4;
                inp_ptr += 4;
            }
        }
    }

}

void im2col(float* im, float* col, int input_chan, int input_x, int input_y, int kernel_x, int kernel_y, int stride_x,
            int stride_y, int dilation_x, int dilation_y, int pad_w0, int pad_w1, int pad_h0, int pad_h1, int output_x,
            int output_y, int col_start, int col_end)
{
    int kernel_size = kernel_x * kernel_y * input_chan;
    int input_xy = input_x * input_y;
    int pad_x = pad_w0;
    int pad_y = pad_h0;
    float* cur_col = col + col_start * kernel_size;
    bool is_1x1 = (kernel_x == 1) && (kernel_y == 1) && (stride_x == 1) && (stride_y == 1);
    bool is_dilation = (dilation_x != 1) || (dilation_y != 1);
    bool is_3x3 = (kernel_x == 3) && (kernel_y == 3) && (!is_dilation);
    bool is_3x3_dilation = (dilation_x != 1) && (dilation_x == dilation_y) && (stride_x == 1) && (stride_y == 1) &&
                           (dilation_x == pad_x) && (dilation_x == pad_y) && (kernel_x == 3) && (kernel_y == 3);
    int col_i, col_j, kch, ky, kx, i;
    int col_end3 = col_end & 0x3;

    if(is_1x1)
    {
#if 0
    // equivlent C code
    for(col_i = (col_start & -4); col_i < (col_end & -4) ; col_i+=4 ){
      for(col_j = 0; col_j < kernel_size ; col_j++ ){
        for(i = 0; i < 4; i++)
         * cur_col++ = *(im + input_xy * col_j + col_i + i);
      }
    }
#else
        {
            int col_cnt = (col_end & -4) - (col_start & -4);
            im2col_fp32_1x1(( float* )im + col_start, input_xy, cur_col, col_cnt, input_chan);
            cur_col += col_cnt * kernel_size;
            col_i = col_end & -4;
        }
#endif
        // final 4 input
        if(col_end3)
        {
            for(col_j = 0; col_j < kernel_size; col_j++)
                for(i = 0; i < 4; i++)
                {
                    if(i < col_end3)
                        *cur_col++ = *(im + input_xy * col_j + col_i + i);
                    else
                        *cur_col++ = 0.0;
                }
        }
    }
    else if(is_3x3)
    {
        bool is_pad0 = (pad_w0 == 0) && (pad_h0 == 0) && (pad_w1 == 0) && (pad_h1 == 0);
        for(col_i = (col_start & -4); col_i < (col_end & -4); col_i += 4)
        {
            cur_col = col + col_i * kernel_size;
            int imy0 = col_i / output_x;
            int imy3 = (col_i + 3) / output_x;
            int imx0 = col_i - imy0 * output_x;
            int imx3 = (col_i + 3) - imy3 * output_x;
            if((imy0 == imy3) &&
               (is_pad0 || (imy0 != 0 && imx0 != 0 && imy0 != (output_y - 1) && imx3 != (output_x - 1))))
            {
                float* l0 = im + (imy0 * stride_y - pad_y) * input_x + (imx0 * stride_x - pad_x);
#if 0
        // equalavant C code
        int stride_x2 = stride_x * 2;
        int stride_x3 = stride_x * 3;
        float * l1 = l0 + input_x;
        float * l2 = l0 + input_x * 2;
        for(i = 0; i < input_chan; i++){
          for(int j=0 ; j < 3; j++){
             cur_col[j*4+0]  = l0[j];
             cur_col[j*4+1]  = l0[j+stride_x];
             cur_col[j*4+2]  = l0[j+stride_x2];
             cur_col[j*4+3]  = l0[j+stride_x3];
             cur_col[j*4+12] = l1[j];
             cur_col[j*4+13] = l1[j+stride_x];
             cur_col[j*4+14] = l1[j+stride_x2];
             cur_col[j*4+15] = l1[j+stride_x3];
             cur_col[j*4+24] = l2[j];
             cur_col[j*4+25] = l2[j+stride_x];
             cur_col[j*4+26] = l2[j+stride_x2];
             cur_col[j*4+27] = l2[j+stride_x3];
          }
          cur_col += 36;
          l0 += input_xy;
          l1 += input_xy;
          l2 += input_xy;
        }
#else
                {
                    im2col_fp32_3x3(l0, input_x, input_y, input_chan, cur_col, stride_x);
                    cur_col += 4 * kernel_size;
                }
#endif
            }
            else
            {
                int cnt_y[4] = {imy0, (col_i + 1) / output_x, (col_i + 2) / output_x, imy3};
                int cnt_x[4] = {imx0, col_i - cnt_y[1] * output_x + 1, col_i - cnt_y[2] * output_x + 2, imx3};
                int imx_start[4] = {cnt_x[0] * stride_x - pad_x, cnt_x[1] * stride_x - pad_x,
                                    cnt_x[2] * stride_x - pad_x, cnt_x[3] * stride_x - pad_x};
                int imy_start[4] = {cnt_y[0] * stride_y - pad_y, cnt_y[1] * stride_y - pad_y,
                                    cnt_y[2] * stride_y - pad_y, cnt_y[3] * stride_y - pad_y};
                for(kch = 0; kch < input_chan; kch++)
                    for(ky = 0; ky < 3; ky++)
                        for(kx = 0; kx < 3; kx++)
                        {
                            int imx[4] = {imx_start[0] + kx, imx_start[1] + kx, imx_start[2] + kx, imx_start[3] + kx};
                            int imy[4] = {imy_start[0] + ky, imy_start[1] + ky, imy_start[2] + ky, imy_start[3] + ky};
                            for(i = 0; i < 4; i++)
                            {
                                if(imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                    *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                                else
                                    *cur_col++ = 0.0;
                            }
                        }
            }
        }
        // final 4 input
        if(col_end3)
        {
            int cnt_y[4] = {col_i / output_x, (col_i + 1) / output_x, (col_i + 2) / output_x, (col_i + 3) / output_x};
            int cnt_x[4] = {col_i - cnt_y[0] * output_x, col_i - cnt_y[1] * output_x + 1,
                            col_i - cnt_y[2] * output_x + 2, col_i - cnt_y[3] * output_x + 3};
            int imx_start[4] = {cnt_x[0] * stride_x - pad_x, cnt_x[1] * stride_x - pad_x, cnt_x[2] * stride_x - pad_x,
                                cnt_x[3] * stride_x - pad_x};
            int imy_start[4] = {cnt_y[0] * stride_y - pad_y, cnt_y[1] * stride_y - pad_y, cnt_y[2] * stride_y - pad_y,
                                cnt_y[3] * stride_y - pad_y};
            for(kch = 0; kch < input_chan; kch++)
                for(ky = 0; ky < 3; ky++)
                    for(kx = 0; kx < 3; kx++)
                    {
                        int imx[4] = {imx_start[0] + kx, imx_start[1] + kx, imx_start[2] + kx, imx_start[3] + kx};
                        int imy[4] = {imy_start[0] + ky, imy_start[1] + ky, imy_start[2] + ky, imy_start[3] + ky};
                        for(i = 0; i < 4; i++)
                        {
                            if((col_i + i) < col_end && imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 &&
                               imy[i] < input_y)
                                *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                            else
                                *cur_col++ = 0.0;
                        }
                    }
        }
    }
    else if(is_3x3_dilation)
    {
        for(col_i = (col_start & -4); col_i < (col_end & -4); col_i += 4)
        {
            cur_col = col + col_i * kernel_size;
            int imy0 = col_i / output_x;
            int imy3 = (col_i + 3) / output_x;
            int imx0 = col_i - imy0 * output_x;
            int imx3 = (col_i + 3) - imy3 * output_x;
            if(imy3 == imy0 && (imy0 >= pad_x) && (imx0 >= pad_x) && (imx3 < output_x - pad_x) &&
               (imy3 < output_y - pad_x))
            {
                for(i = 0; i < input_chan; i++)
                {
                    float* im_00 = im + i * input_xy + (imy0 - pad_x) * input_x + imx0 - pad_x;
                    float32x4_t col_4 = vld1q_f32(im_00);
                    vst1q_f32(cur_col, col_4);
                    cur_col += 4;
                    col_4 = vld1q_f32(im_00 + pad_x);
                    vst1q_f32(cur_col, col_4);
                    cur_col += 4;
                    col_4 = vld1q_f32(im_00 + pad_x * 2);
                    vst1q_f32(cur_col, col_4);
                    cur_col += 4;
                    im_00 += input_x * pad_x;
                    col_4 = vld1q_f32(im_00);
                    vst1q_f32(cur_col, col_4);
                    cur_col += 4;
                    col_4 = vld1q_f32(im_00 + pad_x);
                    vst1q_f32(cur_col, col_4);
                    cur_col += 4;
                    col_4 = vld1q_f32(im_00 + pad_x * 2);
                    vst1q_f32(cur_col, col_4);
                    cur_col += 4;
                    im_00 += input_x * pad_x;
                    col_4 = vld1q_f32(im_00);
                    vst1q_f32(cur_col, col_4);
                    cur_col += 4;
                    col_4 = vld1q_f32(im_00 + pad_x);
                    vst1q_f32(cur_col, col_4);
                    cur_col += 4;
                    col_4 = vld1q_f32(im_00 + pad_x * 2);
                    vst1q_f32(cur_col, col_4);
                    cur_col += 4;
                }
            }
            else
            {
                int cnt_y[4] = {col_i / output_x, (col_i + 1) / output_x, (col_i + 2) / output_x,
                                (col_i + 3) / output_x};
                int cnt_x[4] = {col_i - cnt_y[0] * output_x, col_i - cnt_y[1] * output_x + 1,
                                col_i - cnt_y[2] * output_x + 2, col_i - cnt_y[3] * output_x + 3};
                int imx_start[4] = {cnt_x[0] * stride_x - pad_x, cnt_x[1] * stride_x - pad_x,
                                    cnt_x[2] * stride_x - pad_x, cnt_x[3] * stride_x - pad_x};
                int imy_start[4] = {cnt_y[0] * stride_y - pad_y, cnt_y[1] * stride_y - pad_y,
                                    cnt_y[2] * stride_y - pad_y, cnt_y[3] * stride_y - pad_y};
                for(kch = 0; kch < input_chan; kch++)
                    for(ky = 0; ky < (kernel_y * dilation_y); ky += dilation_y)
                        for(kx = 0; kx < (kernel_x * dilation_x); kx += dilation_x)
                        {
                            int imx[4] = {imx_start[0] + kx, imx_start[1] + kx, imx_start[2] + kx, imx_start[3] + kx};
                            int imy[4] = {imy_start[0] + ky, imy_start[1] + ky, imy_start[2] + ky, imy_start[3] + ky};
                            for(i = 0; i < 4; i++)
                            {
                                if(imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                    *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                                else
                                    *cur_col++ = 0.0;
                            }
                        }
            }
        }
        if(col_end3)
        {
            int cnt_y[4] = {col_i / output_x, (col_i + 1) / output_x, (col_i + 2) / output_x, (col_i + 3) / output_x};
            int cnt_x[4] = {col_i - cnt_y[0] * output_x, col_i - cnt_y[1] * output_x + 1,
                            col_i - cnt_y[2] * output_x + 2, col_i - cnt_y[3] * output_x + 3};
            int imx_start[4] = {cnt_x[0] * stride_x - pad_x, cnt_x[1] * stride_x - pad_x, cnt_x[2] * stride_x - pad_x,
                                cnt_x[3] * stride_x - pad_x};
            int imy_start[4] = {cnt_y[0] * stride_y - pad_y, cnt_y[1] * stride_y - pad_y, cnt_y[2] * stride_y - pad_y,
                                cnt_y[3] * stride_y - pad_y};
            for(kch = 0; kch < input_chan; kch++)
                for(ky = 0; ky < (kernel_y * dilation_y); ky += dilation_y)
                    for(kx = 0; kx < (kernel_x * dilation_x); kx += dilation_x)
                    {
                        int imx[4] = {imx_start[0] + kx, imx_start[1] + kx, imx_start[2] + kx, imx_start[3] + kx};
                        int imy[4] = {imy_start[0] + ky, imy_start[1] + ky, imy_start[2] + ky, imy_start[3] + ky};
                        for(i = 0; i < 4; i++)
                        {
                            if((col_i + i) < col_end && imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 &&
                               imy[i] < input_y)
                                *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                            else
                                *cur_col++ = 0.0;
                        }
                    }
        }
    }
    else
    {    // for general cases
        for(col_i = (col_start & -4); col_i < (col_end & -4); col_i += 4)
        {
            int cnt_y[4] = {col_i / output_x, (col_i + 1) / output_x, (col_i + 2) / output_x, (col_i + 3) / output_x};
            int cnt_x[4] = {col_i - cnt_y[0] * output_x, col_i - cnt_y[1] * output_x + 1,
                            col_i - cnt_y[2] * output_x + 2, col_i - cnt_y[3] * output_x + 3};
            int imx_start[4] = {cnt_x[0] * stride_x - pad_x, cnt_x[1] * stride_x - pad_x, cnt_x[2] * stride_x - pad_x,
                                cnt_x[3] * stride_x - pad_x};
            int imy_start[4] = {cnt_y[0] * stride_y - pad_y, cnt_y[1] * stride_y - pad_y, cnt_y[2] * stride_y - pad_y,
                                cnt_y[3] * stride_y - pad_y};
            for(kch = 0; kch < input_chan; kch++)
                for(ky = 0; ky < (kernel_y * dilation_y); ky += dilation_y)
                    for(kx = 0; kx < (kernel_x * dilation_x); kx += dilation_x)
                    {
                        int imx[4] = {imx_start[0] + kx, imx_start[1] + kx, imx_start[2] + kx, imx_start[3] + kx};
                        int imy[4] = {imy_start[0] + ky, imy_start[1] + ky, imy_start[2] + ky, imy_start[3] + ky};
                        for(i = 0; i < 4; i++)
                        {
                            if(imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 && imy[i] < input_y)
                                *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                            else
                                *cur_col++ = 0.0;
                        }
                    }
        }
        // final 4 input
        if(col_end3)
        {
            int cnt_y[4] = {col_i / output_x, (col_i + 1) / output_x, (col_i + 2) / output_x, (col_i + 3) / output_x};
            int cnt_x[4] = {col_i - cnt_y[0] * output_x, col_i - cnt_y[1] * output_x + 1,
                            col_i - cnt_y[2] * output_x + 2, col_i - cnt_y[3] * output_x + 3};
            int imx_start[4] = {cnt_x[0] * stride_x - pad_x, cnt_x[1] * stride_x - pad_x, cnt_x[2] * stride_x - pad_x,
                                cnt_x[3] * stride_x - pad_x};
            int imy_start[4] = {cnt_y[0] * stride_y - pad_y, cnt_y[1] * stride_y - pad_y, cnt_y[2] * stride_y - pad_y,
                                cnt_y[3] * stride_y - pad_y};
            for(kch = 0; kch < input_chan; kch++)
                for(ky = 0; ky < (kernel_y * dilation_y); ky += dilation_y)
                    for(kx = 0; kx < (kernel_x * dilation_x); kx += dilation_x)
                    {
                        int imx[4] = {imx_start[0] + kx, imx_start[1] + kx, imx_start[2] + kx, imx_start[3] + kx};
                        int imy[4] = {imy_start[0] + ky, imy_start[1] + ky, imy_start[2] + ky, imy_start[3] + ky};
                        for(i = 0; i < 4; i++)
                        {
                            if((col_i + i) < col_end && imx[i] >= 0 && imx[i] < input_x && imy[i] >= 0 &&
                               imy[i] < input_y)
                                *cur_col++ = *(im + input_xy * kch + input_x * imy[i] + imx[i]);
                            else
                                *cur_col++ = 0.0;
                        }
                    }
        }
    }
}

// interleave 0 ~ (output_chan & -16) kernels with 16 in form of k[0-15][0],k[0-15][1],k[0-15][2]..
// interleave (output_chan & -16) ~ ((output_chan + 3) & -4) tail kernls with 4 in form of
// k[0-3][0],k[0-3][1],k[0-3][2]..
void interleave_kernel(float* kernel, float* kernel_interleaved, int kernel_chan, int kernel_size)
{
    int i, j;
    float *cur_kernel0, *cur_kernel1, *cur_kernel2, *cur_kernel3, *cur_kernel4, *cur_kernel5, *cur_kernel6,
        *cur_kernel7;
    float *cur_kernel8, *cur_kernel9, *cur_kernel10, *cur_kernel11, *cur_kernel12, *cur_kernel13, *cur_kernel14,
        *cur_kernel15;
    float* cur_kernel_interleaved = kernel_interleaved;

    // interleave 16 kernels
    for(i = 0; i < (kernel_chan & -16); i += 16)
    {
        cur_kernel0 = kernel + kernel_size * i;
        cur_kernel1 = kernel + kernel_size * (i + 1);
        cur_kernel2 = kernel + kernel_size * (i + 2);
        cur_kernel3 = kernel + kernel_size * (i + 3);
        cur_kernel4 = kernel + kernel_size * (i + 4);
        cur_kernel5 = kernel + kernel_size * (i + 5);
        cur_kernel6 = kernel + kernel_size * (i + 6);
        cur_kernel7 = kernel + kernel_size * (i + 7);
        cur_kernel8 = kernel + kernel_size * (i + 8);
        cur_kernel9 = kernel + kernel_size * (i + 9);
        cur_kernel10 = kernel + kernel_size * (i + 10);
        cur_kernel11 = kernel + kernel_size * (i + 11);
        cur_kernel12 = kernel + kernel_size * (i + 12);
        cur_kernel13 = kernel + kernel_size * (i + 13);
        cur_kernel14 = kernel + kernel_size * (i + 14);
        cur_kernel15 = kernel + kernel_size * (i + 15);
        for(j = 0; j < kernel_size; j++)
        {
            *(cur_kernel_interleaved++) = cur_kernel0[j];
            *(cur_kernel_interleaved++) = cur_kernel1[j];
            *(cur_kernel_interleaved++) = cur_kernel2[j];
            *(cur_kernel_interleaved++) = cur_kernel3[j];
            *(cur_kernel_interleaved++) = cur_kernel4[j];
            *(cur_kernel_interleaved++) = cur_kernel5[j];
            *(cur_kernel_interleaved++) = cur_kernel6[j];
            *(cur_kernel_interleaved++) = cur_kernel7[j];
            *(cur_kernel_interleaved++) = cur_kernel8[j];
            *(cur_kernel_interleaved++) = cur_kernel9[j];
            *(cur_kernel_interleaved++) = cur_kernel10[j];
            *(cur_kernel_interleaved++) = cur_kernel11[j];
            *(cur_kernel_interleaved++) = cur_kernel12[j];
            *(cur_kernel_interleaved++) = cur_kernel13[j];
            *(cur_kernel_interleaved++) = cur_kernel14[j];
            *(cur_kernel_interleaved++) = cur_kernel15[j];
        }
    }

    for(i = (kernel_chan & -16); i < (kernel_chan & -4); i += 4)
    {
        cur_kernel0 = kernel + kernel_size * i;
        cur_kernel1 = kernel + kernel_size * (i + 1);
        cur_kernel2 = kernel + kernel_size * (i + 2);
        cur_kernel3 = kernel + kernel_size * (i + 3);
        for(j = 0; j < kernel_size; j++)
        {
            *(cur_kernel_interleaved++) = cur_kernel0[j];
            *(cur_kernel_interleaved++) = cur_kernel1[j];
            *(cur_kernel_interleaved++) = cur_kernel2[j];
            *(cur_kernel_interleaved++) = cur_kernel3[j];
        }
    }
    // last 4 kernel
    cur_kernel0 = kernel + kernel_size * i;
    cur_kernel1 = kernel + kernel_size * (i + 1);
    cur_kernel2 = kernel + kernel_size * (i + 2);
    if((kernel_chan & 0x3) == 3)
    {
        for(j = 0; j < kernel_size; j++)
        {
            *(cur_kernel_interleaved++) = cur_kernel0[j];
            *(cur_kernel_interleaved++) = cur_kernel1[j];
            *(cur_kernel_interleaved++) = cur_kernel2[j];
            *(cur_kernel_interleaved++) = 0.0;
        }
    }
    else if((kernel_chan & 0x3) == 2)
    {
        for(j = 0; j < kernel_size; j++)
        {
            *(cur_kernel_interleaved++) = cur_kernel0[j];
            *(cur_kernel_interleaved++) = cur_kernel1[j];
            *(cur_kernel_interleaved++) = 0.0;
            *(cur_kernel_interleaved++) = 0.0;
        }
    }
    else if((kernel_chan & 0x3) == 1)
    {
        for(j = 0; j < kernel_size; j++)
        {
            *(cur_kernel_interleaved++) = cur_kernel0[j];
            *(cur_kernel_interleaved++) = 0.0;
            *(cur_kernel_interleaved++) = 0.0;
            *(cur_kernel_interleaved++) = 0.0;
        }
    }

    return;
}

void sgemm4x16(float* col, float* kernel, float* biases, bool bias_term, float* output, int kernel_size,
                      int col_start, int col_end, int kernel_start, int kernel_end, int output_xy, int activation,
                      int cpu_type)
{
    float result[64];
    float* cur_biases = nullptr;
    float *cur_col, *cur_kernel, *cur_output;
    int col_line, kernel_num;
    int i, j;
    int col_end3 = col_end & 0x3;

    for(kernel_num = (kernel_start & -16); kernel_num < (kernel_end & -16); kernel_num += 16)
    {
        if(bias_term)
            cur_biases = ( float* )(biases + kernel_num);
        cur_kernel = ( float* )(kernel + kernel_num * kernel_size);
        cur_output = ( float* )(output + kernel_num * output_xy);
        for(col_line = (col_start & -4); col_line < (col_end & -4); col_line += 4)
        {
            cur_col = ( float* )(col + col_line * kernel_size);
            if(cpu_type == TYPE_A72 || 1)
                sgemm_4x16_a72(cur_biases, cur_col, cur_kernel, kernel_size, cur_output + col_line, output_xy,
                               activation, 0);

        }
        if(col_end3)
        {
            cur_col = ( float* )(col + col_line * kernel_size);
            if(cpu_type == TYPE_A72 || 1)
                sgemm_4x16_a72(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, activation, 0);
            for(i = 0; i < 16; i++)
                for(j = 0; j < (col_end3); j++)
                    *(output + (kernel_num + i) * output_xy + col_line + j) = result[(i << 2) + j];
        }
    }

    return;
}

void sgemm4x4(float* col, float* kernel, float* biases, bool bias_term, float* output, int kernel_size,
                     int col_start, int col_end, int kernel_start, int kernel_end, int output_xy, int activation,
                     int cpu_type)
{
    float result[16];
    float* cur_biases = nullptr;
    int col_line, kernel_num;
    float *cur_col, *cur_kernel, *cur_output;
    int i, j;
    int col_end3 = col_end & 0x3;
    int kernel_end3 = kernel_end & 0x3;

    for(kernel_num = kernel_start & -4; kernel_num < (kernel_end & -4); kernel_num += 4)
    {
        if(bias_term)
            cur_biases = ( float* )(biases + kernel_num);
        cur_kernel = ( float* )(kernel + kernel_num * kernel_size);
        cur_output = ( float* )(output + kernel_num * output_xy);
        for(col_line = (col_start & -4); col_line < (col_end & -4); col_line += 4)
        {
            cur_col = ( float* )(col + col_line * kernel_size);
            if(cpu_type == TYPE_A72 || 1)
                sgemm_4x4_a72(cur_biases, cur_col, cur_kernel, kernel_size, cur_output + col_line, output_xy,
                              activation, 0);
        }
        if(col_end3)
        {
            cur_col = ( float* )(col + col_line * kernel_size);
            if(cpu_type == TYPE_A72 || 1)
                sgemm_4x4_a72(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, activation, 0);
            for(i = 0; i < 4; i++)
            {
                for(j = 0; j < (col_end3); j++)
                    *(output + (kernel_num + i) * output_xy + col_line + j) = result[(i << 2) + j];
            }
        }
    }
    if(kernel_end3)
    {
        if(bias_term)
            cur_biases = ( float* )(biases + kernel_num);
        cur_kernel = ( float* )(kernel + kernel_num * kernel_size);
        for(col_line = (col_start & -4); col_line < (col_end & -4); col_line += 4)
        {
            cur_col = ( float* )(col + col_line * kernel_size);
            if(cpu_type == TYPE_A72 || 1)
                sgemm_4x4_a72(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, activation, 0);
            for(i = 0; i < kernel_end3; i++)
                for(j = 0; j < 4; j++)
                    *(output + (kernel_num + i) * output_xy + col_line + j) = result[(i << 2) + j];
        }
        if(col_end3)
        {
            cur_col = ( float* )(col + col_line * kernel_size);
            if(cpu_type == TYPE_A72 || 1)
                sgemm_4x4_a72(cur_biases, cur_col, cur_kernel, kernel_size, result, 4, activation, 0);
            for(i = 0; i < (kernel_end3); i++)
            {
                for(j = 0; j < (col_end3); j++)
                    *(output + (kernel_num + i) * output_xy + col_line + j) = result[(i << 2) + j];
            }
        }
    }
    return;
}


void conv3x3s1_neon(float* bottom_blob_transform, float* top_blob, float* kernel_interleave, float* bias_data, int in_w,
                    int in_h, int in_c, int out_w, int out_h, int out_c, int activation);

}    // namespace conv_fast
}    // namespace TEngine
