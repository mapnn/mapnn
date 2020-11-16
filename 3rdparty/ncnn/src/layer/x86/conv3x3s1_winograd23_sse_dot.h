// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
// Copyright (C) 2019 BUG1989. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
//
#include "option.h"
#include "mat.h"
namespace ncnn{
static void conv3x3s1_winograd23_sse_dot(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Option& opt,
        int inch, int outch, int outh, int outw)
{
    // BEGIN dot
    Mat top_blob_tm = top_blob;;
    Mat bottom_blob_tm = bottom_blob;;
    {
        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        int nColBlocks = h_tm/4; // may be the block num in Feathercnn
        int nRowBlocks = w_tm/4;

        const int tiles = nColBlocks * nRowBlocks; 

        top_blob_tm.create(16, tiles, outch, 4u, opt.workspace_allocator);

        int nn_outch = outch >> 2;
        int remain_outch_start = nn_outch << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp=0; pp<nn_outch; pp++)
        {
            int p = pp * 4;

            Mat out0_tm = top_blob_tm.channel(p);
            Mat out1_tm = top_blob_tm.channel(p+1);
            Mat out2_tm = top_blob_tm.channel(p+2);
            Mat out3_tm = top_blob_tm.channel(p+3);

            const Mat kernel0_tm = kernel_tm.channel(p);
            const Mat kernel1_tm = kernel_tm.channel(p+1);
            const Mat kernel2_tm = kernel_tm.channel(p+2);
            const Mat kernel3_tm = kernel_tm.channel(p+3);

            for (int i=0; i<tiles; i++)
            {
                float* output0_tm = out0_tm.row(i);
                float* output1_tm = out1_tm.row(i);
                float* output2_tm = out2_tm.row(i);
                float* output3_tm = out3_tm.row(i);

#if __AVX__
                float zero_val = 0.f;

                __m256 _sum0 = _mm256_broadcast_ss(&zero_val);
                __m256 _sum0n = _mm256_broadcast_ss(&zero_val);
                __m256 _sum1 = _mm256_broadcast_ss(&zero_val);
                __m256 _sum1n = _mm256_broadcast_ss(&zero_val);
                __m256 _sum2 = _mm256_broadcast_ss(&zero_val);
                __m256 _sum2n = _mm256_broadcast_ss(&zero_val);
                __m256 _sum3 = _mm256_broadcast_ss(&zero_val);
                __m256 _sum3n = _mm256_broadcast_ss(&zero_val);

                int q = 0;

                for (; q+3<inch; q+=4)
                {    
                    const float* r0 = bottom_blob_tm.channel(q).row(i);
                    const float* r1 = bottom_blob_tm.channel(q+1).row(i);
                    const float* r2 = bottom_blob_tm.channel(q+2).row(i);
                    const float* r3 = bottom_blob_tm.channel(q+3).row(i);

                    const float* k0 = kernel0_tm.row(q);
                    const float* k1 = kernel1_tm.row(q);
                    const float* k2 = kernel2_tm.row(q);
                    const float* k3 = kernel3_tm.row(q);

                    __m256 _r0 = _mm256_loadu_ps(r0);
                    __m256 _r0n = _mm256_loadu_ps(r0+8);
                    // k0
                    __m256 _k0 = _mm256_loadu_ps(k0);
                    __m256 _k0n = _mm256_loadu_ps(k0+8);
                    __m256 _k1 = _mm256_loadu_ps(k1);
                    __m256 _k1n = _mm256_loadu_ps(k1+8);
                    __m256 _k2 = _mm256_loadu_ps(k2);
                    __m256 _k2n = _mm256_loadu_ps(k2+8);
                    __m256 _k3 = _mm256_loadu_ps(k3);
                    __m256 _k3n = _mm256_loadu_ps(k3+8);
                    _sum0 = _mm256_fmadd_ps(_r0, _k0, _sum0);
                    _sum0n = _mm256_fmadd_ps(_r0n, _k0n, _sum0n);
                    _sum1 = _mm256_fmadd_ps(_r0, _k1, _sum1);
                    _sum1n = _mm256_fmadd_ps(_r0n, _k1n, _sum1n);
                    _sum2 = _mm256_fmadd_ps(_r0, _k2, _sum2);
                    _sum2n = _mm256_fmadd_ps(_r0n, _k2n, _sum2n);
                    _sum3 = _mm256_fmadd_ps(_r0, _k3, _sum3);
                    _sum3n = _mm256_fmadd_ps(_r0n, _k3n, _sum3n);
                    
                    // k1
                    _r0 = _mm256_loadu_ps(r1);
                    _r0n = _mm256_loadu_ps(r1+8);                    
                    _k0 = _mm256_loadu_ps(k0+16);
                    _k0n = _mm256_loadu_ps(k0+24);
                    _k1 = _mm256_loadu_ps(k1+16);
                    _k1n = _mm256_loadu_ps(k1+24);
                    _k2 = _mm256_loadu_ps(k2+16);
                    _k2n = _mm256_loadu_ps(k2+24);
                    _k3 = _mm256_loadu_ps(k3+16);
                    _k3n = _mm256_loadu_ps(k3+24);           
                    _sum0 = _mm256_fmadd_ps(_r0, _k0, _sum0);
                    _sum0n = _mm256_fmadd_ps(_r0n, _k0n, _sum0n);
                    _sum1 = _mm256_fmadd_ps(_r0, _k1, _sum1);
                    _sum1n = _mm256_fmadd_ps(_r0n, _k1n, _sum1n);
                    _sum2 = _mm256_fmadd_ps(_r0, _k2, _sum2);
                    _sum2n = _mm256_fmadd_ps(_r0n, _k2n, _sum2n);
                    _sum3 = _mm256_fmadd_ps(_r0, _k3, _sum3);
                    _sum3n = _mm256_fmadd_ps(_r0n, _k3n, _sum3n);
                    // k2   
                    _r0 = _mm256_loadu_ps(r2);
                    _r0n = _mm256_loadu_ps(r2+8);                     
                    _k0 = _mm256_loadu_ps(k0+32);
                    _k0n = _mm256_loadu_ps(k0+40);
                    _k1 = _mm256_loadu_ps(k1+32);
                    _k1n = _mm256_loadu_ps(k1+40);
                    _k2 = _mm256_loadu_ps(k2+32);
                    _k2n = _mm256_loadu_ps(k2+40);
                    _k3 = _mm256_loadu_ps(k3+32);
                    _k3n = _mm256_loadu_ps(k3+40);
                    _sum0 = _mm256_fmadd_ps(_r0, _k0, _sum0);
                    _sum0n = _mm256_fmadd_ps(_r0n, _k0n, _sum0n);
                    _sum1 = _mm256_fmadd_ps(_r0, _k1, _sum1);
                    _sum1n = _mm256_fmadd_ps(_r0n, _k1n, _sum1n);
                    _sum2 = _mm256_fmadd_ps(_r0, _k2, _sum2);
                    _sum2n = _mm256_fmadd_ps(_r0n, _k2n, _sum2n);
                    _sum3 = _mm256_fmadd_ps(_r0, _k3, _sum3);
                    _sum3n = _mm256_fmadd_ps(_r0n, _k3n, _sum3n);
                    // k3   
                    _r0 = _mm256_loadu_ps(r3);
                    _r0n = _mm256_loadu_ps(r3+8);                     
                    _k0 = _mm256_loadu_ps(k0+48);
                    _k0n = _mm256_loadu_ps(k0+56);
                    _k1 = _mm256_loadu_ps(k1+48);
                    _k1n = _mm256_loadu_ps(k1+56);
                    _k2 = _mm256_loadu_ps(k2+48);
                    _k2n = _mm256_loadu_ps(k2+56);
                    _k3 = _mm256_loadu_ps(k3+48);
                    _k3n = _mm256_loadu_ps(k3+56);
                    _sum0 = _mm256_fmadd_ps(_r0, _k0, _sum0);
                    _sum0n = _mm256_fmadd_ps(_r0n, _k0n, _sum0n);
                    _sum1 = _mm256_fmadd_ps(_r0, _k1, _sum1);
                    _sum1n = _mm256_fmadd_ps(_r0n, _k1n, _sum1n);
                    _sum2 = _mm256_fmadd_ps(_r0, _k2, _sum2);
                    _sum2n = _mm256_fmadd_ps(_r0n, _k2n, _sum2n);
                    _sum3 = _mm256_fmadd_ps(_r0, _k3, _sum3);
                    _sum3n = _mm256_fmadd_ps(_r0n, _k3n, _sum3n);
                }

                for (; q<inch; q++)
                {
                    const float* r0 = bottom_blob_tm.channel(q).row(i);

                    const float* k0 = kernel0_tm.row(q);
                    const float* k1 = kernel1_tm.row(q);
                    const float* k2 = kernel2_tm.row(q);
                    const float* k3 = kernel3_tm.row(q);

                    __m256 _r0 = _mm256_loadu_ps(r0);
                    __m256 _r0n = _mm256_loadu_ps(r0+8);
                    __m256 _k0 = _mm256_loadu_ps(k0);
                    __m256 _k0n = _mm256_loadu_ps(k0+8);
                    __m256 _k1 = _mm256_loadu_ps(k1);
                    __m256 _k1n = _mm256_loadu_ps(k1+8);
                    __m256 _k2 = _mm256_loadu_ps(k2);
                    __m256 _k2n = _mm256_loadu_ps(k2+8);
                    __m256 _k3 = _mm256_loadu_ps(k3);
                    __m256 _k3n = _mm256_loadu_ps(k3+8);
                                        
                    _sum0 = _mm256_fmadd_ps(_r0, _k0, _sum0);
                    _sum0n = _mm256_fmadd_ps(_r0n, _k0n, _sum0n);
                    _sum1 = _mm256_fmadd_ps(_r0, _k1, _sum1);
                    _sum1n = _mm256_fmadd_ps(_r0n, _k1n, _sum1n);
                    _sum2 = _mm256_fmadd_ps(_r0, _k2, _sum2);
                    _sum2n = _mm256_fmadd_ps(_r0n, _k2n, _sum2n);
                    _sum3 = _mm256_fmadd_ps(_r0, _k3, _sum3);
                    _sum3n = _mm256_fmadd_ps(_r0n, _k3n, _sum3n);
                }

                _mm256_storeu_ps(output0_tm, _sum0);
                _mm256_storeu_ps(output0_tm+8, _sum0n);
                _mm256_storeu_ps(output1_tm, _sum1);
                _mm256_storeu_ps(output1_tm+8, _sum1n);
                _mm256_storeu_ps(output2_tm, _sum2);
                _mm256_storeu_ps(output2_tm+8, _sum2n);
                _mm256_storeu_ps(output3_tm, _sum3);
                _mm256_storeu_ps(output3_tm+8, _sum3n);
#else
                float sum0[16] = {0.0f};
                float sum1[16] = {0.0f};
                float sum2[16] = {0.0f};
                float sum3[16] = {0.0f};

                int q = 0;
                for (; q+3<inch; q+=4)
                {   
                    const float* r0 = bottom_blob_tm.channel(q).row(i);
                    const float* r1 = bottom_blob_tm.channel(q+1).row(i);
                    const float* r2 = bottom_blob_tm.channel(q+2).row(i);
                    const float* r3 = bottom_blob_tm.channel(q+3).row(i);

                    const float* k0 = kernel0_tm.row(q);
                    const float* k1 = kernel1_tm.row(q);
                    const float* k2 = kernel2_tm.row(q);
                    const float* k3 = kernel3_tm.row(q);

                    for (int n=0; n<16; n++)
                    {
                        sum0[n] += r0[n] * k0[n];
                        k0 += 16;
                        sum0[n] += r1[n] * k0[n];
                        k0 += 16;
                        sum0[n] += r2[n] * k0[n];
                        k0 += 16;
                        sum0[n] += r3[n] * k0[n];
                        k0 -= 16 * 3;

                        sum1[n] += r0[n] * k1[n];
                        k1 += 16;
                        sum1[n] += r1[n] * k1[n];
                        k1 += 16;
                        sum1[n] += r2[n] * k1[n];
                        k1 += 16;
                        sum1[n] += r3[n] * k1[n];
                        k1 -= 16 * 3;

                        sum2[n] += r0[n] * k2[n];
                        k2 += 16;
                        sum2[n] += r1[n] * k2[n];
                        k2 += 16;
                        sum2[n] += r2[n] * k2[n];
                        k2 += 16;
                        sum2[n] += r3[n] * k2[n];
                        k2 -= 16 * 3;

                        sum3[n] += r0[n] * k3[n];
                        k3 += 16;
                        sum3[n] += r1[n] * k3[n];
                        k3 += 16;
                        sum3[n] += r2[n] * k3[n];
                        k3 += 16;
                        sum3[n] += r3[n] * k3[n];
                        k3 -= 16 * 3;
                    }
                }

                for (; q<inch; q++)
                {
                    const float* r0 = bottom_blob_tm.channel(q).row(i);

                    const float* k0 = kernel0_tm.row(q);
                    const float* k1 = kernel1_tm.row(q);
                    const float* k2 = kernel2_tm.row(q);
                    const float* k3 = kernel3_tm.row(q);

                    for (int n=0; n<16; n++)
                    {
                        sum0[n] += r0[n] * k0[n];
                        sum1[n] += r0[n] * k1[n];
                        sum2[n] += r0[n] * k2[n];
                        sum3[n] += r0[n] * k3[n];
                    }
                }

                for (int n=0; n<16; n++)
                {
                    output0_tm[n] = sum0[n];
                    output1_tm[n] = sum1[n];
                    output2_tm[n] = sum2[n];
                    output3_tm[n] = sum3[n];
                }
#endif                
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=remain_outch_start; p<outch; p++)
        {
            Mat out0_tm = top_blob_tm.channel(p);
            const Mat kernel0_tm = kernel_tm.channel(p);

            for (int i=0; i<tiles; i++)
            {
                float* output0_tm = out0_tm.row(i);

                float sum0[16] = {0.0f};

                int q = 0;
                for (; q+3<inch; q+=4)
                {   
                    const float* r0 = bottom_blob_tm.channel(q).row(i);
                    const float* r1 = bottom_blob_tm.channel(q+1).row(i);
                    const float* r2 = bottom_blob_tm.channel(q+2).row(i);
                    const float* r3 = bottom_blob_tm.channel(q+3).row(i);

                    const float* k0 = kernel0_tm.row(q);
                    const float* k1 = kernel0_tm.row(q+1);
                    const float* k2 = kernel0_tm.row(q+2);
                    const float* k3 = kernel0_tm.row(q+3);

                    for (int n=0; n<16; n++)
                    {
                        sum0[n] += r0[n] * k0[n];
                        sum0[n] += r1[n] * k1[n];
                        sum0[n] += r2[n] * k2[n];
                        sum0[n] += r3[n] * k3[n];
                    }
                }

                for (; q<inch; q++)
                {
                    const float* r0 = bottom_blob_tm.channel(q).row(i);
                    const float* k0 = kernel0_tm.row(q);

                    for (int n=0; n<16; n++)
                    {
                        sum0[n] += r0[n] * k0[n];
                    }             
                }

                for (int n=0; n<16; n++)
                {
                    output0_tm[n] = sum0[n];
                }
            }
        }
    }
}
}
